"""Motion forecasting metrics, following the Argoverse 2 challenge definitions.

For each scenario the K most probable predicted trajectories are kept, and the best of them is the
one with the lowest endpoint error (the Argoverse 1 and 2 convention). Then:
    minADE        average displacement error of the best trajectory
    minFDE        final displacement error of the best trajectory
    MR            1 if minFDE > 2.0 m, else 0
    brier-minFDE  minFDE + (1 - p_best)^2; the leaderboard ranks by this at K=6
forecast_metrics returns per-scenario arrays; summarize averages them over a population.

Physical plausibility, per trajectory (any model, from positions only):
    infeasible    lateral acceleration above LAT_ACC_LIMIT at any step while moving, at 10 Hz
                  (step-to-step motion) and at 2 Hz (manoeuvre shape)
    off-road      any point outside the drivable area (src/evaluate.py, needs the raw map)
"""

import numpy as np

MISS_THRESHOLD_M = 2.0

# Feasibility, reported at two resolutions. At 10 Hz a +-4 cm zigzag between steps already exceeds
# the limit, so tracking noise alone flags 2.9% of vehicle-like ground-truth futures on val; the
# Phase 3 coordinate head is flagged in 57% of its most probable modes. Resampled to 2 Hz the noise
# and the zigzag both average out (ground truth 0.3%, coordinate head 0.0%): the path shape is judged.
LAT_ACC_LIMIT = 8.0   # m/s^2, about 0.8 g, the tyre-friction limit
FEAS_MIN_SPEED = 1.0  # m/s; below it the direction of a step is noise


def forecast_metrics(pred, prob, gt, k):
    """Per-scenario metrics for the top-k modes.

    pred: (N, M, T, 2) predicted trajectories, prob: (N, M) mode probabilities,
    gt: (N, T, 2) ground truth. Requires k <= M.
    """
    pred, prob, gt = (np.asarray(x, dtype=np.float64) for x in (pred, prob, gt))
    n, m = prob.shape
    if pred.shape[:2] != (n, m) or gt.shape != (n, *pred.shape[2:]):
        raise ValueError(f"shape mismatch: pred {pred.shape}, prob {prob.shape}, gt {gt.shape}")
    if not 1 <= k <= m:
        raise ValueError(f"k={k} but only {m} modes")

    top = np.argsort(-prob, axis=1, kind="stable")[:, :k]  # ties keep mode order
    pred_k = np.take_along_axis(pred, top[:, :, None, None], axis=1)
    prob_k = np.take_along_axis(prob, top, axis=1)

    err = np.linalg.norm(pred_k - gt[:, None], axis=-1)  # (N, k, T)
    rows = np.arange(n)
    best = np.argmin(err[:, :, -1], axis=1)
    min_fde = err[rows, best, -1]
    return {
        "minADE": err[rows, best].mean(axis=1),
        "minFDE": min_fde,
        "MR": (min_fde > MISS_THRESHOLD_M).astype(np.float64),
        "brier-minFDE": min_fde + (1.0 - prob_k[rows, best]) ** 2,
    }


def summarize(per_scenario, mask=None):
    """Average each per-scenario metric, optionally over a boolean population mask."""
    return {name: float(v[mask].mean() if mask is not None else v.mean()) for name, v in per_scenario.items()}


def infeasible(traj, start, stride=1, dt=0.1):
    """Whether each trajectory (..., 60, 2) that starts from start (..., 2), the t=49 position, ever
    exceeds LAT_ACC_LIMIT of lateral acceleration (speed x yaw rate), using every stride-th point
    (1: 10 Hz, 5: 2 Hz). Returns bool (...)."""
    traj = np.asarray(traj, dtype=np.float64)
    start = np.broadcast_to(np.asarray(start, dtype=np.float64), (*traj.shape[:-2], 2))
    pts = np.concatenate([start[..., None, :], traj], axis=-2)[..., ::stride, :]
    step = np.diff(pts, axis=-2)
    h = dt * stride
    speed = np.linalg.norm(step, axis=-1) / h
    heading = np.arctan2(step[..., 1], step[..., 0])
    yaw_rate = ((np.diff(heading, axis=-1) + np.pi) % (2 * np.pi) - np.pi) / h
    lat = 0.5 * (speed[..., 1:] + speed[..., :-1]) * yaw_rate
    moving = np.minimum(speed[..., 1:], speed[..., :-1]) > FEAS_MIN_SPEED
    return (moving & (np.abs(lat) > LAT_ACC_LIMIT)).any(axis=-1)
