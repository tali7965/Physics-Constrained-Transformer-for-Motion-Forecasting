"""Motion forecasting metrics, following the Argoverse 2 challenge definitions.

For each scenario the K most probable predicted trajectories are kept, and the best of them is the
one with the lowest endpoint error (the Argoverse 1 and 2 convention). Then:
    minADE        average displacement error of the best trajectory
    minFDE        final displacement error of the best trajectory
    MR            1 if minFDE > 2.0 m, else 0
    brier-minFDE  minFDE + (1 - p_best)^2; the leaderboard ranks by this at K=6
forecast_metrics returns per-scenario arrays; summarize averages them over a population.
"""

import numpy as np

MISS_THRESHOLD_M = 2.0


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
