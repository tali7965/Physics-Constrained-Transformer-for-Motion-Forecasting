"""Smooth a coordinate head's predictions after the fact (Phase 7, route A), in two ways: fit the
kinematic bicycle model, with comfort limits measured from the data, or low-pass filter the positions.

Usage:
    python src/smooth.py --limits                                              # measure the comfort limits
    python src/smooth.py --checkpoint outputs/runs/transformer/best.pt --split dev
    python src/smooth.py --checkpoint outputs/runs/transformer/best.pt --split val

Comfort limits (--limits): on the vehicle-like futures of the training subset whose heading changes
by less than TURN_DEG (diagnose.turn_angle), the 95th percentile of |longitudinal acceleration| and
|lateral acceleration| (speed x yaw rate while moving faster than 1 m/s). Both are measured at 2 Hz
over the first 5 s, since 10 Hz differences amplify tracking noise and the last 0.7 s of every
future decelerates artificially (see the development log).

Fitting: physics.fit_controls finds bounded controls whose rollout is closest (ADE) to the predicted
mode, with |a| <= A_COMFORT and lateral acceleration <= A_LAT_COMFORT. Filtering: a quadratic
Savitzky-Golay filter over FILTER_WINDOW steps, with the current position as the first sample. Mode
probabilities are kept either way.

Four versions are scored: the raw predictions, every mode fitted, only gentle modes fitted (predicted
turn under TURN_DEG; the switch uses the prediction, so it is available at run time), and every mode
filtered. Each gets evaluate.evaluate and evaluate.physical, minFDE@6 over the first 5 s (before the
end-of-future artifact), and the 95th percentiles of the most probable modes' accelerations (the
measure behind the limits). Writes outputs/results/<name>-smooth_<split>.json.
"""

import argparse
import json

import numpy as np
import torch
import yaml
from scipy.signal import savgol_filter

from config import PROJECT_ROOT
from data import load_focal, split_meta, split_rows
from diagnose import TURN_DEG, turn_angle
from evaluate import evaluate, load_checkpoint, physical, predict
from metrics import FEAS_MIN_SPEED, forecast_metrics, summarize
from models import pick_device
import physics
from preprocess import DT, git_commit

A_COMFORT = 2.84      # m/s^2, p95 |longitudinal acceleration| of gentle training futures (--limits)
A_LAT_COMFORT = 0.87  # m/s^2, p95 |lateral acceleration| of gentle training futures (--limits)
STRIDE = 5            # 2 Hz
WINDOW = 50           # first 5 s of the 60 future steps
CHUNK = 10 * physics.MAX_FIT_BATCH  # trajectories per progress line
FILTER_WINDOW = 9     # steps (0.9 s); on dev it cuts 10 Hz infeasibility from 57% to 1% at no FDE cost


def motion(traj, start):
    """Longitudinal and lateral acceleration (m/s^2) at 2 Hz over the first 5 s of trajectories
    (..., 60, 2) starting at start (..., 2). Lateral is NaN where the agent moves slower than
    FEAS_MIN_SPEED, as in metrics.infeasible."""
    pts = np.concatenate([start[..., None, :], traj[..., :WINDOW, :]], axis=-2)[..., ::STRIDE, :]
    step = np.diff(pts, axis=-2)
    h = DT * STRIDE
    speed = np.linalg.norm(step, axis=-1) / h
    heading = np.arctan2(step[..., 1], step[..., 0])
    yaw_rate = ((np.diff(heading, axis=-1) + np.pi) % (2 * np.pi) - np.pi) / h
    lateral = 0.5 * (speed[..., 1:] + speed[..., :-1]) * yaw_rate
    moving = np.minimum(speed[..., 1:], speed[..., :-1]) > FEAS_MIN_SPEED
    return np.diff(speed, axis=-1) / h, np.where(moving, lateral, np.nan)


def gentle(traj, start):
    """Whether each trajectory (n, 60, 2) turns by less than TURN_DEG; slow ones (no reliable
    direction) count as gentle."""
    deg = np.degrees(np.abs(turn_angle(start, traj)))
    return ~(deg >= TURN_DEG)


def measure_limits():
    protocol = yaml.safe_load((PROJECT_ROOT / "configs" / "transformer.yaml").read_text())["protocol"]
    rows, _ = split_rows(protocol, split_meta("train")["n_scenarios"])
    data = load_focal("train", rows)
    veh = data["vehicle_like"]
    start, gt = data["history"][veh, -1, :2], data["target"][veh, :, :2]
    keep = gentle(gt, start)
    a_long, a_lat = motion(gt[keep], start[keep])
    print(f"vehicle-like training futures: {veh.sum():,}, gentle (turn < {TURN_DEG:g} deg or slow): {keep.sum():,}")
    for name, values in (("|longitudinal acceleration|", np.abs(a_long)), ("|lateral acceleration|", np.abs(a_lat))):
        p = np.nanpercentile(values, [50, 90, 95, 99])
        print(f"  {name:28s} p50 {p[0]:.2f}  p90 {p[1]:.2f}  p95 {p[2]:.2f}  p99 {p[3]:.2f} m/s^2")


def smooth(history, pred, device):
    """Bounded-control fits to every mode of pred (N, K, 60, 2), from each scene's history."""
    n, k = pred.shape[:2]
    hist = torch.from_numpy(np.repeat(history, k, axis=0))
    flat = torch.from_numpy(pred.reshape(n * k, -1, 2).astype(np.float32))
    out = []
    for i in range(0, n * k, CHUNK):
        fit = physics.fit_controls(hist[i : i + CHUNK].to(device), flat[i : i + CHUNK].to(device),
                                   a_max=A_COMFORT, a_lat=A_LAT_COMFORT)
        out.append(fit.cpu().numpy())
        print(f"  smoothed {min(i + CHUNK, n * k):,} / {n * k:,} modes")
    return np.concatenate(out).reshape(pred.shape)


def savgol(pred, start):
    """Quadratic Savitzky-Golay filter over FILTER_WINDOW steps along the time axis of pred (N, K, 60, 2),
    with the current position start (N, 2) as the first sample."""
    anchor = np.broadcast_to(start[:, None, None], (*pred.shape[:2], 1, 2))
    return savgol_filter(np.concatenate([anchor, pred], axis=2), FILTER_WINDOW, 2, axis=2, mode="interp")[:, :, 1:]


def comfort(traj, start):
    """95th percentiles of |longitudinal| and |lateral| acceleration over every 2 Hz step (first 5 s)
    of trajectories (n, 60, 2): the measure that defines the comfort limits."""
    a_long, a_lat = motion(traj, start)
    return float(np.nanpercentile(np.abs(a_long), 95)), float(np.nanpercentile(np.abs(a_lat), 95))


def top_comfort(pred, prob, data):
    veh = data["vehicle_like"]
    top = pred[veh][np.arange(veh.sum()), np.argmax(prob[veh], axis=1)]
    return comfort(top, data["history"][veh, -1, :2])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limits", action="store_true", help="measure the comfort limits and exit")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("dev", "val"), default="dev")
    args = parser.parse_args()
    if args.limits:
        measure_limits()
        return
    device = pick_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    name = ckpt["config"]["name"]
    if args.split == "dev":
        _, rows = split_rows(ckpt["config"]["protocol"], split_meta("train")["n_scenarios"])
        data, map_split = load_focal("train", rows, scene=True), "train"
    else:
        data, map_split = load_focal("val", scene=True), "val"
    pred, prob = predict(model, data, device)
    n, k = pred.shape[:2]
    start = np.repeat(data["history"][:, -1, :2], k, axis=0)
    mild = gentle(pred.reshape(n * k, -1, 2), start).reshape(n, k)
    print(f"{name} on {args.split}: {n:,} scenes, {mild.mean():.1%} of modes gentle; fitting every mode")
    smoothed = smooth(data["history"], pred, device)
    versions = {"raw": pred, "fitted": smoothed, "fitted where gentle": np.where(mild[..., None, None], smoothed, pred),
                "filtered": savgol(pred, data["history"][:, -1, :2])}

    record = {"model": name, "split": args.split, "limits": {"a": A_COMFORT, "a_lat": A_LAT_COMFORT, "turn_deg": TURN_DEG, "filter_window": FILTER_WINDOW},
              "gentle_modes": float(mild.mean()), "results": {}, "git_commit": git_commit()}
    print(f"\n{'version':<22} {'brier@6':>8} {'minFDE@6':>9} {'5 s':>6} {'minFDE@1':>9} {'off-road@1':>11} "
          f"{'infeas 10Hz@1':>14} {'infeas 2Hz@1':>13} {'p95 |a|@1':>10} {'p95 |a_lat|@1':>14}")
    for version, p in versions.items():
        res = evaluate(p, prob, data)
        res["all"]["K=6, first 5 s"] = summarize(forecast_metrics(p[:, :, :WINDOW], prob, data["target"][:, :WINDOW, :2], 6))
        res["vehicle-like"]["physical"] = physical(p, prob, data, map_split)
        res["vehicle-like"]["comfort p95@1"] = dict(zip(("a", "a_lat"), top_comfort(p, prob, data)))
        record["results"][version] = res
        a, v = res["all"], res["vehicle-like"]
        print(f"{version:<22} {a['K=6']['brier-minFDE']:8.3f} {a['K=6']['minFDE']:9.3f} {a['K=6, first 5 s']['minFDE']:6.3f} {a['K=1']['minFDE']:9.3f} "
              f"{v['physical']['off-road@1']:11.2%} {v['physical']['infeasible 10Hz@1']:14.2%} "
              f"{v['physical']['infeasible 2Hz@1']:13.2%} {v['comfort p95@1']['a']:10.2f} {v['comfort p95@1']['a_lat']:14.2f}")
    veh = data["vehicle_like"]
    gt_a, gt_lat = comfort(data["target"][veh, :, :2], data["history"][veh, -1, :2])
    record["ground truth comfort p95"] = {"a": gt_a, "a_lat": gt_lat}
    print(f"{'ground truth':<22} {'':>8} {'':>9} {'':>6} {'':>9} {'':>11} {'':>14} {'':>13} {gt_a:10.2f} {gt_lat:14.2f}")

    out = PROJECT_ROOT / "outputs" / "results" / f"{name}-smooth_{args.split}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwritten to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
