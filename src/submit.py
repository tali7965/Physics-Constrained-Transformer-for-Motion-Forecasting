"""Write an Argoverse 2 motion forecasting submission for the focal agent from a checkpoint.

Usage:
    python src/submit.py --checkpoint outputs/runs/transformer/best.pt --split test
    python src/submit.py --checkpoint outputs/runs/transformer/best.pt --split val --limit 2000   # pipeline check

Predicts the focal agent's six modes for every scenario of the split, maps them from the focal frame
back to city coordinates, and writes them with the focal track id and the mode probabilities through
av2's ChallengeSubmission to outputs/submissions/<name>_<split>.parquet (one track per scenario).

With --split val the file is then checked end to end: it is read back with
ChallengeSubmission.from_parquet and scored with av2's own metric functions against the focal track's
positions in the raw scenario files, and the result must match src/metrics.py on the same rows.
"""

import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from av2.datasets.motion_forecasting.eval import metrics as av2_metrics
from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission

from config import PROJECT_ROOT, load_config
from data import load_focal, split_meta
from evaluate import load_checkpoint, predict
from metrics import forecast_metrics, summarize
from models import pick_device
from preprocess import HISTORY, git_commit, to_global_frame

MODES = 6  # the leaderboard ranks by brier-minFDE at K=6


def raw_path(split, scenario_id):
    return load_config()["raw_dir"] / split / scenario_id / f"scenario_{scenario_id}.parquet"


def focal_track_id(split, scenario_id):
    return str(pq.read_table(raw_path(split, scenario_id), columns=["focal_track_id"])["focal_track_id"][0])


def focal_future(split, scenario_id):
    """The focal track's 60 future positions (60, 2) in city coordinates, from the raw scenario file."""
    df = pd.read_parquet(raw_path(split, scenario_id), columns=["track_id", "focal_track_id", "timestep",
                                                                 "position_x", "position_y"])
    df = df[df["track_id"] == df["focal_track_id"]].assign(timestep=lambda d: d["timestep"].astype(int))
    future = df[df["timestep"] >= HISTORY].sort_values("timestep")
    return future[["position_x", "position_y"]].to_numpy()


def check(path, split, scenario_ids, pred, prob, target):
    """Score the written file with av2's metric functions against raw ground truth, and compare with
    src/metrics.py on the focal-frame predictions."""
    submission = ChallengeSubmission.from_parquet(path)
    if set(submission.predictions) != set(scenario_ids):
        raise ValueError("submission scenarios differ from the predicted ones")
    av2 = {"minADE": [], "minFDE": [], "brier-minFDE": []}
    for sid in scenario_ids:
        probs, tracks = submission.predictions[sid]
        (traj,) = tracks.values()
        gt = focal_future(split, sid)
        fde = av2_metrics.compute_fde(traj, gt)
        best = np.argmin(fde)
        av2["minADE"].append(av2_metrics.compute_ade(traj, gt)[best])
        av2["minFDE"].append(fde[best])
        av2["brier-minFDE"].append(av2_metrics.compute_brier_fde(traj, gt, probs)[best])
    ours = summarize(forecast_metrics(pred, prob, target, MODES))
    print(f"\ncheck on {len(scenario_ids):,} {split} scenarios, K={MODES}:")
    worst = 0.0
    for name, values in av2.items():
        diff = abs(np.mean(values) - ours[name])
        worst = max(worst, diff)
        print(f"  {name:<13} av2 on the file {np.mean(values):.5f}   src/metrics.py {ours[name]:.5f}   diff {diff:.1e}")
    if worst > 1e-3:
        raise ValueError(f"submission check failed: metrics differ by {worst:.1e}")
    print("  OK: the file reproduces the focal-frame metrics")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=None, help="only the first N scenarios")
    args = parser.parse_args()

    device = pick_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    name = ckpt["config"]["name"]
    if ckpt["config"]["model"].get("modes") != MODES:
        raise ValueError(f"{name} has {ckpt['config']['model'].get('modes')} modes; the leaderboard expects {MODES}")
    n = split_meta(args.split)["n_scenarios"]
    rows = np.arange(n if args.limit is None else min(args.limit, n))
    data = load_focal(args.split, rows, scene=True)
    pred, prob = predict(model, data, device)

    pdir = load_config()["processed_dir"] / args.split
    scenario_ids = [str(s) for s in np.load(pdir / "scenario_id.npy", mmap_mode="r")[rows]]
    origin, theta = np.load(pdir / "origin.npy")[rows], np.load(pdir / "theta.npy")[rows]
    predictions = {}
    for i, sid in enumerate(scenario_ids):
        traj = to_global_frame(pred[i].reshape(-1, 2).astype(np.float64), origin[i], float(theta[i])).reshape(MODES, -1, 2)
        p = prob[i].astype(np.float64)
        predictions[sid] = (p / p.sum(), {focal_track_id(args.split, sid): traj})

    out = PROJECT_ROOT / "outputs" / "submissions" / f"{name}_{args.split}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    ChallengeSubmission(predictions).to_parquet(out)
    print(f"{len(predictions):,} {args.split} scenarios from {args.checkpoint} (trained at {ckpt['git_commit']}, "
          f"written at {git_commit()}) -> {out.relative_to(PROJECT_ROOT)}")

    if args.split == "val":
        check(out, args.split, scenario_ids, pred, prob, data["target"][..., :2])


if __name__ == "__main__":
    main()
