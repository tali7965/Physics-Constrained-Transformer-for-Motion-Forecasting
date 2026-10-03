"""Evaluate a forecaster on the val split (or the dev rows of train) and write the results.

Usage:
    python src/evaluate.py --model cv-tracker
    python src/evaluate.py --model cv-displacement
    python src/evaluate.py --checkpoint outputs/runs/lstm/best.pt
    python src/evaluate.py --checkpoint outputs/runs/lstm/best.pt --split dev
    python src/evaluate.py --checkpoint outputs/runs/transformer/best.pt

Reports minADE, minFDE, MR (K=1, and K=6 for multimodal models) and brier-minFDE (K=6) for all
focal agents and for vehicle-like ones, and writes outputs/results/<name>_<split>.json.

For vehicle-like agents it also reports the share of infeasible trajectories (at 10 Hz and 2 Hz, see
metrics.py) and off-road ones, for the most probable mode (@1) and over all modes (@6), next to the
ground truth's own rates. Pedestrians
are left out: their ground truth is off the drivable area 75% of the time. Off-road reads each
scenario's raw map (about 3 s per 1,000 scenarios).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from av2.map.map_api import ArgoverseStaticMap
from matplotlib.path import Path as PolygonPath

from config import PROJECT_ROOT, load_config
from data import load_focal, split_meta, split_rows
from metrics import forecast_metrics, infeasible, summarize
from models import TransformerForecaster, build_model, constant_velocity, pick_device, scene_batch
from preprocess import git_commit, to_global_frame

CV_MODELS = {"cv-tracker": "tracker", "cv-displacement": "displacement"}


@torch.no_grad()
def predict(model, data, device):
    """Trajectories (N, K, 60, 2) and mode probabilities (N, K); the LSTM gives K=1 with p=1."""
    model.eval()
    n = len(data["history"])
    if not isinstance(model, TransformerForecaster):
        out = []
        for i in range(0, n, 1024):
            x = torch.from_numpy(data["history"][i : i + 1024]).to(device)
            out.append(model(x).cpu().numpy())
        pred = np.concatenate(out)[:, None]
        return pred, np.ones((n, 1))
    preds, probs = [], []
    for i in range(0, n, 256):
        traj, logits = model(scene_batch(data, slice(i, i + 256), device))
        preds.append(traj.cpu().numpy())
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def evaluate(pred, prob, data):
    """Per-population, per-K averaged metrics."""
    results = {}
    for population, mask in (("all", None), ("vehicle-like", data["vehicle_like"])):
        n = len(pred) if mask is None else int(mask.sum())
        results[population] = {"n": n}
        for k in (1, 6):
            if prob.shape[1] >= k:
                per = forecast_metrics(pred, prob, data["target"][..., :2], k)
                results[population][f"K={k}"] = summarize(per, mask)
    return results


def off_road(split, rows, traj):
    """Whether each trajectory (N, M, T, 2), in the focal frames of processed rows of split, has a
    point outside every drivable-area polygon of its scenario's map. Returns bool (N, M)."""
    cfg = load_config()
    pdir, rdir = cfg["processed_dir"] / split, cfg["raw_dir"] / split
    sid = np.load(pdir / "scenario_id.npy", mmap_mode="r")[rows]
    origin, theta = np.load(pdir / "origin.npy")[rows], np.load(pdir / "theta.npy")[rows]
    out = np.zeros(traj.shape[:2], dtype=bool)
    for i in range(len(rows)):
        avm = ArgoverseStaticMap.from_json(rdir / sid[i] / f"log_map_archive_{sid[i]}.json")
        pts = to_global_frame(traj[i].reshape(-1, 2).astype(np.float64), origin[i], float(theta[i]))
        inside = np.zeros(len(pts), dtype=bool)
        for area in avm.get_scenario_vector_drivable_areas():
            inside |= PolygonPath(area.xyz[:, :2]).contains_points(pts)
        out[i] = ~inside.reshape(traj.shape[1], -1).all(axis=1)
    return out


def physical(pred, prob, data, split):
    """Infeasible and off-road shares for vehicle-like agents: the most probable mode (@1), all modes
    (@6, multimodal models only), and the ground truth."""
    mask = data["vehicle_like"]
    pred, prob, gt = pred[mask], prob[mask], data["target"][mask][:, None, :, :2]
    start = data["history"][mask][:, None, -1, :2]
    top = np.argmax(prob, axis=1)  # first mode on ties, as in forecast_metrics
    out = {}
    for name, flags in (("infeasible 10Hz", lambda t: infeasible(t, start, stride=1)),
                        ("infeasible 2Hz", lambda t: infeasible(t, start, stride=5)),
                        ("off-road", lambda t: off_road(split, data["rows"][mask], t))):
        per_mode = flags(pred)
        out[f"{name}@1"] = float(per_mode[np.arange(len(top)), top].mean())
        if prob.shape[1] >= 6:
            out[f"{name}@6"] = float(per_mode.mean())
        out[f"{name} ground truth"] = float(flags(gt).mean())
    return out


def print_table(name, results):
    print(f"{'model':<18} {'population':<13} {'n':>7}  {'K':>2} {'minADE':>7} {'minFDE':>7} {'MR':>6} {'brier-minFDE':>13}")
    for population, res in results.items():
        for key in ("K=1", "K=6"):
            if key in res:
                r = res[key]
                brier = f"{r['brier-minFDE']:13.2f}" if key == "K=6" else f"{'-':>13}"
                print(f"{name:<18} {population:<13} {res['n']:>7,}  {key[2:]:>2} {r['minADE']:7.2f} {r['minFDE']:7.2f} {r['MR']:6.3f} {brier}")
    if "physical" in results["vehicle-like"]:
        print("\nvehicle-like, share of trajectories:")
        for key, value in results["vehicle-like"]["physical"].items():
            print(f"  {key:<29} {value:7.2%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--model", choices=sorted(CV_MODELS))
    which.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "dev"), default="val")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "lstm.yaml",
                        help="protocol source for --split dev with a constant-velocity model")
    args = parser.parse_args()

    device = pick_device()
    model = ckpt = None
    if args.checkpoint:
        model, ckpt = load_checkpoint(args.checkpoint, device)
        cfg = ckpt["config"]
        name = cfg["name"]
    else:
        cfg = yaml.safe_load(args.config.read_text())
        name = args.model

    scene = isinstance(model, TransformerForecaster)
    if args.split == "dev":
        _, rows = split_rows(cfg["protocol"], split_meta("train")["n_scenarios"])
        data = load_focal("train", rows, scene=scene)
    else:
        data = load_focal("val", scene=scene)

    if model is None:
        pred = constant_velocity(data["history"], CV_MODELS[args.model])
        pred, prob = pred[:, None], np.ones((len(pred), 1))  # unimodal
    else:
        pred, prob = predict(model, data, device)

    results = evaluate(pred, prob, data)
    results["vehicle-like"]["physical"] = physical(pred, prob, data, "train" if args.split == "dev" else "val")
    print_table(name, results)

    out = PROJECT_ROOT / "outputs" / "results" / f"{name}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {"model": name, "split": args.split, "results": results, "git_commit": git_commit()}
    if ckpt is not None:
        record.update(checkpoint=str(args.checkpoint), epoch=ckpt["epoch"], trained_at_commit=ckpt["git_commit"])
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwritten to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
