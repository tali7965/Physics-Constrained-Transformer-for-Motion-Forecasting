"""Evaluate a forecaster on the val split (or the dev rows of train) and write the results.

Usage:
    python src/evaluate.py --model cv-tracker
    python src/evaluate.py --model cv-displacement
    python src/evaluate.py --checkpoint outputs/runs/lstm/best.pt
    python src/evaluate.py --checkpoint outputs/runs/lstm/best.pt --split dev

Reports minADE, minFDE, MR (K=1, and K=6 for multimodal models) and brier-minFDE (K=6) for all
focal agents and for vehicle-like ones, and writes outputs/results/<name>_<split>.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from config import PROJECT_ROOT
from data import load_focal, split_meta, split_rows
from metrics import forecast_metrics, summarize
from models import LSTMForecaster, constant_velocity, pick_device
from preprocess import git_commit

CV_MODELS = {"cv-tracker": "tracker", "cv-displacement": "displacement"}


@torch.no_grad()
def predict_lstm(model, history, device, batch_size=1024):
    model.eval()
    out = []
    for i in range(0, len(history), batch_size):
        x = torch.from_numpy(history[i : i + batch_size]).to(device)
        out.append(model(x).cpu().numpy())
    return np.concatenate(out)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = LSTMForecaster(**ckpt["config"]["model"]).to(device)
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


def print_table(name, results):
    print(f"{'model':<18} {'population':<13} {'n':>7}  {'K':>2} {'minADE':>7} {'minFDE':>7} {'MR':>6} {'brier-minFDE':>13}")
    for population, res in results.items():
        for key in ("K=1", "K=6"):
            if key in res:
                r = res[key]
                brier = f"{r['brier-minFDE']:13.2f}" if key == "K=6" else f"{'-':>13}"
                print(f"{name:<18} {population:<13} {res['n']:>7,}  {key[2:]:>2} {r['minADE']:7.2f} {r['minFDE']:7.2f} {r['MR']:6.3f} {brier}")


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

    if args.split == "dev":
        _, rows = split_rows(cfg["protocol"], split_meta("train")["n_scenarios"])
        data = load_focal("train", rows)
    else:
        data = load_focal("val")

    if model is None:
        pred = constant_velocity(data["history"], CV_MODELS[args.model])
    else:
        pred = predict_lstm(model, data["history"], device)
    pred, prob = pred[:, None], np.ones((len(pred), 1))  # both baselines are unimodal

    results = evaluate(pred, prob, data)
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
