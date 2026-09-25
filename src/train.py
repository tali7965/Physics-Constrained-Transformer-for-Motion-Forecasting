"""Train the LSTM encoder-decoder baseline.

Usage:
    python src/train.py --config configs/lstm.yaml
    python src/train.py --config configs/lstm.yaml --overfit 256 --epochs 2000   # pipeline check

Loads the training subset and dev rows (focal history and target) into memory, trains on MPS when
available, and after each epoch records train loss and dev minADE / minFDE / MR in
outputs/runs/<name>/metrics.csv (a summary line goes to train.log). best.pt is saved whenever dev
minADE improves; last.pt at every logged epoch and at the end. With --overfit N the model trains
and is scored on the first N training rows only.
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from config import PROJECT_ROOT
from data import load_focal, split_meta, split_rows
from evaluate import predict_lstm
from metrics import forecast_metrics, summarize
from models import LSTMForecaster, pick_device
from preprocess import git_commit


def ade_loss(pred, target):
    """Mean Euclidean error over all steps; the epsilon keeps the gradient finite at zero error."""
    return torch.sqrt(((pred - target) ** 2).sum(-1) + 1e-12).mean()


def score(model, data, device):
    pred = predict_lstm(model, data["history"], device)
    per = forecast_metrics(pred[:, None], np.ones((len(pred), 1)), data["target"][..., :2], 1)
    return summarize(per)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overfit", type=int, default=None, help="train and score on the first N training rows")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    tc = cfg["train"]
    epochs = args.epochs or tc["epochs"]
    name = cfg["name"] + (f"-overfit{args.overfit}" if args.overfit else "")
    run_dir = PROJECT_ROOT / "outputs" / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(run_dir / "train.log", mode="w"), logging.StreamHandler()],
    )
    log = logging.getLogger("train")

    torch.manual_seed(tc["seed"])
    device = pick_device()
    train_rows, dev_rows = split_rows(cfg["protocol"], split_meta("train")["n_scenarios"])
    if args.overfit:
        train_rows = dev_rows = train_rows[: args.overfit]
    t0 = time.time()
    train = load_focal("train", train_rows)
    dev = train if args.overfit else load_focal("train", dev_rows)
    log.info(f"run {name}: {len(train_rows):,} train rows, {len(dev_rows):,} dev rows, loaded in "
             f"{time.time() - t0:.0f}s, device {device}, commit {git_commit()}")

    x = torch.from_numpy(train["history"]).to(device)
    y = torch.from_numpy(np.ascontiguousarray(train["target"][..., :2])).to(device)
    model = LSTMForecaster(**cfg["model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    steps_per_epoch = -(-len(x) // tc["batch_size"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    shuffle = torch.Generator().manual_seed(tc["seed"])
    log.info(f"{sum(p.numel() for p in model.parameters()):,} parameters, {epochs} epochs x {steps_per_epoch} steps")

    log_every = max(1, epochs // 30)
    fields = ["epoch", "train_ade", "dev_minADE", "dev_minFDE", "dev_MR", "lr", "seconds"]
    best = float("inf")
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            total, count = 0.0, 0
            for idx in torch.randperm(len(x), generator=shuffle).split(tc["batch_size"]):
                idx = idx.to(device)
                loss = ade_loss(model(x[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                opt.step()
                sched.step()
                total += loss.item() * len(idx)
                count += len(idx)
            train_ade = total / count
            if not np.isfinite(train_ade):
                raise RuntimeError(f"epoch {epoch}: non-finite train loss")
            d = score(model, dev, device)
            row = {"epoch": epoch, "train_ade": round(train_ade, 4), "dev_minADE": round(d["minADE"], 4),
                   "dev_minFDE": round(d["minFDE"], 4), "dev_MR": round(d["MR"], 4),
                   "lr": f"{sched.get_last_lr()[0]:.2e}", "seconds": round(time.time() - t0, 1)}
            writer.writerow(row)
            f.flush()

            state = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "dev": d, "git_commit": git_commit()}
            improved = d["minADE"] < best
            if improved:
                best = d["minADE"]
                torch.save(state, run_dir / "best.pt")
            if epoch % log_every == 0 or epoch == epochs:
                torch.save(state, run_dir / "last.pt")
                log.info(f"epoch {epoch:4d}  train ADE {train_ade:.3f}  dev minADE {d['minADE']:.3f}  "
                         f"minFDE {d['minFDE']:.3f}  MR {d['MR']:.3f}  ({row['seconds']}s){'  *' if improved else ''}")

    log.info(f"done: best dev minADE {best:.3f}; checkpoints and metrics.csv in {run_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
