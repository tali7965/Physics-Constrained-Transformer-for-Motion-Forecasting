"""Train a forecaster: the LSTM baseline or the Transformer.

Usage:
    python src/train.py --config configs/lstm.yaml
    python src/train.py --config configs/transformer.yaml
    python src/train.py --config configs/lstm.yaml --overfit 256 --epochs 2000   # pipeline check

Loads the training subset and dev rows into memory (focal history for the LSTM, full scenes for the
Transformer), trains on MPS when available, and after each epoch records train loss and dev metrics
in outputs/runs/<name>/metrics.csv (a summary line goes to train.log). best.pt is saved whenever
the dev metric named by train.select improves (default minADE at K=1); last.pt at every logged
epoch and at the end. With --overfit N the model trains and is scored on the first N training rows
only.
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from config import PROJECT_ROOT
from data import load_focal, split_meta, split_rows
from evaluate import predict
from metrics import forecast_metrics, summarize
from models import build_model, pick_device, scene_batch
from preprocess import git_commit


def ade_loss(pred, target):
    """Mean Euclidean error over all steps; the epsilon keeps the gradient finite at zero error."""
    return torch.sqrt(((pred - target) ** 2).sum(-1) + 1e-12).mean()


def wta_loss(traj, logits, target):
    """Winner-takes-all: the mode with the lowest endpoint error (the metric's rule) gets the ADE
    regression loss, and the logits are trained by cross-entropy to pick it. Returns (reg, cls)."""
    with torch.no_grad():
        best = (traj[:, :, -1] - target[:, None, -1]).norm(dim=-1).argmin(dim=1)
    reg = ade_loss(traj[torch.arange(len(traj), device=traj.device), best], target)
    return reg, F.cross_entropy(logits, best)


def score(model, data, device):
    """Dev metrics: K=1 (most probable mode) as minADE, minFDE, MR; K=6 ones suffixed @6."""
    pred, prob = predict(model, data, device)
    gt = data["target"][..., :2]
    out = summarize(forecast_metrics(pred, prob, gt, 1))
    del out["brier-minFDE"]
    if prob.shape[1] >= 6:
        out.update({f"{k}@6": v for k, v in summarize(forecast_metrics(pred, prob, gt, 6)).items()})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overfit", type=int, default=None, help="train and score on the first N training rows")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    tc = cfg["train"]
    epochs = args.epochs or tc["epochs"]
    select = tc.get("select", "minADE")
    multimodal = cfg["model"].get("type", "lstm") == "transformer"
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
    train = load_focal("train", train_rows, scene=multimodal)
    dev = train if args.overfit else load_focal("train", dev_rows, scene=multimodal)
    log.info(f"run {name}: {len(train_rows):,} train rows, {len(dev_rows):,} dev rows, loaded in "
             f"{time.time() - t0:.0f}s, device {device}, commit {git_commit()}")

    # The LSTM's small inputs live on the device; full scenes stay in RAM and move per batch.
    y = torch.from_numpy(np.ascontiguousarray(train["target"][..., :2]))
    if not multimodal:
        x = torch.from_numpy(train["history"]).to(device)
        y = y.to(device)
    model = build_model(cfg["model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    steps_per_epoch = -(-len(y) // tc["batch_size"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    shuffle = torch.Generator().manual_seed(tc["seed"])
    log.info(f"{sum(p.numel() for p in model.parameters()):,} parameters, {epochs} epochs x {steps_per_epoch} steps, "
             f"best.pt by dev {select}")

    log_every = max(1, epochs // 30)
    dev_keys = ["minADE", "minFDE", "MR"] + (["minADE@6", "minFDE@6", "MR@6", "brier-minFDE@6"] if multimodal else [])
    fields = ["epoch", "train_ade"] + (["train_cls"] if multimodal else []) + [f"dev_{k}" for k in dev_keys] + ["lr", "seconds"]
    best = float("inf")
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            total, total_cls, count = 0.0, 0.0, 0
            for idx in torch.randperm(len(y), generator=shuffle).split(tc["batch_size"]):
                if multimodal:
                    i = idx.numpy()
                    traj, logits = model(scene_batch(train, i, device))
                    reg, cls = wta_loss(traj, logits, y[i].to(device))
                    loss = reg + cls
                    total_cls += cls.item() * len(idx)
                else:
                    idx = idx.to(device)
                    loss = reg = ade_loss(model(x[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                opt.step()
                sched.step()
                total += reg.item() * len(idx)
                count += len(idx)
            train_ade = total / count
            if not np.isfinite(train_ade + total_cls):
                raise RuntimeError(f"epoch {epoch}: non-finite train loss")
            d = score(model, dev, device)
            row = {"epoch": epoch, "train_ade": round(train_ade, 4), **{f"dev_{k}": round(d[k], 4) for k in dev_keys},
                   "lr": f"{sched.get_last_lr()[0]:.2e}", "seconds": round(time.time() - t0, 1)}
            if multimodal:
                row["train_cls"] = round(total_cls / count, 4)
            writer.writerow(row)
            f.flush()

            state = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "dev": d, "git_commit": git_commit()}
            improved = d[select] < best
            if improved:
                best = d[select]
                torch.save(state, run_dir / "best.pt")
            if epoch % log_every == 0 or epoch == epochs:
                torch.save(state, run_dir / "last.pt")
                extra = f"  @6 minFDE {d['minFDE@6']:.3f}  brier {d['brier-minFDE@6']:.3f}" if multimodal else ""
                log.info(f"epoch {epoch:4d}  train ADE {train_ade:.3f}  dev minADE {d['minADE']:.3f}  "
                         f"minFDE {d['minFDE']:.3f}  MR {d['MR']:.3f}{extra}  ({row['seconds']}s){'  *' if improved else ''}")

    log.info(f"done: best dev {select} {best:.3f}; checkpoints and metrics.csv in {run_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
