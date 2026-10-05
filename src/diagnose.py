"""Diagnose the physics head's accuracy gap against the coordinate head, on the dev rows.

Usage:
    python src/diagnose.py --physics outputs/runs/physics/best.pt --coordinate outputs/runs/transformer/best.pt

Three checks, all on dev (val is kept for reporting):
    saturation    share of bounded controls near their limits (|tanh| > 0.95), per control and per
                  1 s band of the future, for the most probable mode and over all modes
    conditioning  mean |d loss / d traj_head output| per 1 s band under the training loss, relative
                  to its mean over all 60 steps, for both heads (a control at step k moves every
                  later position, so early steps may dominate)
    speed bins    minFDE@6 and brier-minFDE@6 of both heads by the focal agent's initial speed, and
                  for vehicle-like agents the off-road share of the most probable mode and the
                  initial-heading error against the first ground-truth step
"""

import argparse

import numpy as np
import torch

from data import load_focal, split_meta, split_rows
from evaluate import load_checkpoint, off_road, predict
from metrics import forecast_metrics
from models import pick_device, scene_batch
import physics
from preprocess import FUTURE
from train import wta_loss

BANDS = [(i, i + 10) for i in range(0, FUTURE, 10)]  # 1 s bands of the 60 future steps
SPEED_EDGES = [0.0, physics.MIN_SPEED, 2.0, 5.0, 10.0, 15.0, np.inf]  # m/s


def capture(model):
    """Forward hook on traj_head: keeps every output (detached, on CPU) in outputs and the last one,
    with its gradient retained when grad is on, in last."""
    store = {"outputs": [], "last": None}

    def hook(module, inputs, output):
        if output.requires_grad:
            output.retain_grad()
        store["last"] = output
        store["outputs"].append(output.detach().cpu())

    model.traj_head.register_forward_hook(hook)
    return store


def scatter(flags, mask):
    """Per-row values from flags over the rows in mask; other rows are NaN."""
    out = np.full(len(mask), np.nan)
    out[mask] = flags
    return out


def band_row(values):
    """values (60,) -> one entry per 1 s band."""
    return "  ".join(f"{values[a:b].mean():7.3f}" for a, b in BANDS)


def saturation(raw, prob):
    raw = raw.unflatten(-1, (FUTURE, 2)).numpy()  # (N, K, 60, 2)
    near = np.abs(np.tanh(raw)) > 0.95
    top = np.argmax(prob, axis=1)
    header = "  ".join(f"{a / 10:.0f}-{b / 10:.0f} s".rjust(7) for a, b in BANDS)
    print(f"\nsaturation: share of controls with |tanh| > 0.95\n{'':22s} {header}   overall")
    for c, name in enumerate(("acceleration", "steering")):
        for label, flags in (("top mode", near[np.arange(len(top)), top, :, c]), ("all modes", near[..., c])):
            per_step = flags.reshape(-1, FUTURE).mean(axis=0)
            print(f"{name:12s} {label:9s} {band_row(per_step)}   {flags.mean():7.3f}")


def conditioning(model, store, dev, device, batches, batch_size):
    """Mean |grad| of the loss w.r.t. traj_head's output per future step, per output channel."""
    model.eval()  # no dropout; gradients still flow
    total = torch.zeros(FUTURE, 2)
    for b in range(batches):
        i = np.arange(b * batch_size, (b + 1) * batch_size)
        traj, logits = model(scene_batch(dev, i, device))
        reg, cls = wta_loss(traj, logits, torch.from_numpy(dev["target"][i, :, :2]).to(device))
        model.zero_grad()
        (reg + cls).backward()
        grad = store["last"].grad.unflatten(-1, (FUTURE, 2)).abs().sum(dim=1)  # WTA: one mode per scene
        total += grad.mean(dim=0).cpu()
    return (total / batches).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--physics", required=True)
    parser.add_argument("--coordinate", required=True)
    parser.add_argument("--batches", type=int, default=8, help="dev batches for the conditioning check")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = pick_device()
    heads = {"coordinate": args.coordinate, "physics": args.physics}
    models = {name: load_checkpoint(path, device) for name, path in heads.items()}
    protocols = {name: ckpt["config"]["protocol"] for name, (_, ckpt) in models.items()}
    if protocols["coordinate"] != protocols["physics"]:
        raise ValueError(f"checkpoints use different protocols: {protocols}")
    _, rows = split_rows(protocols["physics"], split_meta("train")["n_scenarios"])
    dev = load_focal("train", rows, scene=True)
    gt = dev["target"][..., :2]
    print(f"dev: {len(rows):,} scenes")

    results, grads = {}, {}
    for name, (model, _) in models.items():
        store = capture(model)
        pred, prob = predict(model, dev, device)
        results[name] = (pred, prob, forecast_metrics(pred, prob, gt, 6))
        if name == "physics":
            saturation(torch.cat(store["outputs"]), prob)
        grads[name] = conditioning(model, store, dev, device, args.batches, args.batch_size)

    header = "  ".join(f"{a / 10:.0f}-{b / 10:.0f} s".rjust(7) for a, b in BANDS)
    print(f"\nconditioning: mean |grad| per 1 s band / mean over all steps "
          f"({args.batches} x {args.batch_size} dev scenes)\n{'':22s} {header}   mean |grad|")
    for name, channels in (("coordinate", ("x", "y")), ("physics", ("acceleration", "steering"))):
        for c, channel in enumerate(channels):
            g = grads[name][:, c]
            print(f"{name:10s} {channel:11s} {band_row(g / g.mean())}   {g.mean():.2e}")

    # Speed bins. The initial state is the physics head's own (last-step displacement).
    history = torch.from_numpy(dev["history"])
    _, heading, speed = physics.initial_state(history)
    heading, speed = heading.numpy(), speed.numpy()
    first = gt[:, 0] - dev["history"][:, -1, :2]
    gt_heading = np.arctan2(first[:, 1], first[:, 0])
    wrap = lambda a: np.degrees(np.abs((a + np.pi) % (2 * np.pi) - np.pi))
    err_used, err_recorded = wrap(heading - gt_heading), wrap(0.0 - gt_heading)  # recorded heading is 0 in the focal frame

    veh = dev["vehicle_like"]
    road = {}
    for name, (pred, prob, _) in results.items():
        top = np.argmax(prob, axis=1)
        road[name] = off_road("train", rows[veh], pred[veh][np.arange(veh.sum()), top[veh]][:, None])[:, 0]
    road["ground truth"] = off_road("train", rows[veh], gt[veh][:, None])[:, 0]
    road = {name: scatter(flags, veh) for name, flags in road.items()}

    print("\nby initial speed (minFDE / brier-minFDE at K=6, all focal; off-road @1 and heading error, vehicle-like)")
    print(f"{'speed m/s':>11} {'n':>5} {'veh':>5}  {'coord FDE':>9} {'phys FDE':>9} {'gap':>6}  "
          f"{'coord brier':>11} {'phys brier':>10} {'gap':>6}  {'off coord':>9} {'off phys':>9} {'off gt':>7}  "
          f"{'hdg used p50/p95':>16} {'hdg rec p50/p95':>16}")
    for lo, hi in zip(SPEED_EDGES[:-1], SPEED_EDGES[1:]):
        m = (speed >= lo) & (speed < hi)
        mv = m & veh
        if not m.any():
            continue
        c, p = results["coordinate"][2], results["physics"][2]
        fde_c, fde_p = c["minFDE"][m].mean(), p["minFDE"][m].mean()
        br_c, br_p = c["brier-minFDE"][m].mean(), p["brier-minFDE"][m].mean()
        off = [road[k][mv].mean() if mv.any() else np.nan for k in ("coordinate", "physics", "ground truth")]
        hu = np.percentile(err_used[mv], [50, 95]) if mv.any() else (np.nan, np.nan)
        hr = np.percentile(err_recorded[mv], [50, 95]) if mv.any() else (np.nan, np.nan)
        print(f"{f'{lo:g}-{hi:g}':>11} {m.sum():5d} {mv.sum():5d}  {fde_c:9.2f} {fde_p:9.2f} {fde_p - fde_c:6.2f}  "
              f"{br_c:11.2f} {br_p:10.2f} {br_p - br_c:6.2f}  {off[0]:9.1%} {off[1]:9.1%} {off[2]:7.1%}  "
              f"{hu[0]:7.1f}/{hu[1]:7.1f} {hr[0]:7.1f}/{hr[1]:7.1f}")


if __name__ == "__main__":
    main()
