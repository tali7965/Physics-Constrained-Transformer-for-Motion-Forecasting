"""Differentiable kinematic model: controls (acceleration, steering) -> future positions.

The kinematic bicycle model in curvature form: state (x, y, heading, speed), and per 0.1 s step an
acceleration a and a steering fraction s in [-1, 1]. Path curvature is kappa = s * kappa_max(v),
with kappa_max(v) = min(K_MAX, A_LAT / v^2): a 5 m turning radius at low speed and at most A_LAT of
lateral acceleration above ~5.5 m/s. Scaling (rather than clipping) keeps the gradient alive when
the speed limit binds. Curvature form means kappa = tan(steer angle) / wheelbase with no per-type
wheelbase.

Each step: v' = max(v + a dt, 0), and the agent travels ds = (v + v') / 2 * dt along an arc whose
heading turns by kappa * ds. Speed never goes negative, so the model cannot reverse (sustained
reversing is 0.07% of vehicle-like focal futures in the training subset).

The bounds come from ground-truth fitting (--check) on the training subset: with them the best
reachable vehicle-like trajectory is within 0.16 m FDE of the truth on average, and adding the
lateral limit costs nothing measurable.

Usage:
    python src/physics.py --check --n 2000                    # fit controls to ground truth: the head's error floor
    python src/physics.py --check --n 2000 --control-step 5   # the same with knots every 0.5 s
"""

import argparse
import time

import numpy as np
import torch

from preprocess import DT, FUTURE

A_MAX = 8.0     # |acceleration|, m/s^2
K_MAX = 0.2     # |curvature|, 1/m
A_LAT = 6.0     # |lateral acceleration| v^2 kappa, m/s^2
MIN_SPEED = 0.5  # m/s; below it the last-step direction is noise and the recorded heading is used


def initial_state(history):
    """Position, heading and speed at t=49 from (..., 50, 5) history.

    Speed and heading come from the last-step displacement p49 - p48. Below MIN_SPEED the heading is
    the recorded one, which is 0 in the focal frame. The recorded heading is unreliable for
    pedestrians; starting from it leaves 3.3% of non-vehicle futures unreachable (> 2 m FDE).
    """
    step = history[..., -1, 0:2] - history[..., -2, 0:2]
    speed = step.norm(dim=-1) / DT
    heading = torch.where(speed > MIN_SPEED, torch.atan2(step[..., 1], step[..., 0]), torch.zeros_like(speed))
    return history[..., -1, 0:2], heading, speed


def bound(raw):
    """Map unconstrained (..., 60, 2) head outputs to acceleration in [-A_MAX, A_MAX] and steering
    fraction in [-1, 1]."""
    return torch.stack([A_MAX * torch.tanh(raw[..., 0]), torch.tanh(raw[..., 1])], dim=-1)


def interpolation(step):
    """(60, 60 // step + 1) weights that linearly interpolate controls at knots on steps 0, step, ...,
    60 to every step, so controls can only change smoothly over step * 0.1 s. Interpolated controls
    stay within the knots' bounds."""
    if FUTURE % step:
        raise ValueError(f"control step {step} does not divide {FUTURE}")
    s = torch.arange(FUTURE, dtype=torch.float32) / step
    j = s.floor().long()
    w = s - j
    weights = torch.zeros(FUTURE, FUTURE // step + 1)
    weights[torch.arange(FUTURE), j] = 1 - w
    weights[torch.arange(FUTURE), j + 1] = w
    return weights


def interpolate(knots, weights):
    """Knot controls (..., n_knots, 2) -> per-step controls (..., 60, 2)."""
    return torch.einsum("tk,...kc->...tc", weights.to(knots), knots)


def rollout(pos, heading, speed, controls, a_lat=A_LAT, k_max=K_MAX):
    """Integrate controls (..., 60, 2) of acceleration and steering fraction from pos (..., 2),
    heading (...) and speed (...). Returns positions (..., 60, 2); a_lat=None drops the lateral limit."""
    x, y = pos[..., 0], pos[..., 1]
    out = []
    for k in range(FUTURE):
        accel, steer = controls[..., k, 0], controls[..., k, 1]
        new_speed = torch.relu(speed + accel * DT)
        v = 0.5 * (speed + new_speed)
        limit = k_max if a_lat is None else torch.clamp(a_lat / (v * v).clamp(min=1e-6), max=k_max)
        kappa = steer * limit
        ds = v * DT
        mid = heading + 0.5 * kappa * ds
        x, y = x + ds * torch.cos(mid), y + ds * torch.sin(mid)
        heading, speed = heading + kappa * ds, new_speed
        out.append(torch.stack([x, y], dim=-1))
    return torch.stack(out, dim=-2)


def fit_controls(history, target, a_max=A_MAX, k_max=K_MAX, a_lat=A_LAT, steps=1500, control_step=1):
    """Per-scene controls within the bounds that best reproduce target (N, 60, 2) under the ADE loss.

    Projected Adam with step sizes of 0.1 m/s^2 and 0.005 1/m of curvature (at the low-speed limit),
    cosine-decayed, so the result does not depend on the tanh parameterization the model uses. 3000 steps instead of
    1500 lower the vehicle FDE floor by under 1 cm. control_step > 1 fits knots, as the head with
    that control_step outputs them.
    """
    pos, heading, speed = initial_state(history)
    weights = interpolation(control_step) if control_step > 1 else None
    controls = (lambda c: interpolate(c, weights)) if weights is not None else (lambda c: c)
    n = FUTURE // control_step + (control_step > 1)
    unit = torch.tensor([0.1, 0.005 / k_max], device=target.device)  # Adam moves u by ~lr, controls by ~lr * unit
    u = torch.zeros(len(target), n, 2, device=target.device, requires_grad=True)
    limit = torch.tensor([a_max, 1.0], device=target.device) / unit
    opt = torch.optim.Adam([u], lr=1.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        err = (rollout(pos, heading, speed, controls(u * unit), a_lat, k_max) - target).norm(dim=-1)
        opt.zero_grad()
        err.mean(dim=-1).sum().backward()  # scenes are independent: sum keeps each one's step size
        opt.step()
        sched.step()
        with torch.no_grad():
            u.copy_(torch.maximum(torch.minimum(u, limit), -limit))
    with torch.no_grad():
        return rollout(pos, heading, speed, controls(u * unit), a_lat, k_max)


def check(n, control_step=1):
    """Fit controls to n vehicle-like and n other focal futures of the training subset, under loose
    reference bounds and under the model's bounds, with knots every control_step steps."""
    import yaml

    from config import PROJECT_ROOT
    from data import load_focal, split_meta, split_rows
    from models import pick_device

    protocol = yaml.safe_load((PROJECT_ROOT / "configs" / "transformer.yaml").read_text())["protocol"]
    rows, _ = split_rows(protocol, split_meta("train")["n_scenarios"])
    data = load_focal("train", rows)
    device = pick_device()
    rng = np.random.default_rng(0)
    for group, mask in (("vehicle-like", data["vehicle_like"]), ("other", ~data["vehicle_like"])):
        idx = np.sort(rng.choice(np.flatnonzero(mask), n, replace=False))
        history = torch.from_numpy(data["history"][idx]).to(device)
        target = torch.from_numpy(data["target"][idx, :, 0:2]).to(device)
        for a_max, k_max, a_lat in ((20.0, 2.0, None), (A_MAX, K_MAX, A_LAT)):
            t0 = time.time()
            err = (fit_controls(history, target, a_max, k_max, a_lat, control_step=control_step) - target)
            err = err.norm(dim=-1).cpu().numpy()
            ade, fde = err.mean(1), err[:, -1]
            print(f"{group:12s} n={n} step {control_step} |a|<={a_max:g} |k|<={k_max:g} a_lat<={a_lat} "
                  f"ADE mean {ade.mean():.3f} p99 {np.percentile(ade, 99):.3f} | "
                  f"FDE mean {fde.mean():.3f} p99 {np.percentile(fde, 99):.3f} "
                  f"> 0.5 m {(fde > 0.5).mean():.2%} > 2 m {(fde > 2).mean():.2%}  ({time.time() - t0:.0f}s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fit controls to ground-truth futures")
    parser.add_argument("--n", type=int, default=2000, help="scenes per group for --check")
    parser.add_argument("--control-step", type=int, default=1, help="fit knots every this many steps")
    args = parser.parse_args()
    if args.check:
        check(args.n, args.control_step)


if __name__ == "__main__":
    main()
