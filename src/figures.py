"""Figures for the technical report, written to report/figures/.

Usage:
    python src/figures.py

    data_efficiency.pdf  val brier-minFDE at K=6 (all focal) and off-road @1 (vehicle-like) against the
                         number of training scenes, for both heads (outputs/results/<run>_val.json)
    gradient.pdf         mean |d loss / d head output| per 1 s band of the future, relative to its mean
                         over the 60 steps, for both heads (diagnose.conditioning on the dev rows)
    examples.pdf         both heads' modes on three vehicle-like dev scenes, over the drivable area: a
                         turn in which both heads' most probable modes end within 2 m of the ground
                         truth; a turn in which the physics head's most probable mode turns the right
                         way but only 25-75% as far and leaves the road, while the coordinate head's
                         stays on it; and a straight road on which the physics head starts at least 4
                         degrees off and leaves the road, while the coordinate head stays on it. Each
                         scene is drawn at random (seed 0) from the dev scenes that meet its condition,
                         and an x marks the first off-road point of a most probable mode.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from av2.map.map_api import ArgoverseStaticMap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.path import Path as PolygonPath

from config import PROJECT_ROOT, load_config
from data import load_focal, split_meta, split_rows
from diagnose import BANDS, STRAIGHT_DEG, TURN_DEG, capture, chord, conditioning, signed, turn_angle
from evaluate import load_checkpoint, off_road, predict
from metrics import MISS_THRESHOLD_M
from models import pick_device
import physics
from preprocess import DT, FUTURE, to_agent_frame
from visualize import GRID, INK, INK_2, LANE, MUTED

OUT = PROJECT_ROOT / "report" / "figures"
RESULTS = PROJECT_ROOT / "outputs" / "results"
RUNS = {"coordinate": "transformer", "physics": "physics"}
COLOR = {"coordinate": "#2a78d6", "physics": "#eb6834"}  # categorical palette slots 1 and 2
MARKER = {"coordinate": "o", "physics": "s"}
LABEL = {"coordinate": "coordinate head", "physics": "kinematic head"}  # the report's name for the physics head
DRIVABLE = "#efeee9"
COLUMN, PAGE = 3.5, 7.16  # IEEE column and text widths, inches
HEADING_ERROR_DEG = 4.0  # the top initial-heading-error bin of diagnose.py

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": LANE,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"written {(OUT / name).relative_to(PROJECT_ROOT)}")


def data_efficiency():
    scenes = {"-f10": 5_000, "-f25": 12_500, "": 50_000}
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.55))
    for name, run in RUNS.items():
        res = [json.loads((RESULTS / f"{run}{suffix}_val.json").read_text())["results"] for suffix in scenes]
        brier = [r["all"]["K=6"]["brier-minFDE"] for r in res]
        road = [100 * r["vehicle-like"]["physical"]["off-road@1"] for r in res]
        for ax, y in zip(axes, (brier, road)):
            ax.plot(list(scenes.values()), y, color=COLOR[name], marker=MARKER[name], ms=4, lw=1.5, label=LABEL[name])
    for ax, title in zip(axes, ("brier-minFDE, K=6 (m)", "off-road @1 (%)")):
        ax.set_xscale("log")
        ax.set_xticks(list(scenes.values()), ["5k", "12.5k", "50k"])
        ax.minorticks_off()
        ax.set_xlabel("training scenes")
        ax.set_title(title, loc="left")
    axes[1].set_ylim(bottom=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout(w_pad=1.0)
    save(fig, "data_efficiency.pdf")


def gradient(models, dev, device):
    fig, ax = plt.subplots(figsize=(COLUMN, 1.7))
    centres = [(a + b) / 2 * DT for a, b in BANDS]
    for name, (model, _) in models.items():
        g = conditioning(model, capture(model), dev, device, batches=8, batch_size=64)
        channels = (("x", "-"),) if name == "coordinate" else (("acceleration", "-"), ("steering", "--"))
        for c, (channel, ls) in enumerate(channels):
            rel = g[:, c] / g[:, c].mean()
            ax.plot(centres, [rel[a:b].mean() for a, b in BANDS], color=COLOR[name], ls=ls, lw=1.5,
                    marker=MARKER[name], ms=4, label=f"{LABEL[name]}, {channel}")
    ax.set_yscale("log")
    ax.set_xlim(0, FUTURE * DT)
    ax.set_xlabel("future time (s), 1 s bands")
    ax.set_ylabel("|gradient| / mean")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "gradient.pdf")


def examples(models, dev, rows, device):
    veh = np.flatnonzero(dev["vehicle_like"])
    n = len(veh)
    start, gt = dev["history"][veh, -1, :2], dev["target"][veh, :, :2]
    pred, k_top, top, off = {}, {}, {}, {}
    for name, (model, _) in models.items():
        p, q = predict(model, dev, device)
        pred[name], k_top[name] = p[veh], np.argmax(q[veh], axis=1)
        top[name] = pred[name][np.arange(n), k_top[name]]
        off[name] = off_road("train", rows[veh], top[name][:, None])[:, 0]
    fde = {name: np.linalg.norm(top[name][:, -1] - gt[:, -1], axis=-1) for name in top}
    d_gt = turn_angle(start, gt)
    deg = np.degrees(np.abs(d_gt))
    _, h0, _ = physics.initial_state(torch.from_numpy(dev["history"][veh]).double())
    e0 = np.degrees(np.abs(signed(h0.numpy() - chord(start, gt[:, 9])[0])))
    ratio = turn_angle(start, top["physics"]) / d_gt
    cases = {
        "turn, both heads within 2 m": (deg >= TURN_DEG) & (fde["coordinate"] < MISS_THRESHOLD_M)
                                       & (fde["physics"] < MISS_THRESHOLD_M) & ~off["coordinate"] & ~off["physics"],
        "turn, kinematic head turns too little": (deg >= TURN_DEG) & (ratio > 0.25) & (ratio < 0.75)
                                               & off["physics"] & ~off["coordinate"],
        "straight, initial-heading error": (deg < STRAIGHT_DEG) & (e0 >= HEADING_ERROR_DEG) & off["physics"]
                                           & ~off["coordinate"],
    }
    print("dev scenes per example condition: " + ", ".join(f"{title} {mask.sum()}" for title, mask in cases.items()))
    rng = np.random.default_rng(0)
    picks = [(title, int(rng.choice(np.flatnonzero(mask))), int(mask.sum())) for title, mask in cases.items()]

    cfg = load_config()
    pdir, rdir = cfg["processed_dir"] / "train", cfg["raw_dir"] / "train"
    sid = np.load(pdir / "scenario_id.npy", mmap_mode="r")
    origin, theta = np.load(pdir / "origin.npy", mmap_mode="r"), np.load(pdir / "theta.npy", mmap_mode="r")
    fig, axes = plt.subplots(2, 3, figsize=(PAGE, 3.9))
    for j, (title, i, count) in enumerate(picks):
        d, r = veh[i], rows[veh[i]]
        avm = ArgoverseStaticMap.from_json(rdir / sid[r] / f"log_map_archive_{sid[r]}.json")
        areas = [to_agent_frame(a.xyz[:, :2], origin[r], float(theta[r])) for a in avm.get_scenario_vector_drivable_areas()]
        history = dev["history"][d, -30:, :2]  # the last 3 s
        pts = np.vstack([history, gt[i], pred["coordinate"][i].reshape(-1, 2), pred["physics"][i].reshape(-1, 2)])
        mid, half = (pts.min(0) + pts.max(0)) / 2, max(15.0, (pts.max(0) - pts.min(0)).max() / 2 + 4.0)
        extra = f" ({e0[i]:.1f}°)" if "straight" in title else ""
        print(f"({'abc'[j]}) {title}: dev row {r}, scenario {sid[r]}, one of {count}{extra}")
        for k, name in enumerate(("coordinate", "physics")):
            ax = axes[k, j]
            for poly in areas:
                ax.add_patch(Polygon(poly, closed=True, facecolor=DRIVABLE, edgecolor=LANE, lw=0.4, zorder=0))
            for lane in dev["lanes"][d][dev["lane_type"][d] >= 0]:
                ax.plot(*lane.T, color=LANE, lw=0.5, zorder=1)
            ax.plot(*history.T, color=MUTED, lw=1.6, zorder=2)
            ax.plot(*gt[i].T, color=INK, lw=1.6, zorder=3)
            for m, traj in enumerate(pred[name][i]):
                is_top = m == k_top[name][i]
                ax.plot(*traj.T, color=COLOR[name], lw=1.6 if is_top else 0.7, alpha=1.0 if is_top else 0.6, zorder=4)
            outside = ~np.any([PolygonPath(poly).contains_points(top[name][i]) for poly in areas], axis=0)
            if outside.any():
                ax.plot(*top[name][i][np.argmax(outside)], marker="x", ms=6, mew=1.6, color=INK, zorder=6)
            ax.set_xlim(mid[0] - 1.2 * half, mid[0] + 1.2 * half)
            ax.set_ylim(mid[1] - half, mid[1] + half)
            ax.set_aspect("equal")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            x0, y0 = mid[0] - 1.2 * half + 0.06 * half, mid[1] - half + 0.08 * half
            ax.plot([x0, x0 + 10], [y0, y0], color=INK, lw=1.2, zorder=5)
            ax.text(x0 + 5, y0 + 0.05 * half, "10 m", ha="center", va="bottom", fontsize=6.5, zorder=5)
            if j == 0:
                ax.set_ylabel(LABEL[name])
        axes[0, j].set_title(f"({'abc'[j]}) {title}{extra}", loc="left")
    handles = [Line2D([], [], color=MUTED, lw=1.6, label="history (3 s)"),
               Line2D([], [], color=INK, lw=1.6, label="ground truth"),
               Line2D([], [], color=COLOR["coordinate"], lw=1.6, label=LABEL["coordinate"]),
               Line2D([], [], color=COLOR["physics"], lw=1.6, label=LABEL["physics"]),
               Line2D([], [], color=INK, lw=0, marker="x", ms=6, mew=1.6, label="first off-road point"),
               Patch(facecolor=DRIVABLE, edgecolor=LANE, lw=0.4, label="drivable area")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.4, w_pad=0.4)
    save(fig, "examples.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data_efficiency()
    device = pick_device()
    models = {name: load_checkpoint(PROJECT_ROOT / "outputs" / "runs" / run / "best.pt", device) for name, run in RUNS.items()}
    _, rows = split_rows(models["physics"][1]["config"]["protocol"], split_meta("train")["n_scenarios"])
    dev = load_focal("train", rows, scene=True)
    examples(models, dev, rows, device)
    gradient(models, dev, device)


if __name__ == "__main__":
    main()
