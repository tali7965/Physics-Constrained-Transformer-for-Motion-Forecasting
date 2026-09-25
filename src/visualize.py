"""Plot preprocessed scenes next to the raw scenes they came from.

Usage:
    python src/visualize.py --split val --index 0 3 17
    python src/visualize.py --split val --random 6 --seed 0

For each row, writes outputs/figures/<split>_<row>_<scenario-id>.png with two panels:
  left:  the raw scene in the global (city) frame, read with av2, with the processed arrays
         mapped back to global coordinates drawn on top as dots. The two must coincide.
  right: the processed scene in the focal-agent frame, as a model will see it.
Also prints the largest raw-vs-round-trip error per row and exits non-zero above 1 mm.
"""

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet
from av2.map.map_api import ArgoverseStaticMap

from config import PROJECT_ROOT, load_config
from download import SPLITS, scenario_files
from preprocess import CURRENT, FUTURE, HISTORY, rotate, to_global_frame, track_states, wrap_angle

MAX_POSITION_ERROR_M = 1e-3

# Default validated categorical palette (light), slots 1-3, plus neutral chart ink.
SURFACE, INK, INK_2, MUTED, GRID, LANE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FOCAL, FOCAL_FUTURE, OTHERS = "#2a78d6", "#eb6834", "#1baf7a"


def load_processed(split):
    d = load_config()["processed_dir"] / split
    meta = json.loads((d / "meta.json").read_text())
    arrays = {p.stem: np.load(p, mmap_mode="r") for p in d.glob("*.npy")}
    return meta, arrays


def to_global_states(states, origin, theta):
    """Inverse of preprocess.to_frame_states for (..., 5) x, y, vx, vy, heading."""
    out = np.empty(states.shape)
    out[..., 0:2] = to_global_frame(states[..., 0:2], origin, theta)
    out[..., 2:4] = rotate(states[..., 2:4], theta)
    out[..., 4] = wrap_angle(states[..., 4] + theta)
    return out


def state_errors(raw, back):
    """Max position (m), velocity (m/s) and heading (rad) differences between (T, 5) arrays."""
    if len(raw) == 0:
        return np.zeros(3)
    return np.array([
        np.linalg.norm(raw[:, 0:2] - back[:, 0:2], axis=1).max(),
        np.linalg.norm(raw[:, 2:4] - back[:, 2:4], axis=1).max(),
        np.abs(wrap_angle(raw[:, 4] - back[:, 4])).max(),
    ])


def round_trip_errors(a, row, sc, avm):
    """Map the processed row back to the global frame and compare it with the raw scenario."""
    origin, theta = a["origin"][row], float(a["theta"][row])
    n_agents = int((a["agent_type"][row] >= 0).sum())
    n_lanes = int((a["lane_type"][row] >= 0).sum())

    # Agents: slot 0 is the focal track; other slots are matched to the raw track whose t=49
    # position is nearest (track ids are not stored). Valid masks must agree exactly.
    raw_tracks = {}
    for tr in sc.tracks:
        feats, valid = track_states(tr, 0, HISTORY)
        if valid[CURRENT]:
            raw_tracks[tr.track_id] = (feats, valid)
    others = [v for tid, v in raw_tracks.items() if tid != sc.focal_track_id]
    others_now = np.array([f[CURRENT, :2] for f, _ in others]).reshape(-1, 2)
    agent_err = np.zeros(3)
    for k in range(n_agents):
        valid = np.asarray(a["agent_valid"][row, k])
        back = to_global_states(a["agents"][row, k].astype(np.float64), origin, theta)
        if k == 0:
            raw_feats, raw_valid = raw_tracks[sc.focal_track_id]
        else:
            j = int(np.argmin(np.linalg.norm(others_now - back[CURRENT, :2], axis=1)))
            raw_feats, raw_valid = others[j]
        if not (raw_valid == valid).all():
            return {"agents": np.full(3, np.inf), "lanes": np.inf, "target": np.inf}
        agent_err = np.maximum(agent_err, state_errors(raw_feats[valid], back[valid]))

    # Lanes: each processed lane against the closest raw centerline (max point distance).
    raw_cl = np.stack([avm.get_lane_segment_centerline(s.id)[:, :2] for s in avm.get_scenario_lane_segments()])
    back = to_global_frame(a["lanes"][row, :n_lanes].astype(np.float64), origin, theta)
    lane_err = float(np.linalg.norm(raw_cl[None] - back[:, None], axis=-1).max(-1).min(1).max())

    target_err = np.nan
    if "target" in a:
        focal = next(t for t in sc.tracks if t.track_id == sc.focal_track_id)
        raw_future, _ = track_states(focal, HISTORY, FUTURE)
        back = to_global_states(a["target"][row].astype(np.float64), origin, theta)
        target_err = float(state_errors(raw_future, back)[0])
    return {"agents": agent_err, "lanes": lane_err, "target": target_err}


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(LANE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")


def plot_row(a, meta, row, sc, avm, path):
    origin, theta = a["origin"][row], float(a["theta"][row])
    types = a["agent_type"][row]
    n_agents, n_lanes = int((types >= 0).sum()), int((a["lane_type"][row] >= 0).sum())
    agents, valid = a["agents"][row].astype(np.float64), np.asarray(a["agent_valid"][row])
    lanes = a["lanes"][row, :n_lanes].astype(np.float64)
    target = a["target"][row].astype(np.float64) if "target" in a else None
    radius = meta["params"]["radius_m"]

    # Window around the focal agent: its history and future plus a margin, at least +-40 m.
    focal_pts = agents[0, valid[0], :2] if target is None else np.vstack([agents[0, valid[0], :2], target[:, :2]])
    half = max(40.0, float(np.abs(focal_pts).max()) + 20.0)

    fig, (ax_g, ax_l) = plt.subplots(1, 2, figsize=(15, 7.5), facecolor=SURFACE)
    focal_type = meta["object_types"][types[0]]
    fig.suptitle(
        f"{meta['split']} row {row}  ·  {a['scenario_id'][row]}  ·  {a['city'][row]}  ·  focal {focal_type}"
        f"  ·  {n_agents} agents, {n_lanes} lanes kept",
        color=INK, fontsize=12, x=0.01, ha="left",
    )

    # Left: raw data in the global frame, processed data mapped back on top.
    for seg in avm.get_scenario_lane_segments():
        cl = avm.get_lane_segment_centerline(seg.id)[:, :2]
        ax_g.plot(cl[:, 0], cl[:, 1], color=LANE, lw=1, zorder=1)
    for tr in sc.tracks:
        feats, v = track_states(tr, 0, HISTORY)
        if v.any() and tr.track_id != sc.focal_track_id:
            ax_g.plot(feats[v, 0], feats[v, 1], color=OTHERS, lw=1.2, zorder=2)
    focal_track = next(t for t in sc.tracks if t.track_id == sc.focal_track_id)
    feats, v = track_states(focal_track, 0, HISTORY)
    ax_g.plot(feats[v, 0], feats[v, 1], color=FOCAL, lw=2, zorder=3, label="raw focal history")
    if target is not None:
        future, _ = track_states(focal_track, HISTORY, FUTURE)
        ax_g.plot(future[:, 0], future[:, 1], color=FOCAL_FUTURE, lw=2, zorder=3, label="raw focal future")
    ax_g.plot([], [], color=OTHERS, lw=1.2, label="raw other agents")
    ax_g.plot([], [], color=LANE, lw=1, label="raw lane centerlines")
    back = [to_global_frame(agents[:n_agents][valid[:n_agents]][:, :2], origin, theta),
            to_global_frame(lanes.reshape(-1, 2), origin, theta)]
    if target is not None:
        back.append(to_global_frame(target[:, :2], origin, theta))
    back = np.vstack(back)
    ax_g.scatter(back[:, 0], back[:, 1], s=1.5, color=INK, zorder=4, label="processed, mapped back")
    ax_g.add_patch(plt.Circle(origin, radius, fill=False, ls="--", lw=0.8, color=MUTED, zorder=1))
    ax_g.set_xlim(origin[0] - half, origin[0] + half)
    ax_g.set_ylim(origin[1] - half, origin[1] + half)
    style(ax_g, "Raw scene, global frame (m)")
    ax_g.legend(loc="upper left", fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK_2)

    # Right: the processed arrays as stored, focal-agent frame.
    for k in range(n_lanes):
        ls = "--" if a["lane_intersection"][row, k] else "-"
        ax_l.plot(lanes[k, :, 0], lanes[k, :, 1], color=LANE, lw=1, ls=ls, zorder=1)
    for k in range(1, n_agents):
        xy = agents[k, valid[k], :2]
        ax_l.plot(xy[:, 0], xy[:, 1], color=OTHERS, lw=1.2, zorder=2)
        ax_l.scatter(*agents[k, CURRENT, :2], s=12, color=OTHERS, zorder=2)
    ax_l.plot(*agents[0, valid[0], :2].T, color=FOCAL, lw=2, zorder=3, label="focal history")
    if target is not None:
        ax_l.plot(target[:, 0], target[:, 1], color=FOCAL_FUTURE, lw=2, zorder=3, label="focal future (target)")
    ax_l.annotate("", xy=(8, 0), xytext=(0, 0), zorder=4,
                  arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
    ax_l.plot([], [], color=OTHERS, lw=1.2, label="other agents (dot = t=49)")
    ax_l.plot([], [], color=LANE, lw=1, label="lanes (dashed = intersection)")
    ax_l.add_patch(plt.Circle((0, 0), radius, fill=False, ls="--", lw=0.8, color=MUTED, zorder=1))
    ax_l.set_xlim(-half, half)
    ax_l.set_ylim(-half, half)
    style(ax_l, "Processed scene, focal-agent frame (m); arrow = focal heading at t=49")
    ax_l.legend(loc="upper left", fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK_2)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=SPLITS)
    pick = parser.add_mutually_exclusive_group(required=True)
    pick.add_argument("--index", type=int, nargs="+", help="processed row numbers")
    pick.add_argument("--random", type=int, help="number of random rows")
    parser.add_argument("--seed", type=int, default=0, help="seed for --random")
    args = parser.parse_args()

    meta, a = load_processed(args.split)
    n = meta["n_scenarios"]
    if args.index:
        rows = args.index
    else:
        rows = sorted(np.random.default_rng(args.seed).choice(n, size=args.random, replace=False).tolist())
    if max(rows) >= n or min(rows) < 0:
        sys.exit(f"rows must be in [0, {n})")

    split_dir = load_config()["raw_dir"] / args.split
    out_dir = PROJECT_ROOT / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    worst = 0.0
    for row in rows:
        sid = str(a["scenario_id"][row])
        parquet, map_json = scenario_files(split_dir / sid, sid)
        sc = load_argoverse_scenario_parquet(parquet)
        avm = ArgoverseStaticMap.from_json(map_json)
        err = round_trip_errors(a, row, sc, avm)
        path = out_dir / f"{args.split}_{row}_{sid}.png"
        plot_row(a, meta, row, sc, avm, path)
        pos, vel, hdg = err["agents"]
        print(f"row {row:>6}  agents {pos:.1e} m, {vel:.1e} m/s, {hdg:.1e} rad  lanes {err['lanes']:.1e} m"
              f"  target {err['target']:.1e} m  -> {path.relative_to(PROJECT_ROOT)}")
        worst = max(worst, pos, err["lanes"], np.nan_to_num(err["target"]))

    print(f"max round-trip position error: {worst:.1e} m (limit {MAX_POSITION_ERROR_M:g} m)")
    if not worst < MAX_POSITION_ERROR_M:
        sys.exit("FAILED: processed data does not map back onto the raw scene")
    print("OK")


if __name__ == "__main__":
    main()
