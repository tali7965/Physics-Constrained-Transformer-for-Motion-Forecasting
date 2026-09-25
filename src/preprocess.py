"""Preprocess Argoverse 2 scenarios into fixed-shape, focal-agent-centric arrays.

Usage:
    python src/preprocess.py --split val --limit 200 --workers 1   # quick check
    python src/preprocess.py --split train                         # whole split

Frame: origin at the focal agent's position at the last observed step (t=49), x-axis along its
recorded heading. Output: one .npy file per field under <data_root>/processed/<split>/, where
row i is the i-th scenario of the split manifest, plus meta.json with the parameters and
truncation counts. The split is built in <split>.partial/ and swapped in only on success.
"""

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import numpy as np
from av2.datasets.motion_forecasting.data_schema import ObjectType
from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet
from av2.map.lane_segment import LaneType
from av2.map.map_api import ArgoverseStaticMap
from tqdm import tqdm

from config import PROJECT_ROOT, load_config
from download import SPLITS, scenario_files

# Dataset facts, measured constant over every scenario (see README "Data").
HISTORY = 50            # observed steps, 5 s at 10 Hz
FUTURE = 60             # future steps, 6 s
CURRENT = HISTORY - 1   # t=49, the reference step of the agent frame
CENTERLINE_POINTS = 10  # av2 API centerline length
FEATURES = ("x", "y", "vx", "vy", "heading")

OBJECT_TYPES = list(ObjectType)
LANE_TYPES = list(LaneType)


def rotate(xy, angle):
    """Rotate (..., 2) vectors counter-clockwise by `angle` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1]], axis=-1)


def to_agent_frame(xy, origin, theta):
    """Global (..., 2) points -> frame with its origin at `origin` and x-axis at angle `theta`."""
    return rotate(xy - origin, -theta)


def to_global_frame(xy, origin, theta):
    """Inverse of to_agent_frame."""
    return rotate(xy, theta) + origin


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def track_states(track, first, count):
    """(count, 5) array of x, y, vx, vy, heading for timesteps first..first+count-1, plus mask."""
    feats = np.zeros((count, len(FEATURES)))
    valid = np.zeros(count, dtype=bool)
    for s in track.object_states:
        i = s.timestep - first
        if 0 <= i < count:
            feats[i] = (*s.position, *s.velocity, s.heading)
            valid[i] = True
    return feats, valid


def to_frame_states(feats, valid, origin, theta):
    """Transform x, y, vx, vy, heading into the agent frame; invalid steps become zeros."""
    out = np.empty_like(feats)
    out[:, 0:2] = to_agent_frame(feats[:, 0:2], origin, theta)
    out[:, 2:4] = rotate(feats[:, 2:4], -theta)
    out[:, 4] = wrap_angle(feats[:, 4] - theta)
    out[~valid] = 0.0
    return out


def field_specs(params, with_target):
    """Per-scenario shape and dtype of every stored array field."""
    a, l = params["max_agents"], params["max_lanes"]
    specs = {
        "agents": ((a, HISTORY, len(FEATURES)), np.float32),
        "agent_valid": ((a, HISTORY), np.bool_),
        "agent_type": ((a,), np.int8),
        "lanes": ((l, CENTERLINE_POINTS, 2), np.float32),
        "lane_type": ((l,), np.int8),
        "lane_intersection": ((l,), np.bool_),
        "origin": ((2,), np.float64),
        "theta": ((), np.float64),
    }
    if with_target:
        specs["target"] = ((FUTURE, len(FEATURES)), np.float32)
    return specs


def process_scenario(split_dir, sid, params, with_target):
    """Load one raw scenario and return its fixed-shape arrays plus bookkeeping."""
    parquet, map_json = scenario_files(split_dir / sid, sid)
    sc = load_argoverse_scenario_parquet(parquet)
    avm = ArgoverseStaticMap.from_json(map_json)
    assert len(sc.timestamps_ns) == HISTORY + FUTURE, f"{sid}: {len(sc.timestamps_ns)} timestamps"

    radius, max_agents, max_lanes = params["radius_m"], params["max_agents"], params["max_lanes"]
    focal = next(t for t in sc.tracks if t.track_id == sc.focal_track_id)
    focal_now = next((s for s in focal.object_states if s.timestep == CURRENT), None)
    assert focal_now is not None, f"{sid}: focal agent has no state at t={CURRENT}"
    origin = np.asarray(focal_now.position, dtype=np.float64)
    theta = float(focal_now.heading)

    # Agents: present at t=CURRENT and inside the radius, nearest first, focal in slot 0.
    candidates = []
    for tr in sc.tracks:
        if tr.track_id == sc.focal_track_id:
            continue
        now = next((s for s in tr.object_states if s.timestep == CURRENT), None)
        if now is None:
            continue
        dist = float(np.linalg.norm(np.asarray(now.position) - origin))
        if dist < radius:
            candidates.append((dist, tr))
    candidates.sort(key=lambda c: c[0])  # stable: ties keep the parquet order
    kept = [focal] + [tr for _, tr in candidates[: max_agents - 1]]

    specs = field_specs(params, with_target)
    out = {name: np.zeros(shape, dtype) for name, (shape, dtype) in specs.items()}
    out["agent_type"][:] = -1
    out["lane_type"][:] = -1
    for k, tr in enumerate(kept):
        feats, valid = track_states(tr, 0, HISTORY)
        out["agents"][k] = to_frame_states(feats, valid, origin, theta)
        out["agent_valid"][k] = valid
        out["agent_type"][k] = OBJECT_TYPES.index(tr.object_type)

    # Lanes: any centerline point inside the radius, nearest first.
    lanes = []
    for seg in avm.get_scenario_lane_segments():
        cl = avm.get_lane_segment_centerline(seg.id)[:, :2]
        assert len(cl) == CENTERLINE_POINTS, f"{sid}: lane {seg.id} has {len(cl)} centerline points"
        dist = float(np.linalg.norm(cl - origin, axis=1).min())
        if dist < radius:
            lanes.append((dist, seg, cl))
    lanes.sort(key=lambda c: c[0])
    for k, (_, seg, cl) in enumerate(lanes[:max_lanes]):
        out["lanes"][k] = to_agent_frame(cl, origin, theta)
        out["lane_type"][k] = LANE_TYPES.index(seg.lane_type)
        out["lane_intersection"][k] = seg.is_intersection

    out["origin"][:] = origin
    out["theta"][()] = theta

    has_future = any(s.timestep >= HISTORY for s in focal.object_states)
    if with_target:
        feats, valid = track_states(focal, HISTORY, FUTURE)
        assert valid.all(), f"{sid}: focal future has {valid.sum()}/{FUTURE} valid steps"
        out["target"] = to_frame_states(feats, valid, origin, theta).astype(np.float32)
    else:
        assert not has_future, f"{sid}: test scenario unexpectedly contains future states"

    # Invariants of the frame and the selection.
    assert np.abs(out["agents"][0, CURRENT, :2]).max() < 1e-4, f"{sid}: focal not at the origin"
    assert abs(out["agents"][0, CURRENT, 4]) < 1e-6, f"{sid}: focal heading not zero"
    assert out["agent_valid"][: len(kept), CURRENT].all(), f"{sid}: kept agent missing at t={CURRENT}"
    assert all(np.isfinite(v).all() for v in out.values() if v.dtype.kind == "f"), f"{sid}: non-finite"

    return {
        "arrays": out,
        "scenario_id": sid,
        "city": sc.city_name,
        "agent_candidates": len(candidates) + 1,
        "lane_candidates": len(lanes),
    }


def git_commit():
    def git(*args):
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()

    commit = git("rev-parse", "--short", "HEAD")
    return commit + ("-dirty" if git("status", "--porcelain") else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--limit", type=int, default=None, help="first N scenarios of the manifest (default: whole split)")
    parser.add_argument("--workers", type=int, default=8, help="parallel processes (1 = no pool)")
    args = parser.parse_args()

    cfg = load_config()
    params = {key: cfg[key] for key in ("radius_m", "max_agents", "max_lanes")}
    with_target = args.split != "test"
    split_dir = cfg["raw_dir"] / args.split
    ids = (cfg["raw_dir"] / "manifests" / f"{args.split}.txt").read_text().split()
    if args.limit is not None:
        ids = ids[: args.limit]
    n = len(ids)

    out_dir = cfg["processed_dir"] / args.split
    tmp_dir = cfg["processed_dir"] / f"{args.split}.partial"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    specs = field_specs(params, with_target)
    arrays = {
        name: np.lib.format.open_memmap(tmp_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=(n, *shape))
        for name, (shape, dtype) in specs.items()
    }
    scenario_ids, cities, agent_counts, lane_counts = [], [], [], []

    print(f"[{args.split}] preprocessing {n:,} scenarios with {args.workers} worker(s) -> {out_dir}")
    t0 = time.time()
    worker = partial(process_scenario, split_dir, params=params, with_target=with_target)
    pool = Pool(args.workers) if args.workers > 1 else None
    results = pool.imap(worker, ids, chunksize=32) if pool else map(worker, ids)
    try:
        for i, res in enumerate(tqdm(results, total=n, mininterval=10, unit="scn")):
            for name, value in res["arrays"].items():
                arrays[name][i] = value
            scenario_ids.append(res["scenario_id"])
            cities.append(res["city"])
            agent_counts.append(res["agent_candidates"])
            lane_counts.append(res["lane_candidates"])
    finally:
        if pool:
            pool.close()
            pool.join()
    elapsed = time.time() - t0

    for m in arrays.values():
        m.flush()
    del arrays
    assert scenario_ids == ids, "results out of manifest order"
    np.save(tmp_dir / "scenario_id.npy", np.array(scenario_ids))
    np.save(tmp_dir / "city.npy", np.array(cities))

    agent_counts, lane_counts = np.array(agent_counts), np.array(lane_counts)
    meta = {
        "split": args.split,
        "n_scenarios": n,
        "limit": args.limit,
        "params": params,
        "history_steps": HISTORY,
        "future_steps": FUTURE,
        "current_step": CURRENT,
        "features": list(FEATURES),
        "object_types": [t.name for t in OBJECT_TYPES],
        "lane_types": [t.name for t in LANE_TYPES],
        "empty_slot_type": -1,
        "truncated_agents": int((agent_counts > params["max_agents"]).sum()),
        "truncated_lanes": int((lane_counts > params["max_lanes"]).sum()),
        "max_agent_candidates": int(agent_counts.max()),
        "max_lane_candidates": int(lane_counts.max()),
        "seconds": round(elapsed, 1),
        "git_commit": git_commit(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)

    size = sum(p.stat().st_size for p in out_dir.iterdir())
    print(f"[{args.split}] done in {elapsed:.0f}s ({1000 * elapsed / n:.1f} ms per scenario)")
    print(f"[{args.split}] {size / 1024**3:.2f} GB on disk, {size / n / 1024:.1f} KB per scenario")
    for kind, count, cap, peak in (
        ("agents", meta["truncated_agents"], params["max_agents"], meta["max_agent_candidates"]),
        ("lanes", meta["truncated_lanes"], params["max_lanes"], meta["max_lane_candidates"]),
    ):
        print(f"[{args.split}] {kind}: {count:,} scenes over the cap of {cap} ({count / n:.2%}), max candidates {peak}")
    print(f"[{args.split}] OK")


if __name__ == "__main__":
    main()
