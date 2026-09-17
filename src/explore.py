"""Measure and report the raw Argoverse 2 scenario format.

Usage:
    python src/explore.py --split val --limit 100
    python src/explore.py --split val                # whole split

Scans scenarios on disk and prints aggregate facts (timeline, agents, focal
kinematics, map elements). The report is also written to outputs/explore_<split>.txt.
Numbers in the README "Data" section come from this output.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet
from av2.map.map_api import ArgoverseStaticMap

from config import PROJECT_ROOT, load_config
from download import SPLITS, scenario_files


def pct(values, q):
    return float(np.percentile(values, q))


def dist(values):
    v = np.asarray(values)
    return f"min {v.min():g}  median {np.median(v):g}  p95 {pct(v, 95):g}  max {v.max():g}"


def counter_lines(counter, total=None):
    lines = []
    for key, n in counter.most_common():
        share = f"  ({n / total:.1%})" if total else ""
        lines.append(f"    {key:<22} {n:>9,}{share}")
    return lines


def scan(split_dir, ids):
    timesteps, dt_ns, observed, future = Counter(), Counter(), Counter(), Counter()
    tracks_per_scene, focal_speeds = [], []
    object_types, categories, focal_types, cities = Counter(), Counter(), Counter(), Counter()
    full_length_tracks = total_tracks = 0
    lanes_per_scene, boundary_pts, centerline_pts = [], [], Counter()
    lane_types, intersection, crossings, drivable = Counter(), Counter(), [], []
    load_s = 0.0

    for sid in ids:
        parquet, map_json = scenario_files(split_dir / sid, sid)
        t0 = time.perf_counter()
        sc = load_argoverse_scenario_parquet(parquet)
        avm = ArgoverseStaticMap.from_json(map_json)
        load_s += time.perf_counter() - t0

        # Timeline
        timesteps[len(sc.timestamps_ns)] += 1
        for d in np.unique(np.diff(sc.timestamps_ns)):
            dt_ns[int(d)] += 1
        cities[sc.city_name] += 1

        # Agents
        tracks_per_scene.append(len(sc.tracks))
        total_tracks += len(sc.tracks)
        for tr in sc.tracks:
            object_types[tr.object_type.name] += 1
            categories[tr.category.name] += 1
            if len(tr.object_states) == len(sc.timestamps_ns):
                full_length_tracks += 1
            if tr.track_id == sc.focal_track_id:
                focal_types[tr.object_type.name] += 1
                n_obs = sum(s.observed for s in tr.object_states)
                observed[n_obs] += 1
                future[len(tr.object_states) - n_obs] += 1
                focal_speeds.extend(float(np.hypot(*s.velocity)) for s in tr.object_states)

        # Map
        segs = avm.get_scenario_lane_segments()
        lanes_per_scene.append(len(segs))
        for seg in segs:
            lane_types[seg.lane_type.name] += 1
            intersection[seg.is_intersection] += 1
            boundary_pts.append(len(seg.right_lane_boundary.xyz))
            centerline_pts[len(avm.get_lane_segment_centerline(seg.id))] += 1
        crossings.append(len(avm.get_scenario_ped_crossings()))
        drivable.append(len(avm.get_scenario_vector_drivable_areas()))

    n = len(ids)
    lines = [
        f"Scenarios scanned: {n:,}",
        "",
        "Timeline",
        f"  timesteps per scenario:      {dict(timesteps)}",
        f"  sampling interval (ns):      {dict(dt_ns)}",
        f"  focal observed steps:        {dict(observed)}",
        f"  focal future steps:          {dict(future)}",
        "",
        "Agents",
        f"  tracks per scenario:         {dist(tracks_per_scene)}",
        f"  tracks present at all steps: {full_length_tracks:,} / {total_tracks:,} ({full_length_tracks / total_tracks:.1%})",
        "  object types (all tracks):",
        *counter_lines(object_types, total_tracks),
        "  track categories (all tracks):",
        *counter_lines(categories, total_tracks),
        "  focal track object types:",
        *counter_lines(focal_types, n),
        "",
        "Focal-agent speed (m/s, all timesteps)",
        f"  p5 {pct(focal_speeds, 5):.2f}  p50 {pct(focal_speeds, 50):.2f}  p95 {pct(focal_speeds, 95):.2f}  max {max(focal_speeds):.2f}",
        "",
        "Map",
        f"  lane segments per scenario:  {dist(lanes_per_scene)}",
        f"  raw boundary points per lane: {dist(boundary_pts)}",
        f"  api centerline points per lane: {dict(centerline_pts)}",
        "  lane types:",
        *counter_lines(lane_types, sum(lane_types.values())),
        f"  in intersection:             {dict(intersection)}",
        f"  ped crossings per scenario:  {dist(crossings)}",
        f"  drivable areas per scenario: {dist(drivable)}",
        "",
        "Cities",
        *counter_lines(cities, n),
        "",
        f"Load time: {load_s:.0f}s total, {1000 * load_s / n:.0f} ms per scenario (parquet + map json)",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--limit", type=int, default=None, help="number of scenarios (default: whole split)")
    args = parser.parse_args()

    cfg = load_config()
    split_dir = cfg["raw_dir"] / args.split
    manifest = cfg["raw_dir"] / "manifests" / f"{args.split}.txt"
    if not manifest.is_file():
        sys.exit(f"no manifest at {manifest} -- run src/download.py --split {args.split} first")
    ids = manifest.read_text().split()
    if args.limit is not None:
        ids = ids[: args.limit]

    print(f"[{args.split}] scanning {len(ids):,} scenarios in {split_dir}")
    t0 = time.time()
    report = scan(split_dir, ids)
    print(f"[{args.split}] done in {time.time() - t0:.0f}s\n")
    print(report)

    out = PROJECT_ROOT / "outputs" / f"explore_{args.split}.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report + "\n")
    print(f"\nreport written to {out}")


if __name__ == "__main__":
    main()
