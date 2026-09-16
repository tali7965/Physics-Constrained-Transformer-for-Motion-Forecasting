"""Download Argoverse 2 motion-forecasting scenarios from the public S3 bucket.

Usage:
    python src/download.py --split val --limit 100
    python src/download.py --split train --limit 20000
    python src/download.py --split test            # whole split

Resumable: scenarios that already have both files on disk are skipped, so
re-running after an interruption only fetches what is missing.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from config import load_config

BUCKET = "s3://argoverse/datasets/av2/motion-forecasting"
SPLITS = ("train", "val", "test")
MEAN_SCENARIO_BYTES = {"train": 247 * 1024, "val": 247 * 1024, "test": 172 * 1024}


def s5cmd(*args, capture=False):
    env = {**os.environ, "AWS_REGION": "us-east-1"}
    cmd = ["s5cmd", "--no-sign-request", *args]
    return subprocess.run(cmd, env=env, check=True, text=True, capture_output=capture)


def list_scenario_ids(split, manifest_dir):
    manifest = manifest_dir / f"{split}.txt"
    if manifest.exists():
        ids = manifest.read_text().split()
        print(f"[{split}] {len(ids)} scenario ids from cached manifest {manifest}")
        return ids

    print(f"[{split}] listing bucket (train takes a few minutes)...")
    out = s5cmd("ls", f"{BUCKET}/{split}/", capture=True).stdout
    ids = sorted(line.split()[-1].rstrip("/") for line in out.splitlines() if line.strip())
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(ids) + "\n")
    print(f"[{split}] {len(ids)} scenario ids, manifest cached at {manifest}")
    return ids


def fmt_size(n_bytes):
    return f"{n_bytes / 1024**3:.2f} GB" if n_bytes >= 1024**3 else f"{n_bytes / 1024**2:.1f} MB"


def is_complete(scenario_dir, sid):
    return (scenario_dir / f"scenario_{sid}.parquet").is_file() and (
        scenario_dir / f"log_map_archive_{sid}.json"
    ).is_file()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--limit", type=int, default=None, help="number of scenarios (default: whole split)")
    parser.add_argument("--dest", type=Path, default=None, help="raw data dir (default: config raw_dir)")
    parser.add_argument("--workers", type=int, default=64, help="parallel s5cmd transfers")
    args = parser.parse_args()

    raw_dir = args.dest or load_config()["raw_dir"]
    split_dir = raw_dir / args.split

    ids = list_scenario_ids(args.split, raw_dir / "manifests")
    if args.limit is not None:
        if args.limit > len(ids):
            sys.exit(f"--limit {args.limit} exceeds split size {len(ids)}")
        ids = ids[: args.limit]

    todo = [sid for sid in ids if not is_complete(split_dir / sid, sid)]
    done = len(ids) - len(todo)
    est = fmt_size(len(todo) * MEAN_SCENARIO_BYTES[args.split])
    print(f"[{args.split}] target {len(ids)} scenarios -> {split_dir}")
    print(f"[{args.split}] {done} already complete, {len(todo)} to download (~{est})")

    if todo:
        split_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".s5cmd", delete=False) as f:
            for sid in todo:
                f.write(f'cp "{BUCKET}/{args.split}/{sid}/*" "{split_dir / sid}/"\n')
            cmd_file = f.name
        t0 = time.time()
        try:
            s5cmd("--numworkers", str(args.workers), "run", cmd_file)
        finally:
            os.unlink(cmd_file)
        print(f"[{args.split}] transfer finished in {time.time() - t0:.0f}s")

    # Verify on disk rather than trusting s5cmd's exit code.
    incomplete = [sid for sid in ids if not is_complete(split_dir / sid, sid)]
    total_bytes = sum(
        p.stat().st_size for sid in ids for p in (split_dir / sid).glob("*") if p.is_file()
    )
    print(f"[{args.split}] complete: {len(ids) - len(incomplete)}/{len(ids)}  on disk: {fmt_size(total_bytes)}")
    if incomplete:
        sys.exit(f"[{args.split}] {len(incomplete)} incomplete scenarios, e.g. {incomplete[:3]} -- re-run to resume")
    print(f"[{args.split}] OK")


if __name__ == "__main__":
    main()
