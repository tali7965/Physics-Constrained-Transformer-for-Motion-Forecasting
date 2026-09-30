"""Row selection and in-RAM loading of preprocessed focal-agent data."""

import json

import numpy as np

from config import load_config

VEHICLE_LIKE = ("VEHICLE", "BUS", "MOTORCYCLIST")
SCENE_FIELDS = ("agents", "agent_valid", "agent_type", "lanes", "lane_type", "lane_intersection")


def split_meta(split):
    return json.loads((load_config()["processed_dir"] / split / "meta.json").read_text())


def split_rows(protocol, n_train):
    """Deterministic (train_rows, dev_rows) of the train split, both sorted.

    A seeded permutation: the first dev_size rows are the dev set, the next subset_size rows the
    training subset, of which the first `fraction` is used, so smaller fractions are nested.
    """
    perm = np.random.default_rng(protocol["seed"]).permutation(n_train)
    dev_size, subset_size = protocol["dev_size"], protocol["subset_size"]
    if dev_size + subset_size > n_train:
        raise ValueError(f"dev {dev_size} + subset {subset_size} exceeds {n_train} train rows")
    subset = perm[dev_size : dev_size + subset_size]
    train = subset[: int(round(protocol["fraction"] * subset_size))]
    return np.sort(train), np.sort(perm[:dev_size])


def load_focal(split, rows=None, scene=False):
    """Focal-agent arrays of a processed split, copied into RAM.

    Returns a dict with history (N, 50, 5), target (N, 60, 5) or None for test, focal_type (N,),
    vehicle_like (N,) bool, and rows (N,) -- the processed row numbers. With scene=True it also
    holds the full-scene arrays named in SCENE_FIELDS (about 84 KB per scenario).
    """
    d = load_config()["processed_dir"] / split
    meta = split_meta(split)
    rows = np.arange(meta["n_scenarios"]) if rows is None else np.asarray(rows)

    def read(name, *index):
        return np.ascontiguousarray(np.load(d / f"{name}.npy", mmap_mode="r")[(rows, *index)])

    history = read("agents", 0)
    assert read("agent_valid", 0).all(), f"{split}: focal history has invalid steps"
    focal_type = read("agent_type", 0)
    types = meta["object_types"]
    data = {
        "history": history,
        "target": read("target") if (d / "target.npy").exists() else None,
        "focal_type": focal_type,
        "vehicle_like": np.isin(focal_type, [types.index(t) for t in VEHICLE_LIKE]),
        "rows": rows,
    }
    if scene:
        data.update({name: read(name) for name in SCENE_FIELDS})
    return data
