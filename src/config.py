"""Config loading and path resolution."""

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "preprocess.yaml"


def load_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())

    # AV2_DATA_ROOT lets the dataset move to an external SSD without editing configs.
    data_root = Path(os.environ.get("AV2_DATA_ROOT", cfg["data_root"]))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    cfg["data_root"] = data_root
    cfg["raw_dir"] = data_root / cfg["raw_subdir"]
    cfg["processed_dir"] = data_root / cfg["processed_subdir"]
    return cfg


if __name__ == "__main__":
    for key, value in load_config().items():
        print(f"{key}: {value}")
