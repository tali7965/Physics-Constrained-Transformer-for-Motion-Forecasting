# Physics-Constrained Transformer for Motion Forecasting

**Author:** Ali Taheri

Vehicle trajectory forecasting on [Argoverse 2](https://www.argoverse.org/av2.html). Instead of
regressing raw `(x, y)` coordinates, the model predicts **control inputs** (acceleration, curvature)
that are integrated through a **differentiable kinematic bicycle model**, so every predicted
trajectory is kinematically drivable by construction.

The central experiment compares this physics-constrained decoder against an unconstrained coordinate
decoder on accuracy, trajectory feasibility, and data efficiency.

Full scope and timeline: [`VISION_PROJECT_SCOPE.md`](VISION_PROJECT_SCOPE.md).

## Status

Phase 1 (Setup & Data) — in progress. No model code yet.

## Setup

Requires Python 3.12: `av2` publishes macOS arm64 wheels for CPython 3.9–3.12 only, so Homebrew's
3.13/3.14 will not work.

```bash
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Layout

```
configs/preprocess.yaml   data paths and preprocessing parameters
src/config.py             config loading, data_root resolution
src/download.py           fetch scenarios from the public Argoverse S3 bucket
src/explore.py            measure and report the raw scenario format
src/preprocess.py         raw scenarios -> agent-centric tensors
src/visualize.py          scene plots, and visual validation of the frame transform
data/                     raw and preprocessed data (gitignored)
outputs/                  figures and results (gitignored)
```

The dataset does not live in the repo. `data_root` defaults to `data/`; point it elsewhere with:

```bash
export AV2_DATA_ROOT=/Volumes/<ssd>/av2
```

## Data

_Measured scenario format is recorded here in Step 4 (timesteps, sampling rate, observed/future
split, object types, map elements)._

## Results

_Baseline and model results tables land here in Phase 2 onward._
