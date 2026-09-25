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

Phase 1 (Setup & Data) — complete: all three splits are preprocessed and visually checked.
Phase 2 (Baselines) — complete: metrics, constant-velocity and LSTM baselines (see Results).
Phase 3 (Transformer model) is next.

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

```text
configs/preprocess.yaml   data paths and preprocessing parameters
configs/lstm.yaml         train/dev protocol and LSTM baseline hyperparameters
src/config.py             config loading, data_root resolution
src/download.py           fetch scenarios from the public Argoverse S3 bucket
src/explore.py            measure and report the raw scenario format
src/preprocess.py         raw scenarios -> agent-centric tensors
src/visualize.py          scene plots, and visual validation of the frame transform
src/data.py               train/dev row selection, in-memory loading of focal-agent data
src/metrics.py            minADE, minFDE, miss rate, brier-minFDE (Argoverse 2 definitions)
src/models.py             constant-velocity baseline and LSTM encoder-decoder
src/train.py              training loop with dev-set model selection
src/evaluate.py           score a model on val (or dev) and write outputs/results/
data/                     raw and preprocessed data (gitignored)
outputs/                  figures and results (gitignored)
```

The dataset does not live in the repo. `data_root` in `configs/preprocess.yaml` points at the
external SSD; override it without editing the config:

```bash
export AV2_DATA_ROOT=/some/other/path
```

## Downloading the data

Argoverse 2 motion forecasting is a public S3 bucket (no account needed). Measured sizes:

| Split | Scenarios | Size |
| --- | --- | --- |
| train | 199,908 | 47.0 GB |
| val | 24,988 | 5.9 GB |
| test | 24,984 | 4.1 GB |

```bash
brew install s5cmd
source .venv/bin/activate

python src/download.py --split val --limit 100      # quick sanity check (~25 MB)
python src/download.py --split val                  # full val, for evaluation
python src/download.py --split test                 # full test, for the leaderboard
python src/download.py --split train --limit 20000  # training subset (~4.8 GB)
```

Scenarios land in `<data_root>/raw/<split>/<scenario-id>/`. The script is resumable: re-running
skips anything already complete and verifies every scenario on disk before reporting `OK`. Without
`--limit` a whole split is fetched. The first run for each split lists the bucket and caches the
scenario ids under `<data_root>/raw/manifests/` (train takes a few minutes to list).

## Data

Measured over the full validation split (24,988 scenarios) with `python src/explore.py --split val`;
the complete report is written to `outputs/explore_val.txt`.

**Timeline** — every scenario has 110 timesteps at 0.1 s (10 Hz): 50 observed (5 s history) and
60 future (6 s horizon). No exceptions in the split.

**Agents** — 2–255 tracks per scenario (median 52). Object types over all tracks: 73.5% VEHICLE,
9.5% PEDESTRIAN, 7.0% STATIC, 4.8% BACKGROUND, then CONSTRUCTION, RIDERLESS_BICYCLE, BUS, CYCLIST,
MOTORCYCLIST, UNKNOWN (each < 2%). Track categories: 81.1% TRACK_FRAGMENT, 11.9% UNSCORED_TRACK,
5.2% SCORED_TRACK, 1.8% FOCAL_TRACK (one per scenario). Every FOCAL, SCORED and UNSCORED track
spans all 110 steps; every TRACK_FRAGMENT is partial.

**Focal track** — 88.6% VEHICLE, 6.3% PEDESTRIAN, 3.4% BUS, 1.2% CYCLIST, 0.5% MOTORCYCLIST.
Speed over all focal timesteps: p50 5.95 m/s, p95 14.86 m/s, max 29.56 m/s (p5 is 0 — stationary
agents are common).

**Map** — 2–281 lane segments per scenario (median 62); 87.7% VEHICLE, 11.4% BIKE, 0.9% BUS lanes;
34% of segments are in an intersection. Raw lane boundaries have 2–59 points (median 3); the
`av2` API returns a fixed 10-point centerline per segment. Median 4 pedestrian crossings and 3
drivable-area polygons per scenario.

**Cities** — miami 26.6%, pittsburgh 21.3%, austin 21.3%, washington-dc 12.8%, dearborn 12.3%,
palo-alto 5.7%.

**Load cost** — 18 ms per scenario (parquet + map JSON), ~9 min wall time for the whole val split.

## Preprocessing

```bash
python src/preprocess.py --split val --limit 200 --workers 1   # quick check
python src/preprocess.py --split val                           # whole split, 8 workers
python src/preprocess.py --split test
python src/preprocess.py --split train
python src/visualize.py --split val --random 6 --seed 0        # figures + round-trip check
```

Each scenario becomes fixed-shape arrays in a frame centred on the focal agent at the last observed
step (t=49), with the x-axis along its recorded heading. The recorded heading is within 4° of the
velocity direction for 99% of moving focal agents, so it is a safe axis.

**Region and caps** — agents present at t=49 and lane segments with any centerline point within
150 m of the focal agent, nearest first, up to 64 agents (focal in slot 0) and 192 lanes. 150 m
covers the farthest focal future measured (149 m). The caps sit just above the measured p99; the
numbers behind them are in `configs/preprocess.yaml`. A scene over a cap loses its farthest
elements, and `meta.json` counts how often that happens:

| Split | Scenarios | On disk | Over agent cap | Over lane cap | Time, 8 workers |
| --- | --- | --- | --- | --- | --- |
| train | 199,908 | 15.72 GB | 0.69% | 0.16% | 23 min |
| val | 24,988 | 1.96 GB | 0.68% | 0.15% | 2 min |
| test | 24,984 | 1.94 GB | 0.62% | 0.18% | 2 min |

**Layout** — `<data_root>/processed/<split>/`, one `.npy` per field, row *i* = *i*-th scenario of
the split manifest. Load with `np.load(path, mmap_mode="r")`.

| File | Shape | dtype | Content |
| --- | --- | --- | --- |
| `agents.npy` | [N, 64, 50, 5] | float32 | history x, y, vx, vy, heading; zeros where invalid |
| `agent_valid.npy` | [N, 64, 50] | bool | step observed |
| `agent_type.npy` | [N, 64] | int8 | index into `object_types` in `meta.json`; -1 = empty slot |
| `lanes.npy` | [N, 192, 10, 2] | float32 | 10-point centerline from the `av2` map API |
| `lane_type.npy` | [N, 192] | int8 | index into `lane_types`; -1 = empty slot |
| `lane_intersection.npy` | [N, 192] | bool | segment lies in an intersection |
| `target.npy` | [N, 60, 5] | float32 | focal future, same features (train and val only) |
| `origin.npy`, `theta.npy` | [N, 2], [N] | float64 | frame origin and heading in city coordinates |
| `scenario_id.npy`, `city.npy` | [N] | str | |
| `meta.json` | | | parameters, type vocabularies, truncation counts, git commit |

Every focal type is kept (`agent_type[:, 0]`). Vehicle-only selection and training subsets are
index choices at training time, and the test split keeps every focal agent for the leaderboard.

**Checks** — every scenario asserts that the focal agent sits at the origin with zero heading, that
kept agents are present at t=49, that all values are finite, and that train/val have 60 future
steps. `src/visualize.py` maps processed rows back to city coordinates and compares them with the
raw scenario; the largest position error seen is under 1e-5 m (float32 precision). Two full val
runs, and two full test runs, produced bit-identical files; re-processing 20 random rows per split
matches the stored arrays exactly. 143 test parquets store `timestep` as float64 instead of int64;
the preprocessing casts it and asserts nothing is lost.

## Results

**Protocol** — a seeded permutation of the train split gives a fixed *dev* set of 5,000 scenarios,
used only for model selection, and a fixed *training subset* of 50,000 scenarios. The 10% and 25%
subsets for the data-efficiency study are prefixes of it, so they are nested. The official val split
(24,988 scenarios) is used only for the numbers reported here. All settings live in
`configs/lstm.yaml`.

**Metrics** — the Argoverse 2 definitions: of the K most probable modes, the best is the one with the
lowest endpoint error; minADE and minFDE are its average and final errors; a miss is a minFDE over
2 m; brier-minFDE adds (1 − p_best)² and ranks the leaderboard at K=6. `src/metrics.py` matches the
`av2` package's per-scenario functions exactly.

**Baselines on val, K=1** (both baselines predict a single trajectory):

| Model | Population | minADE | minFDE | MR |
| --- | --- | --- | --- | --- |
| Constant velocity, tracker velocity | all focal (24,988) | 4.70 | 12.15 | 0.863 |
| Constant velocity, last-step displacement | all focal | 4.40 | 11.66 | 0.855 |
| LSTM encoder–decoder | all focal | **3.15** | **8.33** | **0.821** |
| Constant velocity, tracker velocity | vehicle-like (23,113) | 5.00 | 12.96 | 0.905 |
| Constant velocity, last-step displacement | vehicle-like | 4.69 | 12.43 | 0.899 |
| LSTM encoder–decoder | vehicle-like | **3.31** | **8.78** | **0.856** |

Vehicle-like means VEHICLE, BUS and MOTORCYCLIST focal agents, the population the Argoverse 2 paper
uses for its baselines. Its focal-history-only LSTM (Table 5, beta dataset) reports 3.05 / 8.28 /
0.85, close to ours. The LSTM (270k parameters) uses only the focal agent's history: a 2-layer
encoder and an autoregressive decoder over 60 steps, trained with an ADE loss for 30 epochs, about
5 minutes on the M4 GPU. Training and dev error end almost equal (3.12 vs 3.15 m), so it underfits
rather than overfits.

```bash
python src/evaluate.py --model cv-tracker
python src/evaluate.py --model cv-displacement
python src/train.py --config configs/lstm.yaml
python src/evaluate.py --checkpoint outputs/runs/lstm/best.pt
```
