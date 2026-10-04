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
Phase 3 (Transformer model) — complete: a 1.4M-parameter Transformer with a K=6 coordinate decoder
beats the LSTM on every metric (see Results).
Phase 4 (physics-constrained decoder) — complete: it trains stably and every predicted trajectory is
feasible, at a cost in accuracy that Phase 5 has to explain (see Results).
Phase 5 (experiments) is next.

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
configs/transformer.yaml  the same protocol, Transformer hyperparameters
configs/physics.yaml      the Transformer with the physics head
src/config.py             config loading, data_root resolution
src/download.py           fetch scenarios from the public Argoverse S3 bucket
src/explore.py            measure and report the raw scenario format
src/preprocess.py         raw scenarios -> agent-centric tensors
src/visualize.py          scene plots, frame-transform validation, predicted-mode plots
src/data.py               train/dev row selection, in-memory loading of focal or full-scene data
src/metrics.py            minADE, minFDE, miss rate, brier-minFDE (Argoverse 2 definitions), feasibility
src/models.py             constant-velocity baseline, LSTM encoder-decoder, Transformer
src/physics.py            differentiable kinematic bicycle model, control fitting to ground truth
src/train.py              training loop (ADE or winner-takes-all loss), dev-set model selection
src/evaluate.py           score a model on val (or dev), incl. off-road, and write outputs/results/
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

### Transformer

**Model** — every agent (its 50 history steps) and every lane (the 9 segments of its centerline) is
encoded PointNet-style into one token: a shared MLP per step or segment, max-pooled, plus type
embeddings; a learned timestep embedding keeps the order of history steps. A 4-layer Transformer
encoder runs joint self-attention over all 64 agent and 192 lane tokens, so agent–agent and
agent–map attention share the same layers; empty slots are masked, and there is no slot positional
encoding, so outputs do not depend on slot order (checked: shuffling slots changes outputs by
3e-14 m in float64). Six mode queries — the focal token plus a learned mode embedding — attend to
the scene in a 2-layer Transformer decoder; an MLP maps each mode to 60 future positions and a
linear layer to a logit. d=128, 8 heads, 1.43M parameters. The trajectory MLP is the part the
Phase 4 physics head replaces.

**Training** — winner-takes-all: the mode with the lowest endpoint error (the metric's rule) gets the
ADE loss, and the logits get a cross-entropy loss towards that mode. Adam, cosine decay, gradient
clipping 1.0, batch 64, 30 epochs, all focal types, same training subset and dev set as the LSTM.
`best.pt` is the epoch with the lowest dev brier-minFDE at K=6. The full run takes 3.1 h on the M4
GPU (116 scenes/s; the 50k scenes, 4.3 GB, are held in RAM).

Mode embeddings must start at the scale of the tokens they are added to: initialised at std 0.02
next to a layer-normed focal token, all six queries were nearly identical and the modes collapsed
(endpoints 0.6 m apart, probabilities stuck at 1/6). At std 1 the modes separate.

**Learning rate** — chosen on the 10% subset (5,000 scenes, 30 epochs), by dev brier-minFDE:

| lr | dev brier-minFDE | dev minFDE, K=6 | dev minADE / minFDE, K=1 |
| --- | --- | --- | --- |
| 3e-4 | 4.18 | 3.59 | 3.63 / 9.00 |
| **1e-3** | **3.95** | **3.29** | **3.51 / 8.75** |
| 3e-3 | 6.04 | 5.39 | 4.38 / 11.25 |

**Results on val** (K=1 uses the most probable mode):

| Model | Population | K | minADE | minFDE | MR | brier-minFDE |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM encoder–decoder | all focal | 1 | 3.15 | 8.33 | 0.821 | |
| Transformer | all focal | 1 | **2.45** | **6.09** | **0.709** | |
| Transformer | all focal | 6 | 1.06 | 1.94 | 0.303 | 2.57 |
| LSTM encoder–decoder | vehicle-like | 1 | 3.31 | 8.78 | 0.856 | |
| Transformer | vehicle-like | 1 | **2.56** | **6.40** | **0.740** | |
| Transformer | vehicle-like | 6 | 1.10 | 2.01 | 0.317 | 2.64 |

The Transformer beats the LSTM at K=1 by 22% on minADE and 27% on minFDE (all focal). Its weak
point is mode scoring: the classification loss ends at 1.64 against 1.79 for uniform guessing, and
the most probable mode is often not the closest one (K=1 minFDE 6.09 vs 1.94 at K=6). Dev metrics
were still improving slightly at epoch 30. In the mode plots some low-probability modes wobble or
leave the lanes, which is what the kinematic decoder of Phase 4 is meant to rule out.

```bash
python src/train.py --config configs/transformer.yaml
python src/evaluate.py --checkpoint outputs/runs/transformer/best.pt
python src/visualize.py --split train --random 6 --seed 0 --checkpoint outputs/runs/transformer/best.pt
```

### Physics-constrained decoder

**Model** — the Transformer above with only `traj_head` changed: per mode it outputs 60 pairs of
controls instead of 60 positions, bounded with `tanh` to an acceleration in ±8 m/s² and a steering
fraction in ±1. `src/physics.py` integrates them through a kinematic bicycle model in curvature form
from the focal agent's state at t=49 (position; heading and speed from the last-step displacement,
the recorded heading below 0.5 m/s). Curvature is the steering fraction times
min(0.2 1/m, 6 m/s² / v²), a 5 m turning radius at low speed and at most 6 m/s² of lateral
acceleration above it; speed cannot go negative. Every mode is therefore drivable by construction.
The parameter count (1.43M), losses, protocol and all other settings are unchanged, and the head is
used for every focal type, pedestrians included. Fitting controls directly to ground-truth futures
(`python src/physics.py --check`, 2,000 vehicle-like training scenes) puts the head's error floor at
0.16 m FDE on average, with 0.55% of futures more than 2 m away.

**Feasibility metrics** — for vehicle-like agents, a trajectory is *infeasible* if its lateral
acceleration (speed × yaw rate) exceeds 8 m/s² at any step while moving faster than 1 m/s. It is
measured at 10 Hz, where ±4 cm of step-to-step zigzag already crosses the limit, and on the
trajectory resampled to 2 Hz, which judges the manoeuvre's shape. A trajectory is *off-road* if any
point lies outside every drivable-area polygon of the scenario's map. Both are reported for the most
probable mode (@1), over all six modes (@6), and for the ground truth.

**Learning rate** — chosen on the 10% subset as for the coordinate head (best coordinate head there:
3.95):

| lr | dev brier-minFDE | dev minFDE, K=6 | dev minADE / minFDE, K=1 |
| --- | --- | --- | --- |
| 1e-4 | 4.39 | 3.75 | 3.53 / 9.29 |
| **3e-4** | **4.19** | **3.53** | 3.94 / 10.29 |
| 1e-3 | 5.81 | 5.20 | 4.20 / 10.69 |

**Results on val** (K=1 uses the most probable mode):

| Model | Population | K | minADE | minFDE | MR | brier-minFDE |
| --- | --- | --- | --- | --- | --- | --- |
| Transformer, coordinate head | all focal | 1 | **2.45** | **6.09** | **0.709** | |
| Transformer, physics head | all focal | 1 | 2.74 | 6.95 | 0.755 | |
| Transformer, coordinate head | all focal | 6 | **1.06** | **1.94** | **0.303** | **2.57** |
| Transformer, physics head | all focal | 6 | 1.25 | 2.55 | 0.431 | 3.16 |
| Transformer, coordinate head | vehicle-like | 1 | **2.56** | **6.40** | **0.740** | |
| Transformer, physics head | vehicle-like | 1 | 2.89 | 7.35 | 0.791 | |
| Transformer, coordinate head | vehicle-like | 6 | **1.10** | **2.01** | **0.317** | **2.64** |
| Transformer, physics head | vehicle-like | 6 | 1.31 | 2.67 | 0.457 | 3.28 |

**Physical plausibility on val**, vehicle-like agents (23,113), share of trajectories:

| | Infeasible, 10 Hz | Infeasible, 2 Hz | Off-road |
| --- | --- | --- | --- |
| Ground truth | 2.93% | 0.41% | 0.19% |
| Coordinate head @1 / @6 | 57.8% / 70.0% | 0.00% / 0.01% | **1.52% / 1.75%** |
| Physics head @1 / @6 | **0.00% / 0.00%** | 0.00% / 0.00% | 5.17% / 5.68% |

The physics head meets the Phase 4 milestone. It trains stably, and none of its trajectories are
infeasible at either resolution. In the six dev mode plots its paths are smooth and follow the lane direction.

It does not yet pay for itself, though. On val it is behind the coordinate head on every accuracy
metric: brier-minFDE is 3.16 against 2.57, and K=1 minFDE is 6.95 against 6.09. It still beats the
LSTM at K=1 (2.74 / 6.95 against 3.15 / 8.33). The gap on dev grew from 0.24 m on the 10% subset to
0.63 m on the full subset (3.21 against 2.58).

The physics head underfits. Train ADE ends at 1.18 against 1.04, and dev brier-minFDE improves only
slowly after epoch 20 (3.30 at epoch 20, 3.21 at epoch 28). Since the head can reach vehicle-like
ground-truth futures to within 0.16 m, the limit looks like optimisation rather than what the head
can express. That is not yet tested.

The coordinate head's infeasibility is almost entirely step-to-step jitter. At 2 Hz its paths are
already feasible (0.00%, against 0.41% for the ground truth).

The physics head leaves the drivable area more than three times as often (5.2% against 1.5%), and the cause is
not yet known. Its mode scoring is slightly better: the classification loss ends at 1.59 against
1.64.

The full run takes 3.3 h on the M4 GPU, against 3.1 h for the coordinate head.

```bash
python src/physics.py --check --n 2000
python src/train.py --config configs/physics.yaml
python src/evaluate.py --checkpoint outputs/runs/physics/best.pt
python src/visualize.py --split train --random 6 --seed 0 --checkpoint outputs/runs/physics/best.pt
```
