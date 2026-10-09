# Physics-Constrained Transformer for Motion Forecasting

A compact Transformer for vehicle trajectory forecasting on [Argoverse 2](https://www.argoverse.org/av2.html),
with two interchangeable decoders. The **coordinate head** regresses future positions. The **physics
head** predicts acceleration and steering and integrates them through a differentiable kinematic
bicycle model, so every trajectory is drivable by construction.

The goal is not the most accurate forecaster. It is to build a physically accurate model and measure
how it performs against an otherwise identical unconstrained one.

[Technical report (PDF)](report/report.pdf) ·
[Pretrained weights](https://github.com/tali7965/Physics-Constrained-Transformer-for-Motion-Forecasting/releases/tag/v1.0) ·
[Development log](docs/development-log.md)

![Predicted trajectories of both heads on three held-out scenes](assets/examples.gif)

*Top: coordinate head. Bottom: physics head. Panels (b) and (c) show its failure modes. Overall, its
most probable trajectory stays on the road in about 95% of vehicle scenes.*

## Results

Argoverse 2 validation split (24,988 scenarios), K = 6 unless noted. Feasibility and off-road are
measured on the most probable trajectory of vehicle-like agents.

| Decoder | brier-minFDE | minFDE | MR | minFDE (K=1) | Infeasible (10 Hz / 2 Hz) | Off-road |
| --- | --- | --- | --- | --- | --- | --- |
| Coordinate head | **2.57** | **1.94** | **0.303** | **6.09** | 57.8% / 0.0% | **1.5%** |
| Physics head | 3.16 | 2.55 | 0.431 | 6.95 | **0.0% / 0.0%** | 5.2% |

The gap holds across ablations (brier-minFDE at each model's own K; minFDE for K = 1):

| Setting | Full data | Seed 1 | 25% data | 10% data | No map | K = 3 | K = 1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Coordinate head | **2.57** | **2.56** | **2.88** | **3.33** | **3.18** | **3.17** | **5.19** |
| Physics head | 3.16 | 3.18 | 3.61 | 3.98 | 3.58 | 3.72 | 5.81 |

### Findings

- The physics head eliminates infeasible trajectories but trails by 0.4–0.7 m in every setting, and
  less training data does not narrow the gap.
- The coordinate head's infeasibility is step-to-step jitter: at 2 Hz its trajectories are as
  feasible as the ground truth.
- Drivable is not road-aware. The bicycle model limits how the vehicle moves, not where it goes.
  Because position is the running sum of the controls, small heading or steering errors compound
  into drifts across the lane: an uncorrected initial heading, or an incomplete turn. That is why the
  physics head leaves the road more often.
- Integration also gives early controls about 60× the gradient of late ones, so late steering
  learns slowly.

## Installation

Requires Python 3.12 (`av2` ships macOS arm64 wheels for Python 3.9–3.12 only).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Scenarios are fetched from the public Argoverse S3 bucket with [s5cmd](https://github.com/peak/s5cmd)
and preprocessed into fixed-shape, agent-centric arrays. Set `AV2_DATA_ROOT`, or `data_root` in
`configs/preprocess.yaml`, to the storage location.

```bash
export AV2_DATA_ROOT=/path/to/av2
python src/download.py --split train      # 47 GB; also --split val (5.9 GB) and --split test (4.1 GB)
python src/preprocess.py --split train    # also --split val and --split test
```

The training subset is sampled from the full train split, so the whole split is required.

## Usage

### Pretrained weights

```bash
curl -L --create-dirs -o weights/physics.pt \
  https://github.com/tali7965/Physics-Constrained-Transformer-for-Motion-Forecasting/releases/download/v1.0/physics.pt
python src/evaluate.py --checkpoint weights/physics.pt
```

`transformer.pt` (the coordinate head) is in the same release.

### Training

```bash
python src/train.py --config configs/transformer.yaml   # coordinate head
python src/train.py --config configs/physics.yaml       # physics head
```

A full run takes about 3 h on an Apple M4. Ablation configs are in `configs/phase5/`.

### Evaluation and analysis

```bash
python src/evaluate.py --checkpoint outputs/runs/physics/best.pt        # validation metrics
python src/visualize.py --split val --random 6 --checkpoint outputs/runs/physics/best.pt
python src/diagnose.py --physics outputs/runs/physics/best.pt --coordinate outputs/runs/transformer/best.pt
python src/submit.py --checkpoint outputs/runs/physics/best.pt --split test   # submission file
```

## Repository structure

```text
configs/            experiment configs (YAML); phase5/ holds the ablations
src/
  download.py       dataset download from S3
  preprocess.py     raw scenarios -> agent-centric arrays
  models.py         LSTM and Transformer forecasters, coordinate and physics heads
  physics.py        differentiable kinematic bicycle model
  train.py          winner-takes-all training with dev-set model selection
  evaluate.py       Argoverse 2 metrics, feasibility and off-road rates
  diagnose.py       physics-head diagnostics
  figures.py        report figures
report/             technical report (LaTeX source and PDF)
docs/               development log
```

## Citation

```bibtex
@misc{taheri2026physics,
  author       = {Ali Taheri},
  title        = {Drivable by Construction, Behind on Accuracy: A Kinematic Decoder for
                  Transformer Motion Forecasting on Argoverse 2},
  year         = {2026},
  howpublished = {\url{https://github.com/tali7965/Physics-Constrained-Transformer-for-Motion-Forecasting}}
}
```

## License

The code is released under the [MIT License](LICENSE). Argoverse 2 data is © 2021 Argo AI, LLC,
licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The pretrained
weights and figures are derived from it and are for non-commercial use only.
