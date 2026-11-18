"""CPU-only figures and tables for the road-aware extension.
Run: python report/roadaware_figures.py
Reads committed report/results/*.json. Does not load a model, dataset, torch, or GPU.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = {p.stem: json.loads(p.read_text()) for p in (ROOT / "results").glob("*.json")}
GROUPS = [("Original", "physics", "raw"), ("Original + ranking", "physics", "rerank"),
          ("Road-aware, raw", "roadaware", "raw"), ("Road-aware + ranking", "roadaware", "rerank")]
KEYS = ["brier-minFDE@6", "minFDE@6", "MR@6", "minFDE@1", "offroad@1", "offroad@6"]

def values(prefix, tag, key):
    return np.array([DATA[f"{prefix}-seed{s}"]["val"][tag][key] for s in range(3)])

def fmt(v, scale=1, decimals=3):
    return f"${scale*v.mean():.{decimals}f} \\pm {scale*v.std(ddof=1):.{decimals}f}$"

lines = []
for label, prefix, tag in GROUPS:
    cells = [fmt(values(prefix, tag, k), 100 if k.startswith("offroad") else 1,
                 2 if k.startswith("offroad") else 3) for k in KEYS]
    lines.append(label + " & " + " & ".join(cells) + r" \\")
(ROOT / "roadaware_means.tex").write_text("\n".join(lines) + "\n" + r"\bottomrule" + "\n")
lines = []
for s in range(3):
    b=DATA[f"physics-seed{s}"]["val"]["raw"]
    r=DATA[f"roadaware-seed{s}"]["val"]["rerank"]
    cells=[str(s), f"{b[KEYS[0]]:.3f} / {r[KEYS[0]]:.3f}",
           f"{100*b['offroad@1']:.2f} / {100*r['offroad@1']:.2f}",
           f"{100*b['offroad@6']:.2f} / {100*r['offroad@6']:.2f}"]
    lines.append(" & ".join(cells) + r" \\")
(ROOT / "roadaware_seeds.tex").write_text("\n".join(lines) + "\n" + r"\bottomrule" + "\n")

from presentation_figures import build_figures
build_figures()
