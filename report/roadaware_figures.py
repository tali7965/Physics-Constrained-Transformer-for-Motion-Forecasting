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

plt.rcParams.update({"font.family":"serif", "font.serif":["Liberation Serif","DejaVu Serif"],
    "font.size":8, "axes.labelsize":8, "xtick.labelsize":7, "ytick.labelsize":7,
    "pdf.fonttype":42, "axes.spines.top":False,"axes.spines.right":False})
fig, axes = plt.subplots(1,2,figsize=(3.5,2.15),sharey=True)
colors=["#5c5c62","#8d8d92","#edab8d","#d65d2e"]
labels=["Original","+ rank","Trained","+ rank"]
for ax,key,title in zip(axes,["offroad@1","offroad@6"],["Most probable mode","All six modes"]):
    for j,(_,prefix,tag) in enumerate(GROUPS):
        y=100*values(prefix,tag,key)
        ax.bar(j,y.mean(),color=colors[j],width=.64,alpha=.88,zorder=2)
        ax.scatter(j+np.array([-.12,0,.12]), y, s=10,facecolor="white",edgecolor="#26262b",linewidth=.65,zorder=3)
    ax.set_xticks(range(4), labels, rotation=45, ha="right")
    ax.set_title(title,fontsize=8)
    ax.set_ylim(0,6.6); ax.set_yticks([0,2,4,6])
    ax.yaxis.grid(True,color="#dedee1",linewidth=.5,zorder=0)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Off-road trajectories (%)")
fig.tight_layout(w_pad=.7,pad=.5)
(ROOT/"figures").mkdir(exist_ok=True)
fig.savefig(ROOT/"figures/roadaware.pdf",bbox_inches="tight",pad_inches=.02)
plt.close(fig)
print("Generated roadaware_means.tex, roadaware_seeds.tex, figures/roadaware.pdf from stored seed results.")
