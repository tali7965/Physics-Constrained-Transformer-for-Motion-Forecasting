"""README animation: the same three baseline-selected dev scenes, before and after the road-aware fix.

python src/animate.py             # render committed measured predictions, CPU only
python src/animate.py --refresh   # infer the same three scenes from weights, CPU only; needs AV2 data

The manifest fixes scene identity and records checkpoint hashes, probabilities and geometry.
Selection is inherited from the original baseline illustration, not repeated to favour the new head.
No training, GPU, test-set sampling or synthetic trajectories are involved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path as PolygonPath
import numpy as np
from config import PROJECT_ROOT, load_config
from figures import draw_map, INK, MUTED, DRIVABLE, LANE, PAGE
from preprocess import DT, FUTURE, to_agent_frame

OUT = PROJECT_ROOT / "assets/examples.gif"
MANIFEST = PROJECT_ROOT / "assets/examples-scenes.json"
NAMES = ("coordinate", "physics", "roadaware")
COLORS = {"coordinate": "#2a78d6", "physics": "#c98569", "roadaware": "#1b936d"}
LABELS = {"coordinate": "coordinate", "physics": "original physics", "roadaware": "road-aware physics"}
FPS, HOLD = 10, 20
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]


def refresh(manifest):
    import torch
    from av2.map.map_api import ArgoverseStaticMap
    from data import load_focal
    from evaluate import load_checkpoint, predict, rerank_setting, off_road
    torch.set_num_threads(4)
    device = torch.device("cpu")  # deliberately never auto-select a GPU
    rows = np.array([s["row"] for s in manifest["scenes"]])
    data = load_focal("train", rows, scene=True)
    cfg = load_config()
    pdir, rdir = cfg["processed_dir"] / "train", cfg["raw_dir"] / "train"
    sid = np.load(pdir / "scenario_id.npy", mmap_mode="r")
    origin = np.load(pdir / "origin.npy", mmap_mode="r")
    theta = np.load(pdir / "theta.npy", mmap_mode="r")
    predictions, probabilities, provenance = {}, {}, {}
    for name, filename in zip(NAMES, ("transformer.pt", "physics.pt", "physics-roadaware.pt")):
        path = PROJECT_ROOT / "weights" / filename
        model, checkpoint = load_checkpoint(path, device)
        setting = rerank_setting(checkpoint["config"])
        predictions[name], probabilities[name] = predict(model, data, device, rerank=setting)
        provenance[name] = {"file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "run": checkpoint["config"]["name"], "lane_rerank": setting}
        print(f"CPU predictions: {name}", flush=True)
        del model
    scenes = []
    for i, old in enumerate(manifest["scenes"]):
        row = old["row"]
        assert str(sid[row]) == old["scenario_id"], "Dataset row order differs from the recorded scenes"
        avm = ArgoverseStaticMap.from_json(rdir / str(sid[row]) / f"log_map_archive_{sid[row]}.json")
        areas = [to_agent_frame(a.xyz[:, :2], origin[row], float(theta[row])) for a in avm.get_scenario_vector_drivable_areas()]
        history, gt = data["history"][i, -30:, :2], data["target"][i, :, :2]
        pred = {n: predictions[n][i] for n in NAMES}
        prob = {n: probabilities[n][i] for n in NAMES}
        top = {n: int(prob[n].argmax()) for n in NAMES}
        first_off = {}
        for n in NAMES:
            outside = ~np.any([PolygonPath(a).contains_points(pred[n][top[n]]) for a in areas], axis=0)
            first_off[n] = int(outside.argmax()) if outside.any() else None
            # Independently verify the plotted flag in the original city-frame metric.
            flags = off_road("train", rows[i:i+1], pred[n][None])
            assert bool(flags[0,top[n]]) == bool(outside.any())
        points = np.vstack([history, gt] + [pred[n].reshape(-1,2) for n in NAMES])
        scene = {"row": row, "scenario_id": str(sid[row]), "title": old["title"],
                 "history": history.tolist(), "gt": gt.tolist(),
                 "pred": {n: pred[n].tolist() for n in NAMES},
                 "prob": {n: prob[n].tolist() for n in NAMES}, "top": top, "first_off": first_off,
                 "areas": [a.tolist() for a in areas],
                 "lanes": data["lanes"][i][data["lane_type"][i] >= 0].tolist(),
                 "mid": ((points.min(0)+points.max(0))/2).tolist(),
                 "half": max(15.0, float((points.max(0)-points.min(0)).max()/2+4)),
                 "top_fde": {n: float(np.linalg.norm(pred[n][top[n],-1]-gt[-1])) for n in NAMES}}
        scenes.append(scene)
        print(f"row {row}: top-mode FDE {scene['top_fde']}; first off-road step {first_off}", flush=True)
    manifest.update({"checkpoints": provenance, "device": "cpu", "scenes": scenes})
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def render(manifest):
    scenes = manifest["scenes"]
    fig, axes = plt.subplots(3, 3, figsize=(PAGE, 6.35), dpi=140, facecolor="white")
    lines, marks = [], []
    for j, raw in enumerate(scenes):
        scene = dict(raw, history=np.array(raw["history"]), gt=np.array(raw["gt"]),
                     areas=[np.array(a) for a in raw["areas"]], lanes=np.array(raw["lanes"]))
        here = scene["history"][-1]
        for k, name in enumerate(NAMES):
            ax = axes[k, j]
            draw_map(ax, scene)
            gt, = ax.plot([], [], color=INK, lw=1.6, zorder=3)
            lines.append((gt, np.vstack([here, scene["gt"]])))
            for m, trajectory in enumerate(raw["pred"][name]):
                is_top = m == raw["top"][name]
                line, = ax.plot([], [], color=COLORS[name], lw=1.8 if is_top else .65,
                                alpha=1 if is_top else .40, zorder=5 if is_top else 4)
                lines.append((line, np.vstack([here, trajectory])))
            step = raw["first_off"][name]
            if step is not None:
                mark, = ax.plot(*raw["pred"][name][raw["top"][name]][step], marker="x", ms=6,
                                mew=1.6, color=INK, zorder=6, visible=False)
                marks.append((mark, step))
            if j == 0:
                ax.set_ylabel(LABELS[name], fontsize=8)
        axes[0,j].set_title(raw["title"], loc="left", fontsize=8)
    handles = [Line2D([],[],color=MUTED,lw=1.6,label="history (3 s)"),
               Line2D([],[],color=INK,lw=1.6,label="ground truth"),
               Line2D([],[],color=INK,lw=1.8,label="top mode (thick)"),
               Line2D([],[],color=INK,lw=.65,alpha=.4,label="other 5 modes"),
               Line2D([],[],color=INK,lw=0,marker="x",ms=6,label="first off-road point"),
               Patch(facecolor=DRIVABLE,edgecolor=LANE,lw=.4,label="drivable area")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7, bbox_to_anchor=(.5,.005))
    clock = fig.text(.99,.995,"",ha="right",va="top",fontsize=8,color=INK)
    fig.tight_layout(rect=(0,.07,1,.98),h_pad=.65,w_pad=.45)
    def update(frame):
        step = min(frame, FUTURE-1)
        for line, path in lines: line.set_data(*path[:step+2].T)
        for mark, at in marks: mark.set_visible(step>=at)
        clock.set_text(f"t = +{(step+1)*DT:.1f} s")
        return [l for l,_ in lines]+[m for m,_ in marks]+[clock]
    OUT.parent.mkdir(parents=True,exist_ok=True)
    FuncAnimation(fig,update,frames=FUTURE+HOLD,blit=False).save(OUT,writer=PillowWriter(fps=FPS))
    # Disposable verification frame, not another public README artifact.
    update(FUTURE-1)
    review = PROJECT_ROOT / "outputs/figures/readme-animation-final.png"
    review.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(review,dpi=140,facecolor="white")
    plt.close(fig)
    print(f"Written {OUT.relative_to(PROJECT_ROOT)} ({OUT.stat().st_size/1e6:.1f} MB)",flush=True)


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--refresh",action="store_true")
    options = args.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if options.refresh: manifest = refresh(manifest)
    render(manifest)


if __name__ == "__main__":
    main()
