"""Animated version of the report's example figure, for the README: assets/examples.gif.

Usage:
    python src/animate.py

The three dev scenes of report/figures/examples.pdf (the same selection, figures.example_scenes),
with the coordinate head above the kinematic (physics) head. The ground truth and each head's six
modes unroll over the 6 s future in real time, then the last frame holds for 2 s.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from config import PROJECT_ROOT
from figures import COLOR, INK, LABEL, PAGE, draw_map, example_scenes, legend_handles, load
from models import pick_device
from preprocess import DT, FUTURE

OUT = PROJECT_ROOT / "assets" / "examples.gif"
FPS, HOLD = 10, 20  # one frame per 0.1 s step (GIF timing has 10 ms steps); extra frames on the last step


def main():
    device = pick_device()
    models, dev, rows = load(device)
    scenes = example_scenes(models, dev, rows, device)

    fig, axes = plt.subplots(2, 3, figsize=(PAGE, 3.9), dpi=140, facecolor="white")
    lines = []  # (artist, path from the current position, its first FUTURE + 1 points revealed over time)
    marks = []  # (artist, step at which it appears)
    for j, scene in enumerate(scenes):
        here = scene["history"][-1]
        for k, name in enumerate(("coordinate", "physics")):
            ax = axes[k, j]
            draw_map(ax, scene)
            gt, = ax.plot([], [], color=INK, lw=1.6, zorder=3)
            lines.append((gt, np.vstack([here, scene["gt"]])))
            for m, traj in enumerate(scene["pred"][name]):
                is_top = m == scene["top"][name]
                line, = ax.plot([], [], color=COLOR[name], lw=1.6 if is_top else 0.7, alpha=1.0 if is_top else 0.6, zorder=4)
                lines.append((line, np.vstack([here, traj])))
            step = scene["first_off"][name]
            if step is not None:
                mark, = ax.plot(*scene["pred"][name][scene["top"][name], step], marker="x", ms=6, mew=1.6, color=INK,
                                zorder=6, visible=False)
                marks.append((mark, step))
            if j == 0:
                ax.set_ylabel(LABEL[name])
        axes[0, j].set_title(scene["title"], loc="left")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    clock = fig.text(0.995, 0.995, "", ha="right", va="top", fontsize=8, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.4, w_pad=0.4)

    def update(frame):
        step = min(frame, FUTURE - 1)  # future step index, 0..59
        for line, path in lines:
            line.set_data(*path[: step + 2].T)
        for mark, at in marks:
            mark.set_visible(step >= at)
        clock.set_text(f"t = +{(step + 1) * DT:.1f} s")
        return [line for line, _ in lines] + [mark for mark, _ in marks] + [clock]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, update, frames=FUTURE + HOLD, blit=False).save(OUT, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"written {OUT.relative_to(PROJECT_ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
