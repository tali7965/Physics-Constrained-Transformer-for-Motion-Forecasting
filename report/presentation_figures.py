"""Report figures from committed measurements only. No model, dataset access, or GPU required."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
INK, MUTED, RULE = "#25262a", "#66696e", "#c9cbcd"
BLUE, CLAY, GREEN = "#2874b7", "#ba775c", "#21866b"
plt.rcParams.update({"font.family":"serif", "font.serif":["Liberation Serif","DejaVu Serif"],
    "mathtext.fontset":"stix", "font.size":8.5, "axes.labelsize":8.5, "axes.titlesize":10,
    "xtick.labelsize":8, "ytick.labelsize":8, "text.color":INK, "axes.labelcolor":INK,
    "xtick.color":MUTED, "ytick.color":INK, "pdf.fonttype":42,
    "axes.spines.top":False, "axes.spines.right":False, "axes.spines.left":False,
    "axes.edgecolor":RULE, "axes.linewidth":.6})


def quantitative():
    records={p.stem:json.loads(p.read_text()) for p in (ROOT/"results").glob("*.json")}
    groups=[("Original", "physics", "raw",CLAY),
            ("Original + ranking", "physics", "rerank",CLAY),
            ("Road-aware", "roadaware", "raw",GREEN),
            ("Road-aware + ranking", "roadaware", "rerank",GREEN)]
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.15),sharey=True)
    for ax,key,title in zip(axes,["offroad@1","offroad@6"],["(a) Most probable mode", "(b) All six modes"]):
        for j,(label,prefix,tag,color) in enumerate(groups):
            values=100*np.array([records[f"{prefix}-seed{s}"]["val"][tag][key] for s in range(3)])
            y=3-j
            ax.hlines(y,0,6.5,color="#ececed",lw=.6,zorder=0)
            ax.errorbar(values.mean(),y,xerr=values.std(ddof=1),fmt="D",ms=4.8,color=color,
                        elinewidth=1.3,capsize=3,zorder=4)
            ax.scatter(values,y+np.array([-.14,0,.14]),s=16,facecolors="white",edgecolors=color,lw=.9,zorder=3)
            ax.text(6.78,y,f"{values.mean():.2f}%",ha="right",va="center",fontsize=8.5,color=INK)
        ax.set_title(title,loc="left",pad=12)
        ax.set_xlim(0,6.95);ax.set_xticks([0,2,4,6]);ax.set_ylim(-.45,3.45)
        ax.set_yticks(range(4),[g[0] for g in groups][::-1])
        ax.tick_params(axis="y",length=0,pad=7)
        ax.set_xlabel("Off-road trajectories (%)",labelpad=6)
    axes[1].tick_params(axis="y",labelleft=False)
    fig.subplots_adjust(left=.195,right=.99,bottom=.22,top=.80,wspace=.16)
    fig.savefig(OUT/"roadaware.pdf",bbox_inches="tight",pad_inches=.03)
    plt.close(fig)


def qualitative():
    data=json.loads((ROOT.parent/"assets/examples-scenes.json").read_text())
    names=["coordinate","physics","roadaware"]
    labels=["Coordinate","Original kinematic","Road-aware kinematic"]
    colors=[BLUE,CLAY,GREEN]
    titles=["(a) Turn: intent error remains", "(b) Turn: understeer case", "(c) Straight: heading-error case"]
    fig,axes=plt.subplots(3,3,figsize=(7.16,5.6))
    for j,scene in enumerate(data["scenes"]):
        mid,half=np.array(scene["mid"]),scene["half"]
        for k,name in enumerate(names):
            ax=axes[k,j]
            for a in scene["areas"]:
                ax.add_patch(Polygon(a,closed=True,facecolor="#f0f0ec",edgecolor="#c7c8c1",lw=.4,zorder=0))
            for lane in scene["lanes"]:
                a=np.array(lane);ax.plot(*a.T,color="#c3c5bf",lw=.45,zorder=1)
            history=np.array(scene["history"]);gt=np.array(scene["gt"])
            ax.plot(*history.T,color="#93958f",lw=1.4,zorder=2)
            ax.plot(*gt.T,color=INK,lw=1.3,zorder=5)
            top=scene["top"][name]
            for m,trajectory in enumerate(scene["pred"][name]):
                a=np.vstack([history[-1],trajectory]);is_top=m==top
                ax.plot(*a.T,color=colors[k],lw=1.65 if is_top else .6,
                        alpha=1 if is_top else .35,zorder=6 if is_top else 3)
            step=scene["first_off"][name]
            if step is not None:
                ax.plot(*scene["pred"][name][top][step],marker="x",ms=5.5,mew=1.25,color=INK,zorder=7)
            ax.set_xlim(mid[0]-1.2*half,mid[0]+1.2*half)
            ax.set_ylim(mid[1]-half,mid[1]+half);ax.set_aspect("equal")
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(True);spine.set_color("#d2d3d0");spine.set_linewidth(.5)
            x0,y0=mid[0]-1.2*half+.10*half,mid[1]-half+.13*half
            ax.plot([x0,x0+10],[y0,y0],color=INK,lw=1,zorder=8)
            ax.text(x0+5,y0+.05*half,"10 m",ha="center",fontsize=6.8,zorder=8)
            ax.text(.97,.96,f"FDE {scene['top_fde'][name]:.1f} m",transform=ax.transAxes,
                    ha="right",va="top",fontsize=7.6,color=INK,
                    bbox={"facecolor":"white","edgecolor":"none","alpha":.9,"pad":1.7},zorder=9)
            if j==0:ax.set_ylabel(labels[k],fontsize=8.5,labelpad=7,color=colors[k])
            if k==0:ax.set_title(titles[j],fontsize=8.4,loc="left",pad=7)
    handles=[Line2D([],[],color="#93958f",lw=1.4,label="History (3 s)"),
             Line2D([],[],color=INK,lw=1.3,label="Ground truth"),
             Line2D([],[],color=MUTED,lw=1.65,label="Top mode (thick)"),
             Line2D([],[],color=MUTED,lw=.6,alpha=.4,label="Other five modes"),
             Line2D([],[],color=INK,lw=0,marker="x",ms=5,label="First off-road point"),
             Patch(facecolor="#f0f0ec",edgecolor="#c7c8c1",lw=.5,label="Drivable area")]
    fig.legend(handles=handles,loc="lower center",ncol=3,frameon=False,fontsize=8,
               bbox_to_anchor=(.52,.005),columnspacing=2.1,handlelength=2)
    fig.subplots_adjust(left=.07,right=.995,bottom=.10,top=.95,hspace=.14,wspace=.08)
    fig.savefig(OUT/"roadaware_examples.pdf",bbox_inches="tight",pad_inches=.03)
    plt.close(fig)


def build_figures():
    OUT.mkdir(exist_ok=True)
    quantitative();qualitative()
    print("Written figures/roadaware.pdf and figures/roadaware_examples.pdf (committed measurements, CPU only).")

if __name__ == "__main__":
    build_figures()
