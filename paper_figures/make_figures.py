"""Figures for the paper draft. Run: py -3 paper/make_figures.py  (writes paper/fig1_pipeline.png, paper/fig2_gold_auc.png)
All numbers come from EXPERIMENTS.md (macro AUC on the 58 expert-labelled studies, out-of-fold)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, BLUE_LIGHT, ORANGE = "#2a78d6", "#dce9f9", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=300)
    ax.set_xlim(0, 128); ax.set_ylim(0, 48); ax.axis("off")
    W, H = 28.5, 18

    def box(x, y, title, body, fill="#ffffff", edge=INK2):
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fill, ec=edge, lw=1.0))
        ax.text(x + W / 2, y + H - 2.4, title, ha="center", va="top", fontsize=7.8, weight="bold")
        ax.text(x + W / 2, y + H - 7.0, "\n".join(body), ha="center", va="top", fontsize=6.1, color=INK2,
                linespacing=1.4)

    def arrow(x0, y0, x1, y1, color=INK2):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, shrinkA=0, shrinkB=0))

    xs = [1, 33, 65]
    box(xs[0], 28, "Radiology reports", ["4,407 studies, 9 languages", "no image-level labels"])
    box(xs[1], 28, "LLM report labelling", ["Qwen2.5-7B, Gemma 4 26B", "prompt = official criteria",
                                            "+ two public label sets"])
    box(xs[2], 28, "Soft training target", ["round 1: mean of label sets", "round k: 0.5 labels",
                                            "+ 0.5 out-of-fold teacher"])
    box(xs[0], 2, "MRI series", ["5 slots: sag/cor/ax fat-sat,", "sag non-fat-sat, cor T1",
                                 "100 mm crop, 224 px"])
    box(xs[1], 2, "2.5D reader", ["2D CNN per 3-slice window", "transformer over windows",
                                  "attention pooling per finding"])
    box(xs[2], 2, "Site-grouped 5 folds", ["+ one full-data model", "EMA, expert studies x3",
                                           "out-of-fold predictions"])
    box(99, 15, "Test time", ["dense windows", "(every slice a centre)", "rank average of families"],
        fill=BLUE_LIGHT, edge=BLUE)

    for y in (37, 11):
        arrow(xs[0] + W + 0.9, y, xs[1] - 0.9, y)
        arrow(xs[1] + W + 0.9, y, xs[2] - 0.9, y)
    arrow(xs[2] + 8, 27.1, xs[2] + 8, 20.9)                       # target -> training
    arrow(xs[2] + W + 0.9, 11, 98.1, 19)                          # training -> test
    arrow(xs[2] + 19, 20.9, xs[2] + 19, 27.1, color=ORANGE)       # OOF predictions -> next-round target
    ax.text(xs[2] + 6.8, 24, "train", ha="right", va="center", fontsize=6.2, color=INK2)
    ax.text(xs[2] + 20.2, 24, "next round", ha="left", va="center", fontsize=6.2, color=ORANGE)
    fig.savefig(os.path.join(OUT, "fig1_pipeline.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig2():
    targets = [("LLM report labels (ours)", 0.898), ("Round-1 target (3 label sets)", 0.900),
               ("Round-2 target (teacher M4-M6)", 0.932), ("Round-3 target (teacher M5-M7)", 0.937),
               ("Round-4 target (teacher M7 + 2xM9)", 0.942)]
    models = [("M2  ResNet-34, 130 mm", 0.801), ("M4  ConvNeXt-T reader, 130 mm", 0.890),
              ("M5  ConvNeXt-T reader, 100 mm", 0.907), ("M7  = M5 on round-2 target", 0.925),
              ("M9  EffNetV2-S, round-3 target", 0.928), ("M12  EffNetV2-S, round-4 target", 0.930),
              ("Ensemble M7 + M9 + 2xM12", 0.933)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=300, sharex=True,
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.95})
    for ax, rows, title in [(axes[0], targets, "Training targets"), (axes[1], models, "Image models (out-of-fold)")]:
        names = [r[0] for r in rows][::-1]
        vals = [r[1] for r in rows][::-1]
        y = range(len(rows))
        ax.hlines(y, 0.78, vals, color=GRID, lw=2, zorder=1)
        ax.scatter(vals, y, s=42, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
        ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xlim(0.78, 0.965)
        ax.set_title(title, fontsize=9, loc="left", color=INK, weight="bold")
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(axis="y", length=0)
    fig.supxlabel("macro AUC on the 58 expert-labelled studies (higher is better)", fontsize=7.5, color=INK2, y=-0.03)
    fig.savefig(os.path.join(OUT, "fig2_gold_auc.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


AQUA = "#1baf7a"


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)


def fig3():
    """(a) our own models: OOF / public / private across the development steps;
    (b) private score vs our share W in the blends, per public base, with the W the public LB preferred."""
    steps = ["M4", "M4\n+M5", "M4+M5\n+M6", "M7", "M7\n+2xM9", "M7+M9\n+2xM12"]
    oof = [0.890, 0.903, 0.908, 0.925, 0.932, 0.933]
    pub = [0.936, 0.941, 0.943, 0.946, 0.946, 0.946]
    prv = [0.924, 0.928, 0.929, 0.932, 0.932, 0.931]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=300, gridspec_kw={"wspace": 0.38})
    x = range(len(steps))
    for ys, c, name in [(pub, BLUE, "public LB"), (prv, ORANGE, "private LB"), (oof, AQUA, "OOF, 58 expert studies")]:
        a.plot(x, ys, color=c, lw=2, marker="o", ms=5, mec="white", mew=1.2, label=name, zorder=3)
    a.text(5.15, pub[-1], "public", va="center", fontsize=7, color=INK2)
    a.text(5.15, prv[-1] - 0.0018, "private", va="center", fontsize=7, color=INK2)
    a.text(5.15, oof[-1] + 0.0018, "OOF", va="center", fontsize=7, color=INK2)
    a.set_xticks(list(x)); a.set_xticklabels(steps, fontsize=6.8)
    a.set_xlim(-0.3, 5.9); a.set_ylim(0.885, 0.950)
    a.set_ylabel("macro AUC", fontsize=7.5)
    a.set_title("a  Our own models", loc="left", fontsize=9, weight="bold")
    a.legend(fontsize=6.5, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.02))
    _clean(a)

    bases = [("S6 base", [0.07, 0.20, 0.35], [0.938, 0.939, 0.939], 0.20, AQUA),
             ("R7 base", [0.20, 0.30, 0.40], [0.941, 0.942, 0.942], 0.20, ORANGE),
             ("0.958 base", [0.05, 0.10, 0.20, 0.30], [0.942, 0.942, 0.943, 0.943], 0.10, BLUE)]
    for name, ws, ps, chosen, c in bases:
        b.plot(ws, ps, color=c, lw=2, marker="o", ms=5, mec="white", mew=1.2, zorder=3)
        b.scatter([chosen], [ps[ws.index(chosen)]], s=120, facecolor="none", edgecolor=INK, lw=1.1, zorder=4)
        b.text(ws[-1] + 0.012, ps[-1], name, va="center", fontsize=7, color=INK2)
    b.set_xlim(0.0, 0.50); b.set_ylim(0.9365, 0.9445)
    b.set_xlabel("share of our models (W)", fontsize=7.5)
    b.set_ylabel("private LB (3 decimals)", fontsize=7.5)
    b.set_title("b  Blends: private score vs our share", loc="left", fontsize=9, weight="bold")
    b.text(0.005, 0.9369, "ring = W preferred by the public LB", fontsize=6.5, color=INK2)
    _clean(b)
    fig.savefig(os.path.join(OUT, "fig3_public_private.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig4():
    """Per-finding AUC on the 58 expert studies: report labels vs our image ensemble (dumbbell)."""
    rows = [("ACL", .982, .982), ("MCL", .982, .960), ("Medial meniscus", .941, .969),
            ("Lateral meniscus", .912, .866), ("Medial OA", .936, .991), ("Lateral OA", .838, .844),
            ("Patellofemoral OA", .879, .889), ("Effusion", .819, .967), ("Synovitis", .808, .830),
            ("Baker's cyst", .895, .975), ("Contusion", .862, .955), ("Fracture", .842, .965)]
    rows = sorted(rows, key=lambda r: r[2] - r[1])
    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=300)
    for i, (n, rep, img) in enumerate(rows):
        ax.plot([rep, img], [i, i], color=GRID, lw=2.2, zorder=1)
    ax.scatter([r[1] for r in rows], range(len(rows)), s=40, color=ORANGE, edgecolor="white", lw=1.2,
               zorder=3, label="report labels (LLM, criteria prompt)")
    ax.scatter([r[2] for r in rows], range(len(rows)), s=40, color=BLUE, edgecolor="white", lw=1.2,
               zorder=3, label="image ensemble (OOF)")
    for i, (n, rep, img) in enumerate(rows):
        d = img - rep
        ax.text(max(rep, img) + 0.006, i, f"{d:+.3f}", va="center", fontsize=6.8, color=INK2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlim(0.79, 1.02)
    ax.set_xlabel("AUC on the 58 expert-labelled studies", fontsize=7.5)
    ax.legend(fontsize=6.8, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.savefig(os.path.join(OUT, "fig4_per_finding.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("figures written to", OUT)
