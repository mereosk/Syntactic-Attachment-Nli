"""
Figures for the PP-attachment human rating study.

Reads the output of analyze_human.py (results/human/items.csv and responses.csv) and writes,
into the same folder, as PNG and PDF:
  fig_item_lean      mean human lean per item (95% CI), sorted, one panel per category
  fig_answer_shares  diverging stacked bars of the 5-point answers per category
  fig_item_violin    distribution of item mean leans per category (violin, item dots, mean ± 95% CI)
  fig_human_summary  fig_answer_shares and fig_item_violin side by side as panels (a) and (b)

Usage (run prep_responses.py and analyze_human.py first):
  python scripts/plot_human.py [--dir results/human]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
CATS = ["Ins/Pos", "Mtx/Emb", "Ser/Mul"]
COLORS = {"Ins/Pos": "#4C72B0", "Mtx/Emb": "#DD8452", "Ser/Mul": "#55A868"}  # same as src/visualization.py
BALANCED_HALF_WIDTH = 0.5  # keep in sync with analyze_human.py

# diverging scale: High reading (warm) <- grey midpoint -> Low reading (cool)
LEVELS = [(-2, "Clearly High", "#c4472b"), (-1, "Somewhat High", "#f0a58c"),
          (0, "Both equally natural", "#d6d5d0"),
          (1, "Somewhat Low", "#8fb8ea"), (2, "Clearly Low", "#2a6bc4")]

TEXT_W, COL_W = 6.8, 3.3   # inches: full text width / single column of a two-column paper (ACL style)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_item_lean(items, out):
    cats = [c for c in CATS if c in set(items["category"])]
    widths = [max((items["category"] == c).sum(), 1) for c in cats]
    fig, axes = plt.subplots(1, len(cats), figsize=(TEXT_W, 2.6), sharey=True,
                             gridspec_kw={"width_ratios": widths})
    axes = np.atleast_1d(axes)
    for ax, cat in zip(axes, cats):
        g = items[items["category"] == cat].sort_values("mean_lean").reset_index(drop=True)
        x = np.arange(len(g))
        ax.axhspan(-BALANCED_HALF_WIDTH, BALANCED_HALF_WIDTH, color="#ecebe7", zorder=0)
        ax.axhline(0, color="#555", lw=0.8, zorder=1)
        ax.vlines(x, g["ci_low"], g["ci_high"], color=COLORS[cat], lw=0.8, alpha=0.6, zorder=2)
        sig = g["leans"] != "n.s."
        ax.scatter(x[sig], g["mean_lean"][sig], s=9, color=COLORS[cat], edgecolor="white", lw=0.4, zorder=3)
        ax.scatter(x[~sig], g["mean_lean"][~sig], s=9, facecolor="white", edgecolor=COLORS[cat], lw=0.8, zorder=3)
        n_hi, n_lo = int((g["leans"] == "High").sum()), int((g["leans"] == "Low").sum())
        ax.set_title(f"$\\bf{{{cat}}}$ ({len(g)} items)\nLow {n_lo} · High {n_hi} · n.s. {len(g) - n_lo - n_hi}",
                     fontsize=7.5)
        ax.set_xticks([])
        ax.set_xlim(-1, len(g))
        ax.set_xlabel("items, sorted")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(-2.15, 2.15)
    axes[0].set_yticks([-2, -1, 0, 1, 2])
    axes[0].set_yticklabels(["−2 High", "−1", "0", "+1", "+2 Low"])
    axes[0].set_ylabel("mean human lean (95% CI)")
    handles = [Line2D([], [], marker="o", ls="", color="#666", markeredgecolor="white", label="CI excludes 0"),
               Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#666", label="n.s."),
               Patch(color="#ecebe7", label=f"balanced (|lean| < {BALANCED_HALF_WIDTH})")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, out, "fig_item_lean")


def draw_answer_shares(ax, resp, cats, label_min=0.05, fontsize=7):
    """Diverging stacked bars of the 5-point answers; first category on top. Returns legend handles."""
    shares = (resp.groupby("category")["human_lean"].value_counts(normalize=True)
                  .unstack(fill_value=0).reindex(index=cats, columns=[v for v, _, _ in LEVELS], fill_value=0))
    y = np.arange(len(cats))[::-1]
    # start each bar so that the 'equal' segment is centred on 0
    left = start = -(shares[-2] + shares[-1] + shares[0] / 2).values
    handles = []
    for v, label, color in LEVELS:
        w = shares[v].values
        handles.append(ax.barh(y, w, left=left, height=0.6, color=color, edgecolor="white", lw=1, label=label))
        for yi, l, wi in zip(y, left, w):
            if wi >= label_min:
                ax.text(l + wi / 2, yi, f"{wi:.0%}", ha="center", va="center", fontsize=fontsize,
                        color="white" if abs(v) == 2 else "#222")
        left = left + w
    ax.axvline(0, color="#555", lw=0.8)
    lim = np.ceil(max(left.max(), -start.min()) * 10) / 10 + 0.05
    ax.set_xlim(-lim, lim)
    ticks = np.arange(-1, 1.01, 0.25)
    ticks = ticks[np.abs(ticks) <= lim]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(t):.0%}" for t in ticks])
    ax.set_xlabel("← High      share of ratings      Low →")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return handles


def draw_item_violin(ax, items, cats):
    """Horizontal violins of item mean leans with item dots and the category mean ± 95% CI."""
    y = np.arange(len(cats))[::-1]
    data = [items.loc[items["category"] == c, "mean_lean"].values for c in cats]
    ax.axvspan(-BALANCED_HALF_WIDTH, BALANCED_HALF_WIDTH, color="#ecebe7", zorder=0)
    ax.axvline(0, color="#555", lw=0.8, zorder=1)
    vp = ax.violinplot(data, positions=y, widths=0.8, showextrema=False, orientation="horizontal")
    for body, c in zip(vp["bodies"], cats):
        body.set_facecolor(COLORS[c]); body.set_edgecolor(COLORS[c]); body.set_alpha(0.25); body.set_zorder(2)
    rng = np.random.default_rng(0)
    for yi, d, c in zip(y, data, cats):
        ax.scatter(d, yi + rng.uniform(-0.12, 0.12, len(d)), s=7, color=COLORS[c],
                   edgecolor="white", lw=0.4, zorder=3)
        m, ci = d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        ax.errorbar(m, yi, xerr=ci, fmt="D", ms=4.5, color="#222", mfc="white", mew=1.2,
                    elinewidth=1.4, capsize=2.5, zorder=4)
    ax.set_xlim(-2.1, 2.1)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels(["−2", "−1", "0", "+1", "+2"])
    ax.set_xlabel("← High      item mean lean      Low →")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    return [Line2D([], [], marker="D", ls="", ms=4.5, color="#222", mfc="white", mew=1.2,
                   label="category mean ± 95% CI"),
            Patch(color="#ecebe7", label=f"balanced (|lean| < {BALANCED_HALF_WIDTH})")]


def cat_labels(cats, df, unit):
    n = df.groupby("category").size()
    return [f"{c}\n({n[c]} {unit})" for c in cats]


def plot_answer_shares(resp, out):
    cats = [c for c in CATS if c in set(resp["category"])]
    fig, ax = plt.subplots(figsize=(TEXT_W, 1.9))
    handles = draw_answer_shares(ax, resp, cats)
    ax.set_yticks(np.arange(len(cats))[::-1], cat_labels(cats, resp, "ratings"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout()
    save(fig, out, "fig_answer_shares")


def plot_item_violin(items, out):
    cats = [c for c in CATS if c in set(items["category"])]
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    handles = draw_item_violin(ax, items, cats)
    ax.set_yticks(np.arange(len(cats))[::-1], cat_labels(cats, items, "items"))
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.55, 0.98))
    fig.tight_layout()
    save(fig, out, "fig_item_violin")


def plot_summary(items, resp, out):
    """(a) answer shares and (b) item violins side by side, sharing the category rows."""
    cats = [c for c in CATS if c in set(items["category"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_W, 2.1), sharey=True, gridspec_kw={"width_ratios": [1.2, 1]})
    h1 = draw_answer_shares(a1, resp, cats, label_min=0.10, fontsize=6.5)
    h2 = draw_item_violin(a2, items, cats)
    a1.set_yticks(np.arange(len(cats))[::-1], cat_labels(cats, items, "items"))
    a1.set_title("(a) Distribution of ratings", loc="left", fontweight="bold")
    a2.set_title("(b) Distribution of item means", loc="left", fontweight="bold")
    fig.legend(handles=h1 + h2, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    save(fig, out, "fig_human_summary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "results" / "human",
                    help="folder with analyze_human.py output; figures are written here too")
    a = ap.parse_args()
    items = pd.read_csv(a.dir / "items.csv")
    resp = pd.read_csv(a.dir / "responses.csv")
    plot_item_lean(items, a.dir)
    plot_answer_shares(resp, a.dir)
    plot_item_violin(items, a.dir)
    plot_summary(items, resp, a.dir)
    print(f"wrote fig_item_lean, fig_answer_shares, fig_item_violin and fig_human_summary (.png, .pdf) to {a.dir}")


if __name__ == "__main__":
    main()
