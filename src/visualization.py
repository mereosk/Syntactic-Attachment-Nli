"""Visualization functions for analysis results."""

import numpy as np
import matplotlib.pyplot as plt

from .config import RESULTS_DIR, MODEL_ORDER, FAMILY, PROMPT_FORMAT


CATS = ["Ins/Pos", "Mtx/Emb", "Ser/Mul"]
COLORS = {"Ins/Pos": "#4C72B0", "Mtx/Emb": "#DD8452", "Ser/Mul": "#55A868"}
PANELS = [("SRB_rate", "(A) Signed recency bias"),
          ("JS3", "(B) JS distance (E/N/C)"),
          ("DA", "(C) Disambiguation asymmetry"),
          ("neutral_amb", "(D) P(neutral) on ambiguous premise")]


def plot_results(summary):
    models_present = [m for m in MODEL_ORDER if m in set(summary["model"].astype(str))]
    fig, axes = plt.subplots(1, 4, figsize=(26, 5))
    x = np.arange(len(models_present))
    for ax, (metric, title) in zip(axes, PANELS):
        for i, cat in enumerate(CATS):
            sub = summary[summary["category"] == cat].set_index("model")[metric].reindex(models_present)
            ax.bar(x + (i - 1) * 0.27, sub.values, width=0.27, label=cat, color=COLORS[cat])
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(sum(FAMILY[m] == "classifier" for m in models_present) - 0.5, color="grey", ls="--", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models_present, rotation=45, ha="right", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend()
    plt.suptitle(f"prompt format: {PROMPT_FORMAT}", y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"results_{PROMPT_FORMAT}.png", dpi=300, bbox_inches="tight")
    plt.show()
