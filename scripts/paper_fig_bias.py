"""Figure 2: attachment bias per item, all items vs human-ambiguous items.
  python scripts/paper_fig_bias.py [ns]        -> results/paper/fig2_bias           (95% CI of human lean includes 0)
  python scripts/paper_fig_bias.py balanced    -> results/paper/figA_bias_balanced  (|mean lean| < 0.5)
Run model_analysis.py first (reads results/analysis/step1_controls.csv).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from src.config import RESULTS
from human_analysis import HUMAN_DIR
R, OUT = ROOT / "results" / "analysis", ROOT / "results" / "paper"   # model_analysis.py output -> paper assets
OUT.mkdir(parents=True, exist_ok=True)
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from model_analysis import load_models, item_table, load_humans
matplotlib.rcdefaults()   # undo the human-figure style that model_analysis.py imports
MODE = sys.argv[1] if len(sys.argv) > 1 else "ns"
NAME = {"ns": "fig2_bias", "balanced": "figA_bias_balanced"}[MODE]
_, m, _ = load_models(RESULTS); t = item_table(m)
H, black, *_ = load_humans(HUMAN_DIR / "key.csv", HUMAN_DIR / "v1.csv", HUMAN_DIR / "v2.csv")
K = H[[p for p in H.columns if p not in black]]
lean = K.mean(axis=1); se = K.std(axis=1) / np.sqrt(K.notna().sum(axis=1))
amb = set(lean.index[(lean - 1.96 * se <= 0) & (lean + 1.96 * se >= 0)]) if MODE == "ns" else set(lean.index[lean.abs() < 0.5])
cat = t.drop_duplicates("item_id").set_index("item_id")["category"]
hum = (lean / 2).groupby(cat.reindex(lean.index)).mean()       # human mean lean rescaled to [-1, 1]

order = ["DeBERTa-v3-base", "DeBERTa-v3-large", "RoBERTa-large", "BART-large",
         "Flan-T5-large", "Qwen2.5-1.5B", "Qwen2.5-7B",
         "Llama-3-8B", "Llama-3.3-70B", "DeepSeek-V4-Pro", "GPT-4.1"]
CATS = ["Ins/Pos", "Mtx/Emb", "Ser/Mul"]
C_ALL, C_AMB = "#b8b8b8", "#2a78d6"
ctrl = pd.read_csv(R / "step1_controls.csv")
fails = set(zip(ctrl.loc[~ctrl.passes, "model"], ctrl.loc[~ctrl.passes, "category"]))
plt.rcParams.update({"hatch.linewidth": 0.5, "hatch.color": "#333", "font.size": 8, "axes.edgecolor": "#888", "xtick.color": "#444", "ytick.color": "#222"})
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=True)
y = np.arange(len(order))
for ax, c in zip(axs, CATS):
    for k, (sub, col, off) in enumerate([(None, C_ALL, -0.19), (amb, C_AMB, 0.19)]):
        data = []
        for mo in order:
            g = t[(t.model == mo) & (t.category == c)]
            if sub is not None: g = g[g.item_id.isin(sub)]
            data.append(g.bias.to_numpy())
        bp = ax.boxplot(data, positions=y + off, orientation="horizontal", widths=0.32, patch_artist=True,
                        showfliers=False, whis=(5, 95),
                        medianprops=dict(color="#111", lw=1.3), whiskerprops=dict(color="#666", lw=0.7),
                        capprops=dict(color="#666", lw=0.7))
        for p, mo in zip(bp["boxes"], order):
            p.set_facecolor(col); p.set_edgecolor("#555"); p.set_linewidth(0.6)
            if (mo, c) in fails:
                p.set_hatch("/////"); p.set_edgecolor("#333")
    ax.axvline(0, color="#333", lw=0.7)
    ax.axvline(hum[c], color="#d03b3b", lw=1.1, ls="--")
    n_amb = len([i for i in amb if cat.get(i) == c]); n_all = int((cat == c).sum())
    ax.set_title(f"{c}  (n = {n_all} / {n_amb})", fontsize=9)
    ax.set_xlim(-1.05, 1.05); ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.grid(axis="x", color="#eee", lw=0.6); ax.set_axisbelow(True)
    ax.axhline(3.5, color="#bbb", lw=0.6, ls=":")
axs[0].set_yticks(y); axs[0].set_yticklabels(order); axs[0].invert_yaxis()
axs[1].set_xlabel("← prefers High        P(E | Low reading) − P(E | High reading)        prefers Low →")
fig.legend(handles=[Patch(facecolor=C_ALL, edgecolor="#555", label="all items"),
                    Patch(facecolor=C_AMB, edgecolor="#555", label=("humans leave open (95% CI ∋ 0)" if MODE == "ns" else "humans find balanced (|lean| < 0.5)")),
                    Patch(facecolor="white", edgecolor="#333", hatch="/////", label="controls < 0.90"),
                    Line2D([], [], color="#d03b3b", ls="--", label="human mean lean (÷2)")],
           loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.995), handlelength=1.6, columnspacing=1.2)
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(OUT / f"{NAME}.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / f"{NAME}.pdf", bbox_inches="tight")
print(hum.round(2).to_dict())
