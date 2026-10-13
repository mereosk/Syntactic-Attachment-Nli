"""Table 1, appendix tables (controls, bias, Kendall), the P(E) vs P(C) scatter, and copies of the
model_analysis.py appendix figures (JSD, clustermaps)  ->  results/paper/
  python scripts/paper_tables.py      (run model_analysis.py first; run paper_tables_extra.py AFTER this, it overwrites tabA_bias.tex)
"""
import sys, shutil
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
from matplotlib.lines import Line2D
from model_analysis import load_models, item_table
matplotlib.rcdefaults()   # undo the human-figure style that model_analysis.py imports
ORDER = ["DeBERTa-v3-base", "DeBERTa-v3-large", "RoBERTa-large", "BART-large", "Flan-T5-large",
         "Qwen2.5-1.5B", "Llama-3.2-3B", "Mistral-7B", "Qwen2.5-7B", "Llama-3-8B",
         "Llama-3.3-70B", "DeepSeek-V4-Pro", "GPT-4.1"]
EXCL = {"Llama-3.2-3B", "Mistral-7B"}
CATS = ["Ins/Pos", "Mtx/Emb", "Ser/Mul"]
name = lambda m: m + (r"$^\dagger$" if m in EXCL else "")

# ---------------- scatter P(E) vs P(C) -----------------------------------------
s = pd.read_csv(RESULTS)
a = s[s.slot.isin(["amb_high", "amb_low"])]
d = a.groupby(["model", "item_id", "slot"])[["p_entailment", "p_neutral", "p_contradiction"]].mean().reset_index()
plt.rcParams.update({"font.size": 7, "axes.edgecolor": "#999"})
fig, axs = plt.subplots(3, 5, figsize=(7.0, 4.6), sharex=True, sharey=True)
col = {"amb_high": "#2a78d6", "amb_low": "#eb6834"}
for ax, mo in zip(axs.flat, ORDER):
    g = d[d.model == mo]
    ax.plot([0, 1], [1, 0], color="#aaa", lw=0.6, ls="--")
    for sl in ["amb_high", "amb_low"]:
        h = g[g.slot == sl]
        ax.scatter(h.p_entailment, h.p_contradiction, s=5, c=col[sl], alpha=0.55, linewidths=0)
    ax.set_title(mo + (" †" if mo in EXCL else ""), fontsize=7)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    ax.grid(color="#eee", lw=0.5); ax.set_axisbelow(True)
for ax in axs.flat[len(ORDER):]: ax.axis("off")
for ax in axs[-1]: ax.set_xlabel("P(E)")
for ax in axs[:, 0]: ax.set_ylabel("P(C)")
axs.flat[-1].legend(handles=[Line2D([], [], marker="o", ls="", color=col["amb_high"], label="High reading"),
                             Line2D([], [], marker="o", ls="", color=col["amb_low"], label="Low reading"),
                             Line2D([], [], color="#aaa", ls="--", label="P(N) = 0")], loc="center", frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_scatter_EC.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_scatter_EC.png", dpi=300, bbox_inches="tight")

# ---------------- copy appendix figures from the analysis run --------------------
for f, g in [("fig_step4_jsd.png", "figA_jsd.png"), ("step5_human_human.png", "figA_cluster_humans.png"),
             ("step5_models_humans.png", "figA_cluster_models_humans.png")]:
    shutil.copy(f"{R}/{f}", f"{OUT}/{g}")

# ---------------- Table 1 (main) -------------------------------------------------
c1 = pd.read_csv(f"{R}/step1_controls.csv"); c2 = pd.read_csv(f"{R}/step2_contradiction.csv")
k = pd.read_csv(f"{R}/step5_model_vs_human.csv"); k = k[k.category != "all"]
rows = []
for mo in ORDER:
    x1, x2, xk = c1[c1.model == mo], c2[c2.model == mo], k[k.model == mo]
    w = x2.n_items
    fails = [c.split("/")[0][:3] for c in CATS if not x1[x1.category == c].passes.iloc[0]]
    rows.append(dict(
        Model=name(mo),
        ctrl=f"{x1[['acc_H', 'acc_L']].min(axis=1).min():.2f}",
        fails=",".join(fails) if fails else "--",
        pc_ent=f"{(x2['meanC_ctrl_ent'] * w).sum() / w.sum():.2f}",
        pc_amb=f"{(x2['meanC_amb'] * w).sum() / w.sum():.2f}",
        neither=f"{int(x2.neither_entailed.sum())}",
        loserC=f"{(x2['loser_more_C_than_N'] * w).sum() / w.sum():.2f}",
        tau_avg=f"{(xk.tau_avg_human * xk.n_items).sum() / xk.n_items.sum():.2f}",
        tau_best=f"{(xk.tau_best_individual * xk.n_items).sum() / xk.n_items.sum():.2f}"))
t1 = pd.DataFrame(rows)
ceil = k.drop_duplicates("category"); ceil_v = (ceil.human_ceiling_loo * ceil.n_items).sum() / ceil.n_items.sum()
lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lcccccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{Controls} & \multicolumn{4}{c}{Contradiction / ambiguity} & \multicolumn{2}{c}{Kendall $\tau_b$} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}\cmidrule(lr){8-9}",
         r"Model & min acc & fails & $\bar P(C)$ ent & $\bar P(C)$ amb & neither & loser C & avg & best \\", r"\midrule"]
for i, r in t1.iterrows():
    if i == 4: lines.append(r"\midrule")
    lines.append(" & ".join(r.values) + r" \\")
lines += [r"\midrule", rf"Human ceiling & & & & & & & {ceil_v:.2f} & \\", r"\bottomrule", r"\end{tabular}}",
          r"\caption{Per-model summary. \emph{min acc}: lowest control accuracy over categories and High/Low controls; "
          r"\emph{fails}: categories with accuracy $<0.90$. $\bar P(C)$ on entailed controls and on ambiguous pairs. "
          r"\emph{neither}: ambiguous items where both readings get $P(E)<\max(P(N),P(C))$ (of 130). "
          r"\emph{loser C}: share of items where the dispreferred reading gets more contradiction than neutral. "
          r"Kendall $\tau_b$ between model bias and human lean, computed within category and averaged, against the mean human "
          r"(\emph{avg}) and the best-matching participant (\emph{best}). Human ceiling: each participant vs.\ the mean of the others. "
          r"$^\dagger$Excluded from the main analysis (high $P(C)$ on entailed controls).}",
          r"\label{tab:main}", r"\end{table}"]
open(f"{OUT}/tab1_main.tex", "w").write("\n".join(lines))

# ---------------- Appendix tables -----------------------------------------------
def latex_table(df, cols, header, caption, label, star=False):
    env = "table*" if star else "table"
    L = [rf"\begin{{{env}}}[t]", r"\centering", r"\small", r"\resizebox{\textwidth}{!}{%" if star else r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule", header + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        L.append(" & ".join(str(r[c]) for c in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\end{{{env}}}"]
    return "\n".join(L)

# controls per category
w = c1.pivot_table(index="model", columns="category", values=["acc_H", "acc_L"]).reindex(ORDER)
df = pd.DataFrame({"Model": [name(m) for m in ORDER]})
for c in CATS:
    for h in ["acc_H", "acc_L"]:
        df[f"{c}{h}"] = [f"{v:.2f}" if v >= 0.9 else rf"\textbf{{{v:.2f}}}" for v in w[(h, c)]]
hdr = r"Model & \multicolumn{2}{c}{Ins/Pos} & \multicolumn{2}{c}{Mtx/Emb} & \multicolumn{2}{c}{Ser/Mul} \\ & H & L & H & L & H & L"
open(f"{OUT}/tabA_controls.tex", "w").write(latex_table(df, df.columns, hdr,
    r"Control accuracy per category for High (H) and Low (L) controls; bold: $<0.90$. Accuracy = share of items where the entailed hypothesis gets the higher $P(E)$.", "tab:controls"))

# bias tests (main subset = n.s.)
b = pd.read_csv(f"{R}/step3_all_vs_ns.csv")
fmt = lambda med, p: f"{med:+.2f}" + ("$^{*}$" if p < .05 else "") if pd.notna(p) else f"{med:+.2f}"
df = pd.DataFrame({"Model": [name(m) for m in ORDER]})
for c in CATS:
    g = b[b.category == c].set_index("model").reindex(ORDER)
    df[c + "all"] = [fmt(m, p) for m, p in zip(g.median_all, g.p_all)]
    df[c + "amb"] = [fmt(m, p) for m, p in zip(g.median_sub, g.p_sub)]
n_amb = b.drop_duplicates("category").set_index("category").n_sub
hdr = (r"Model & \multicolumn{2}{c}{Ins/Pos} & \multicolumn{2}{c}{Mtx/Emb} & \multicolumn{2}{c}{Ser/Mul} \\ & all & amb & all & amb & all & amb")
open(f"{OUT}/tabA_bias.tex", "w").write(latex_table(df, df.columns, hdr,
    "Median attachment bias $P(E|H^l)-P(E|H^h)$ on all items and on human-ambiguous items "
    f"(Ins/Pos {n_amb['Ins/Pos']}, Mtx/Emb {n_amb['Mtx/Emb']}, Ser/Mul {n_amb['Ser/Mul']} items). "
    r"$^{*}$Wilcoxon signed-rank $p<.05$ against 0 (uncorrected).", "tab:bias"))

# kendall per category
df = pd.DataFrame({"Model": [name(m) for m in ORDER]})
for c in CATS:
    g = k[k.category == c].set_index("model").reindex(ORDER)
    df[c + "a"] = [f"{v:.2f}" for v in g.tau_avg_human]; df[c + "b"] = [f"{v:.2f}" for v in g.tau_best_individual]
ce = ceil.set_index("category")
hdr = (r"Model & \multicolumn{2}{c}{Ins/Pos} & \multicolumn{2}{c}{Mtx/Emb} & \multicolumn{2}{c}{Ser/Mul} \\ & avg & best & avg & best & avg & best")
tab = latex_table(df, df.columns, hdr,
    r"Kendall $\tau_b$ between per-item model bias and human lean, against the mean human (avg) and the best-matching participant (best). "
    "Human ceiling (participant vs. mean of others): " + ", ".join(f"{c} {ce.loc[c, 'human_ceiling_loo']:.2f}" for c in CATS) +
    "; mean pairwise human $\\tau$: " + ", ".join(f"{c} {ce.loc[c, 'human_mean_pairwise']:.2f}" for c in CATS) + ".", "tab:kendall")
open(f"{OUT}/tabA_kendall.tex", "w").write(tab)
t1.to_csv(f"{OUT}/tab1_main.csv", index=False)
print(t1.to_string(index=False))
