"""Appendix tables backing the Results claims -> results/paper/:
  tabA_bias.tex  median b_x on all / human-ambiguous / other items (+ Wilcoxon, Mann-Whitney)
  tabA_rq2.tex   RQ2 per category: both entailed (%), neither (n), loser rejected (%), mean P(N)
  tabA_human.tex  human lean per category (participant-level Wilcoxon), H / L / n.s. item counts
Usage: python scripts/paper_tables_extra.py   (after model_analysis.py and paper_tables.py) -> results/paper/
"""
import sys
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from src.config import RESULTS
from human_analysis import HUMAN_DIR
R, OUT = ROOT / "results" / "analysis", ROOT / "results" / "paper"   # model_analysis.py output -> paper assets
OUT.mkdir(parents=True, exist_ok=True)

ORDER = ['DeBERTa-v3-base', 'DeBERTa-v3-large', 'RoBERTa-large', 'BART-large', 'Flan-T5-large', 'Qwen2.5-1.5B',
         'Llama-3.2-3B', 'Mistral-7B', 'Qwen2.5-7B', 'Llama-3-8B', 'Llama-3.3-70B', 'DeepSeek-V4-Pro', 'GPT-4.1']
CATS = ['Ins/Pos', 'Mtx/Emb', 'Ser/Mul']
name = lambda m: m + ('$^\\dagger$' if m in ('Llama-3.2-3B', 'Mistral-7B') else '')
fmt = lambda v: ('+' if v >= 0 else '-') + f'{abs(v):.2f}'

n = pd.read_csv(R / 'step3_all_vs_ns.csv').set_index(['model', 'category'])
rows = []
for m in ORDER:
    cells = []
    for c in CATS:
        r = n.loc[(m, c)]
        s = lambda v, p: fmt(v) + ('$^{*}$' if p < .05 else '')
        cells += [s(r.median_all, r.p_all), s(r.median_sub, r.p_sub),
                  fmt(r.median_rest) + ('$^{\\ddagger}$' if r.p_sub_vs_rest < .05 else '')]
    rows.append(name(m) + ' & ' + ' & '.join(cells) + r' \\')
    if m == 'BART-large':
        rows.append(r'\midrule')
open(OUT / 'tabA_bias.tex', 'w').write(r"""\begin{table*}[t]
\centering\small
\begin{tabular}{lccccccccc}
\toprule
Model & \multicolumn{3}{c}{Ins/Pos} & \multicolumn{3}{c}{Mtx/Emb} & \multicolumn{3}{c}{Ser/Mul} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
 & all & amb & other & all & amb & other & all & amb & other \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Median attachment bias $b_x$ on all items, on human-ambiguous items (\emph{amb}; Ins/Pos 31, Mtx/Emb 9, Ser/Mul 9) and on the remaining items where humans have a significant preference (\emph{other}). $^{*}$Wilcoxon signed-rank $p<.05$ against 0; $^{\ddagger}$Mann--Whitney $p<.05$, amb vs.\ other (uncorrected). $^\dagger$Excluded models.}
\label{tab:bias}
\end{table*}
""")

d = pd.read_csv(RESULTS)
d = d[d.slot.isin(['amb_high', 'amb_low'])]
cols = ['p_entailment', 'p_neutral', 'p_contradiction']
a = d.groupby(['model', 'category', 'item_id', 'slot'])[cols].mean()
a = a.div(a.sum(axis=1), axis=0).reset_index()
pe = a.pivot_table(index=['model', 'category', 'item_id'], columns='slot', values='p_entailment')
both = ((pe.amb_high >= .5) & (pe.amb_low >= .5)).groupby(['model', 'category']).mean()
pn = a.groupby('model').p_neutral.mean()
c2 = pd.read_csv(R / 'step2_contradiction.csv').set_index(['model', 'category'])
rows = []
for m in ORDER:
    cells = []
    for c in CATS:
        cells += [f'{100 * both[(m, c)]:.0f}', f'{int(c2.loc[(m, c)].neither_entailed)}',
                  f'{100 * c2.loc[(m, c)].loser_more_C_than_N:.0f}']
    rows.append(name(m) + ' & ' + ' & '.join(cells) + f' & {pn[m]:.2f}' + r' \\')
    if m == 'BART-large':
        rows.append(r'\midrule')
open(OUT / 'tabA_rq2.tex', 'w').write(r"""\begin{table*}[t]
\centering\small
\begin{tabular}{lcccccccccc}
\toprule
Model & \multicolumn{3}{c}{Ins/Pos ($n{=}50$)} & \multicolumn{3}{c}{Mtx/Emb ($n{=}50$)} & \multicolumn{3}{c}{Ser/Mul ($n{=}30$)} & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
 & both & neither & rej. & both & neither & rej. & both & neither & rej. & $\bar P(N)$ \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\caption{What models do with the two readings of an ambiguous premise. \emph{both}: \% of items where both readings get $P(E)\geq0.5$; \emph{neither}: number of items where both readings are more neutral or contradicted than entailed; \emph{rej.}: \% of items where the losing reading gets $P(C)>P(N)$; $\bar P(N)$: mean neutral probability over all ambiguous pairs. $^\dagger$Excluded models.}
\label{tab:rq2}
\end{table*}
""")
print(f'wrote {OUT}/tabA_bias.tex, tabA_rq2.tex')

# ---------------- human summary table (tab:human) ----------------------------------
import numpy as np
from scipy.stats import wilcoxon
from model_analysis import load_humans
H, black, *_ = load_humans(HUMAN_DIR / 'key.csv', HUMAN_DIR / 'v1.csv', HUMAN_DIR / 'v2.csv')
K = H[[p for p in H.columns if p not in black]]
cat = pd.Series({i: {'I': 'Ins/Pos', 'T': 'Mtx/Emb', 'S': 'Ser/Mul'}[i[0]] for i in K.index})
lean = K.mean(axis=1); se = K.std(axis=1) / np.sqrt(K.notna().sum(axis=1))
lab = pd.Series(np.select([lean + 1.96 * se < 0, lean - 1.96 * se > 0], ['H', 'L'], 'ns'), index=K.index)
rows = []
for c in CATS:
    ids = cat.index[cat == c]
    pp = K.loc[ids].mean(axis=0)                       # each participant's mean lean in this category
    p = wilcoxon(pp).pvalue
    ps = '.' + f'{p:.2f}'.split('.')[1] if p >= .001 else '$<.001$'
    v = lab[ids].value_counts()
    rows.append(f"{c} & {len(ids)} & ${lean[ids].mean():+.2f}$ & {ps} & {v.get('H', 0)} & {v.get('L', 0)} & {v.get('ns', 0)} \\\\")
open(OUT / 'tabA_human.tex', 'w').write(r"""\begin{table}[h]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrr}
\toprule
 & $n$ & lean & $p$ & H & L & n.s. \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Human judgments per category. \emph{lean}: mean lean ($-2$ High, $+2$ Low); $p$: Wilcoxon signed-rank test of participants' mean lean against 0 ($n=17$ participants); H / L / n.s.: number of items whose 95\% CI lies below 0, above 0, or includes 0.}
\label{tab:human}
\end{table}
""")
print(f'wrote {OUT}/tabA_human.tex'); print('\n'.join(rows))
