"""Appendix table tab:clear: % items with a clear High / clear Low preference.
Clear = preferred reading has P(E) >= 0.5 and |b_x| >= TH (main 0.3; robustness 0.2, 0.4).
  python scripts/paper_tab_clear.py [0.3]   ->  results/paper/tabA_clear.tex (tabA_clear_<TH>.tex otherwise)"""
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from src.config import RESULTS
from human_analysis import HUMAN_DIR
R, OUT = ROOT / "results" / "analysis", ROOT / "results" / "paper"   # model_analysis.py output -> paper assets
OUT.mkdir(parents=True, exist_ok=True)
from model_analysis import load_models, item_table
TH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
ORDER = ["DeBERTa-v3-base", "DeBERTa-v3-large", "RoBERTa-large", "BART-large", "Flan-T5-large",
         "Qwen2.5-1.5B", "Llama-3.2-3B", "Mistral-7B", "Qwen2.5-7B", "Llama-3-8B",
         "Llama-3.3-70B", "DeepSeek-V4-Pro", "GPT-4.1"]
EXCL = {"Llama-3.2-3B", "Mistral-7B"}; CATS = ["Ins/Pos", "Mtx/Emb", "Ser/Mul"]
_, m, _ = load_models(RESULTS); t = item_table(m)
t["clear"] = (t.winner_pE >= 0.5) & (t.bias.abs() >= TH)
rows = []
for i, mo in enumerate(ORDER):
    cells = []
    for c in CATS:
        g = t[(t.model == mo) & (t.category == c)]
        hi = round(100 * (g.clear & (g.bias < 0)).mean()); lo = round(100 * (g.clear & (g.bias > 0)).mean())
        cells.append(f"{hi} / {lo}")
    rows.append(("\\midrule\n" if i == 4 else "") + mo + ("$^\\dagger$" if mo in EXCL else "") + " & " + " & ".join(cells) + " \\\\")
tex = r"""\begin{table}[t]
\centering
\small
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccc}
\toprule
Model & Ins/Pos & Mtx/Emb & Ser/Mul \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Share of items (\%) with a clear High / clear Low preference: the preferred reading has $P(E)\geq 0.5$ and $|b_x|\geq """ + f"{TH:g}" + r"""$. Remaining items show no clear preference. $^\dagger$Excluded models.}
\label{tab:clear}
\end{table}
"""
out = OUT / ("tabA_clear.tex" if TH == 0.3 else f"tabA_clear_{TH}.tex")
open(out, "w").write(tex); print(out)
