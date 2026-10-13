"""RQ2 extra numbers: share of ambiguous items where BOTH readings are entailed
(P(E) >= 0.5 for H^h and H^l), and mean P(N) on ambiguous pairs.
Distributions are averaged over the 9 prompts (LLMs) and renormalised over E/N/C.
Usage: python scripts/model_rq2_both.py   -> results/analysis/rq2_both_entailed.csv
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

d = pd.read_csv(RESULTS)
d = d[d.slot.isin(["amb_high", "amb_low"])]
cols = ["p_entailment", "p_neutral", "p_contradiction"]
a = d.groupby(["model", "category", "item_id", "slot"])[cols].mean()
a = a.div(a.sum(axis=1), axis=0).reset_index()
pe = a.pivot_table(index=["model", "category", "item_id"], columns="slot", values="p_entailment")
both = ((pe.amb_high >= 0.5) & (pe.amb_low >= 0.5)).rename("both_entailed").reset_index()

out = pd.DataFrame({
    "both_entailed_all": both.groupby("model").both_entailed.mean(),
    "mean_PN_amb": a.groupby("model").p_neutral.mean(),
})
per_cat = both.groupby(["model", "category"]).both_entailed.mean().unstack()
out = out.join(per_cat.add_prefix("both_")).round(3).sort_values("both_entailed_all")
print(out.to_string())
out.to_csv(R / "rq2_both_entailed.csv")
