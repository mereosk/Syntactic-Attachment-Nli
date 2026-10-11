"""Item-level and summary metrics from the scored pairs."""
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wilcoxon, friedmanchisquare

P_COLS = ["p_entailment", "p_neutral", "p_contradiction"]


def win(a, b, tol=1e-6):          # 1 = a wins, 0 = b wins, 0.5 = tie
    a, b = a.to_numpy(), b.to_numpy()
    return np.where(np.abs(a - b) <= tol, 0.5, (a > b).astype(float))


def item_metrics(scores):
    agg = (scores.groupby(["model", "family", "item_id", "category", "slot"], as_index=False)
                 [P_COLS + ["coverage"]].mean())
    idx = ["model", "family", "item_id", "category"]
    pe, pn, pc = (agg.pivot_table(index=idx, columns="slot", values=c) for c in P_COLS)
    out = pe.index.to_frame(index=False)
    H = np.stack([pe["amb_high"], pn["amb_high"], pc["amb_high"]], axis=1)
    L = np.stack([pe["amb_low"],  pn["amb_low"],  pc["amb_low"]],  axis=1)
    out["low_wins"]    = win(pe["amb_low"], pe["amb_high"])
    out["srb_diff"]    = (pe["amb_low"] - pe["amb_high"]).to_numpy()
    out["js3"]         = [jensenshannon(h, l, base=2) for h, l in zip(H, L)]
    out["neutral_amb"] = ((pn["amb_high"] + pn["amb_low"]) / 2).to_numpy()
    out["ctrlH_ok"]    = win(pe["ctrlH_ent"], pe["ctrlH_non"])
    out["ctrlL_ok"]    = win(pe["ctrlL_ent"], pe["ctrlL_non"])
    return out


def summarise(items, by=("model", "family", "category")):
    s = items.groupby(list(by)).agg(
        SRB_rate=("low_wins", lambda x: x.mean() - 0.5),
        SRB_mean=("srb_diff", "mean"),
        JS3=("js3", "mean"),
        neutral_amb=("neutral_amb", "mean"),
        acc_H=("ctrlH_ok", "mean"),
        acc_L=("ctrlL_ok", "mean"),
        n=("item_id", "count"),
    ).reset_index()
    s["DA"] = s["acc_H"] - s["acc_L"]
    return s


def safe(f):
    try:
        return f()
    except ValueError:
        return np.nan            # no variation at all -> test undefined


def paired_p(wide):
    cols = [wide[c] for c in wide.columns]
    return wilcoxon(*cols).pvalue if len(cols) == 2 else friedmanchisquare(*cols).pvalue
