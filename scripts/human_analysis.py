"""
Analysis of the PP-attachment human rating study.

Inputs (download each sheet from the Google Sheet with File > Download > CSV):
  key.csv           the 'key' sheet written by build_forms.gs
  responses_v1.csv  'Form Responses 1' (version 1)
  responses_v2.csv  'Form Responses 2' (version 2)
Optional:
  --models nli_scores_letters.csv   model results (long format) for the human-model comparison

Usage (defaults read data/human_results/, run scripts/human_prep.py first):
  python scripts/human_analysis.py [key.csv v1.csv v2.csv] [--models results/nli_scores_letters.csv]

Scale coding: human_lean runs from -2 (clearly the High reading) to +2 (clearly the Low reading),
so its sign matches the models' signed recency bias (positive = Low / recency).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCALE = {"Clearly A": -2, "Somewhat A": -1, "Both equally natural": 0, "Somewhat B": 1, "Clearly B": 2}
MAX_FAILED_CHECKS = 2      # participants failing more attention checks than this are excluded
BALANCED_HALF_WIDTH = 0.5  # |mean human_lean| below this = 'balanced' item
ROOT = Path(__file__).resolve().parents[1]
HUMAN_DIR = ROOT / "data" / "human_results"


def load_long(key, path, version):
    resp = pd.read_csv(path)
    # grid questions are exported as '"sentence" [Your answer]' -> strip the row suffix
    titles = set(key["question_title"])
    resp.columns = [c.rsplit(" [", 1)[0] if c.endswith("]") and c.rsplit(" [", 1)[0] in titles else c
                    for c in resp.columns]
    resp["participant"] = [f"v{version}_p{i + 1}" for i in range(len(resp))]
    qcols = [c for c in resp.columns if c in set(key["question_title"])]
    meta = [c for c in resp.columns if c not in qcols]
    long = resp.melt(id_vars=meta, value_vars=qcols, var_name="question_title", value_name="answer")
    long["version"] = version
    long = long.merge(key[key["version"] == version], on=["version", "question_title"], how="left")
    long["score_A"] = long["answer"].map(SCALE)
    bad = long[long["score_A"].isna() & long["answer"].notna()]
    if len(bad):
        raise ValueError(f"Unexpected answers, e.g. {bad['answer'].iloc[0]!r}")
    # re-orient so that negative = High reading, positive = Low reading
    high_is_a = long["high_is_A"].astype(str).str.lower().isin(["true", "1"])
    long["human_lean"] = np.where(high_is_a, long["score_A"], -long["score_A"])
    return long


def check_participants(long):
    """Attention checks per participant; excluded = more than MAX_FAILED_CHECKS failed."""
    chk = long[long["kind"] == "check"].copy()
    chk["passed"] = np.where(chk["expected"] == "High", chk["human_lean"] < 0, chk["human_lean"] > 0)
    part = chk.groupby("participant").agg(checks=("passed", "size"), failed=("passed", lambda x: int((~x).sum())))
    part["excluded"] = part["failed"] > MAX_FAILED_CHECKS
    return part


def split_half_reliability(items_long, n_splits=200, seed=0):
    """Correlation of item means between random halves of participants, Spearman-Brown corrected."""
    rng = np.random.default_rng(seed)
    parts = items_long["participant"].unique()
    rs = []
    for _ in range(n_splits):
        perm = rng.permutation(parts)
        h1, h2 = set(perm[: len(perm) // 2]), set(perm[len(perm) // 2:])
        m1 = items_long[items_long["participant"].isin(h1)].groupby("item_id")["human_lean"].mean()
        m2 = items_long[items_long["participant"].isin(h2)].groupby("item_id")["human_lean"].mean()
        both = pd.concat([m1, m2], axis=1).dropna()
        if len(both) > 2:
            r = both.corr().iloc[0, 1]
            rs.append(2 * r / (1 + r))
    return float(np.nanmean(rs)) if rs else np.nan


def model_item_lean(models_csv):
    s = pd.read_csv(models_csv)
    amb = s[s["slot"].isin(["amb_high", "amb_low"])]
    pe = (amb.groupby(["model", "item_id", "slot"])["p_entailment"].mean()
             .unstack("slot").reset_index())
    pe["model_lean"] = pe["amb_low"] - pe["amb_high"]          # + = prefers Low (same sign as human_lean)
    d = pe["model_lean"]
    pe["low_wins"] = np.where(d.abs() <= 1e-6, 0.5, (d > 0).astype(float))
    return pe[["model", "item_id", "model_lean", "low_wins"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key", nargs="?", default=HUMAN_DIR / "key.csv")
    ap.add_argument("v1", nargs="?", default=HUMAN_DIR / "v1.csv", help="output of human_prep.py")
    ap.add_argument("v2", nargs="?", default=HUMAN_DIR / "v2.csv", help="output of human_prep.py")
    ap.add_argument("--models", help="nli_scores_letters.csv (long model results)")
    ap.add_argument("--out", default=ROOT / "results" / "human", type=Path, help="output folder")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    key = pd.read_csv(a.key)
    long = pd.concat([load_long(key, a.v1, 1), load_long(key, a.v2, 2)], ignore_index=True)

    part = check_participants(long)
    keep = set(part.index[~part["excluded"]])
    print(f"participants: {len(part)} | excluded: {int(part['excluded'].sum())} "
          f"(more than {MAX_FAILED_CHECKS} failed checks)")

    items_long = long[(long["kind"] == "item") & long["participant"].isin(keep)]
    items_long[["participant", "version", "item_id", "section", "category", "answer", "human_lean"]].to_csv(
        a.out / "responses.csv", index=False)

    # per-item human lean
    it = (items_long.groupby(["item_id", "section", "category"])["human_lean"]
          .agg(n="size", mean_lean="mean", sd="std",
               share_high=lambda x: (x < 0).mean(), share_equal=lambda x: (x == 0).mean(),
               share_low=lambda x: (x > 0).mean())
          .reset_index())
    it["se"] = it["sd"] / np.sqrt(it["n"])
    it["ci_low"], it["ci_high"] = it["mean_lean"] - 1.96 * it["se"], it["mean_lean"] + 1.96 * it["se"]
    it["balanced"] = it["mean_lean"].abs() < BALANCED_HALF_WIDTH
    it["leans"] = np.select([it["ci_high"] < 0, it["ci_low"] > 0], ["High", "Low"], "n.s.")
    it.to_csv(a.out / "items.csv", index=False)

    cat = (it.groupby("category")
             .agg(items=("item_id", "size"), mean_lean=("mean_lean", "mean"),
                  balanced=("balanced", "sum"),
                  lean_High=("leans", lambda x: (x == "High").sum()),
                  lean_Low=("leans", lambda x: (x == "Low").sum()))
             .reset_index())
    cat.to_csv(a.out / "categories.csv", index=False)
    part.to_csv(a.out / "participants.csv")
    print("\nPer category (human_lean: - = High, + = Low):")
    print(cat.round(3).to_string(index=False))
    print(f"\nsplit-half reliability of item means (Spearman-Brown): {split_half_reliability(items_long):.3f}")

    if a.models:
        ml = model_item_lean(a.models).merge(it[["item_id", "category", "mean_lean", "balanced"]], on="item_id")
        rows = []
        for (model, c), g in ml.groupby(["model", "category"]):
            rho = spearmanr(g["mean_lean"], g["model_lean"]).correlation if len(g) > 2 else np.nan
            bal = g[g["balanced"]]
            rows.append(dict(model=model, category=c, n_items=len(g), spearman_rho=rho,
                             SRB_rate_all=g["low_wins"].mean() - 0.5,
                             n_balanced=len(bal),
                             SRB_rate_balanced=(bal["low_wins"].mean() - 0.5) if len(bal) else np.nan,
                             human_mean_lean=g["mean_lean"].mean()))
        comp = pd.DataFrame(rows)
        comp.to_csv(a.out / "vs_models.csv", index=False)
        print("\nHuman vs model (per model x category):")
        print(comp.round(3).to_string(index=False))

    print(f"\nwrote responses.csv, items.csv, categories.csv, participants.csv"
          + (", vs_models.csv" if a.models else "") + f" to {a.out}")


if __name__ == "__main__":
    main()
