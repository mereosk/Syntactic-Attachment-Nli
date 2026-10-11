"""
Standalone script to run full evaluation.
Usage: python scripts/run_evaluation.py
"""


import os
import shutil

import numpy as np
import pandas as pd
import torch
from huggingface_hub import login

from src.config import (SEED, HF_TOKEN,
                        OPENROUTER_API_KEY, OPENAI_API_KEY,
                        RESULTS, SEED_RESULTS, SEED_CACHE, API_CACHE_PATH,
                        SECTIONS, MODELS, MODEL_ORDER, MODELS_TO_RUN, REPLACE,
                        RESULTS_DIR, PROMPT_FORMAT)
from src.utils import load_data, build_pairs, load_results, save_model_rows
from src.prompts import VARIANTS, build_prompt
from src.scoring import run_one
from src.metrics import item_metrics, summarise, safe, paired_p
from src.visualization import plot_results


def main():
    """Run complete model evaluation."""
    torch.manual_seed(SEED)
    if HF_TOKEN:
        login(HF_TOKEN)
    print("keys loaded | OpenRouter:", bool(OPENROUTER_API_KEY), "| OpenAI:", bool(OPENAI_API_KEY))

    if SEED_RESULTS and not os.path.exists(RESULTS):
        shutil.copy(SEED_RESULTS, RESULTS); print("seeded results from", SEED_RESULTS)
    if SEED_CACHE and not os.path.exists(API_CACHE_PATH):
        shutil.copy(SEED_CACHE, API_CACHE_PATH); print("seeded API cache from", SEED_CACHE)
    
    print("="*100)
    print("STARTING NLI MODEL EVALUATION")
    print("="*100)
    
    # Load data
    data = load_data()
    
    print(f"\nLoaded {len(data)} examples")

    all_pairs = build_pairs(data)
    pairs = [p for p in all_pairs if p["section"] in SECTIONS]
    print(f"{len(data)} triplets -> {len(all_pairs)} pairs; this run scores {len(pairs)} pairs "
          f"(sections {SECTIONS})")

    print(f"format={PROMPT_FORMAT} | {len(VARIANTS)} prompts per pair\n")
    txt, amap = build_prompt(VARIANTS[0][0], VARIANTS[0][1], all_pairs[0]["premise"], all_pairs[0]["hypothesis"])
    print(txt, "\n\nanswer map:", amap)

    existing = load_results()
    for kind, name, short in MODELS:
        if MODELS_TO_RUN is not None and short not in MODELS_TO_RUN:
            continue
        todo = list(SECTIONS)
        if not REPLACE and len(existing):
            done = set(existing.loc[existing["model"] == short, "section"])
            todo = [s for s in todo if s not in done]
        if not todo:
            print(f"skip {short} (sections {SECTIONS} already scored)")
            continue
        pairs_ = [p for p in pairs if p["section"] in todo]
        print(f"\n=== {short} ({kind}) | sections {todo} | {len(pairs_)} pairs ===")
        try:
            rows = run_one(kind, name, short, pairs_)
        except Exception as e:
            print(f"FAILED {short}: {e}")
            continue
        save_model_rows(short, rows, todo)
        print(f"  saved {len(rows)} rows | mean coverage = {np.mean([r['coverage'] for r in rows]):.3f}")

    print("\nrows per model x section:")
    print(load_results().pivot_table(index="model", columns="section", values="slot",
                                     aggfunc="size").reindex(MODEL_ORDER))

    scores = load_results()
    items = item_metrics(scores)
    summary = summarise(items)
    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    summary = summary.sort_values(["model", "category"])
    summary.to_csv(RESULTS_DIR / f"summary_{PROMPT_FORMAT}.csv", index=False)
    print(summary.round(3).to_string(index=False))

    coverage = (scores.groupby("model")
                      .agg(mean_coverage=("coverage", "mean"),
                           min_coverage=("coverage", "min"),
                           share_missing=("n_missing", lambda x: (x > 0).mean()))
                      .reindex(MODEL_ORDER).dropna(how="all"))
    print("\nCoverage (classifiers = 1 by definition; API = lower bound):")
    print(coverage.round(3))
    cov_tmpl = (scores[scores["family"] == "generative"]
                .pivot_table(index="model", columns="template", values="coverage")
                .reindex(MODEL_ORDER).dropna(how="all"))
    print("\nCoverage per template (LLMs):")
    print(cov_tmpl.round(3))

    gen = scores[scores["family"] == "generative"]
    rob_rows = []
    for model, g in gen.groupby("model"):
        by_order = g.groupby(["item_id", "slot", "order"])["p_entailment"].mean().unstack()
        by_tmpl  = g.groupby(["item_id", "slot", "template"])["p_entailment"].mean().unstack()
        rob_rows.append({
            "model": model,
            "order_spread":    by_order.mean().max() - by_order.mean().min(),
            "order_p":         safe(lambda: paired_p(by_order)),
            "template_spread": by_tmpl.mean().max() - by_tmpl.mean().min(),
            "template_p":      safe(lambda: paired_p(by_tmpl)),
        })
    robust = pd.DataFrame(rob_rows).set_index("model").reindex(MODEL_ORDER).dropna(how="all")
    print(robust.round(4))

    variant_srb = []
    for (t, o), g in gen.groupby(["template", "order"]):
        s = summarise(item_metrics(g), by=("model", "category"))
        s["variant"] = f"{t}-{o}"
        variant_srb.append(s[["model", "category", "variant", "SRB_rate"]])
    variant_srb = pd.concat(variant_srb)
    sign_table = variant_srb.pivot_table(index=["model", "category"], columns="variant", values="SRB_rate")
    sign_table["sign_flips"] = sign_table.apply(lambda r: len(set(np.sign(r[r != 0]))) > 1, axis=1)
    print("\nSRB_rate per prompt variant:")
    print(sign_table.round(3).to_string())

    robust.to_csv(RESULTS_DIR / f"robustness_{PROMPT_FORMAT}.csv")
    sign_table.to_csv(RESULTS_DIR / f"sign_stability_{PROMPT_FORMAT}.csv")

    plot_results(summary)

    print("\n" + "="*100)
    print("EVALUATION COMPLETE!")
    print("="*100)
    print("\nOutputs saved to:")
    print(f"  - {RESULTS}")
    print(f"  - summary_{PROMPT_FORMAT}.csv")
    print(f"  - robustness_{PROMPT_FORMAT}.csv")
    print(f"  - sign_stability_{PROMPT_FORMAT}.csv")
    print(f"  - results_{PROMPT_FORMAT}.png")


if __name__ == "__main__":
    main()
