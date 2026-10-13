"""
Quick API check on a few pairs before a full run.
Usage: python -m scripts.eval_smoke_test
"""


from src.utils import load_data, build_pairs
from src.prompts import TEMPLATES, ORDERS, build_prompt
from src.scoring import api_answer_probs, _save_api_cache

SMOKE_SLUGS = ["gpt-4.1", "deepseek/deepseek-v4-pro"]


def main():
    all_pairs = build_pairs(load_data())
    smoke_pairs = [p for p in all_pairs if p["slot"] in ("amb_high", "ctrlH_ent", "ctrlH_non")][:3]
    for slug in SMOKE_SLUGS:
        print(f"\n--- {slug} ---")
        for p in smoke_pairs:
            for order in ORDERS[:2]:
                txt, amap = build_prompt(list(TEMPLATES)[0], order, p["premise"], p["hypothesis"])
                res = api_answer_probs(slug, txt)
                lab_probs = {amap[w]: round(v, 3) for w, v in res["probs"].items()}
                print(f"{p['item_id']:4} {p['slot']:10} {order} | {lab_probs} | "
                      f"cov={res['coverage']:.3f} | missing={res['missing']} | {res['served_by']}")
                print("      top tokens:", res.get("top"))
    _save_api_cache()


if __name__ == "__main__":
    main()
