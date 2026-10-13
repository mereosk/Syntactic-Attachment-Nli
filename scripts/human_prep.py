"""
Clean the raw Google-Forms exports so human_analysis.py / model_analysis.py can read them.
  python scripts/human_prep.py [r1.csv r2.csv key.csv]   ->  data/human_results/v1.csv, v2.csv
  (defaults: raw_v1.csv, raw_v2.csv, key.csv in data/human_results/)
- drops empty '[Row 1]' / 'Column N' leftovers from editing the forms
- merges the English-proficiency answer into one column (it moved in v1)
- strips the ' [Your answer]' grid suffix
- matches edited sentences (T21, T38) to whatever title the key uses for that version
"""
import sys, re
from pathlib import Path
import pandas as pd

HUMAN_DIR = Path(__file__).resolve().parents[1] / "data" / "human_results"

# sentences edited during data collection: all wordings that belong to the same item
ALIASES = {"T21": ["The flight crew decided to land during the storm",
                   "The flight crew decided to land at the military base"],
           "T38": ["The pilot chose to land at dawn",
                   "The pilot chose to land at the flight school"]}
PROF = "How would you describe your English language proficiency?"


def norm(t):
    return re.sub(r'[".\s]+$', "", re.sub(r'^[\s"]+', "", str(t))).strip()


def main(r1=HUMAN_DIR / "raw_v1.csv",
         r2=HUMAN_DIR / "raw_v2.csv",
         key_path=HUMAN_DIR / "key.csv"):
    key = pd.read_csv(key_path)
    alias_to_id = {norm(a): iid for iid, al in ALIASES.items() for a in al}
    for ver, src, dst in [(1, r1, HUMAN_DIR / "v1.csv"), (2, r2, HUMAN_DIR / "v2.csv")]:
        kt = key[key["version"] == ver]
        title_of = dict(zip(kt["item_id"], kt["question_title"]))
        by_norm = {norm(t): t for t in kt["question_title"]}
        r = pd.read_csv(src)
        r = r[[c for c in r.columns if not c.endswith("[Row 1]") and not c.startswith("Column")]]
        profs = [c for c in r.columns if c.startswith(PROF)]
        r["proficiency"] = r[profs].bfill(axis=1).iloc[:, 0] if profs else None
        r = r.drop(columns=profs + [c for c in r.columns if c.startswith("What is your native")])
        new = {}
        for c in r.columns:
            if not c.endswith("[Your answer]"):
                continue
            n = norm(c.rsplit(" [", 1)[0])
            if n in by_norm:
                new[c] = by_norm[n]
            elif n in alias_to_id:
                new[c] = title_of[alias_to_id[n]]
            else:
                raise ValueError(f"v{ver}: question not in key: {c}")
        r = r.rename(columns=new)
        missing = set(kt["question_title"]) - set(r.columns)
        if missing:
            raise ValueError(f"v{ver}: key titles without responses: {missing}")
        r.to_csv(dst, index=False)
        print(f"{dst}: {len(r)} participants, {len(new)} questions matched")


if __name__ == "__main__":
    main(*sys.argv[1:4])
