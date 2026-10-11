import json
import os

import pandas as pd

from .config import DATA_FILE, CATEGORY, SLOTS, RESULTS


def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_pairs(data):
    """Expand each item into one premise/hypothesis pair per slot."""
    all_pairs = []
    for item in data:
        for slot, (p_key, h_key) in SLOTS.items():
            all_pairs.append({"item_id": item["id"], "section": item["id"][0],
                              "category": CATEGORY[item["type"]], "slot": slot,
                              "premise": item[p_key], "hypothesis": item[h_key]})
    return all_pairs


def load_results():
    if not os.path.exists(RESULTS):
        return pd.DataFrame()
    df = pd.read_csv(RESULTS)
    if "section" not in df.columns:            # files from the earlier run
        df["section"] = df["item_id"].str[0]
    return df


def save_model_rows(short, new_rows, sections):
    df = load_results()
    if len(df):
        df = df[~((df["model"] == short) & (df["section"].isin(sections)))]
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(RESULTS, index=False)
