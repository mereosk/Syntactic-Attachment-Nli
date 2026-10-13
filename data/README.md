# Data

## `final_dataset.json`

130 triplets (JSON list). Fields:

| Field | Meaning |
|---|---|
| `id` | `I1`–`I50` Ins/Pos, `T1`–`T50` Mtx/Emb, `S1`–`S30` Ser/Mul |
| `type` | `Verb Instrument vs Object Attribute`, `Matrix vs Embedded Verb Attachment`, `Serial Attachment` |
| `premise` | ambiguous premise |
| `hypothesis_a` / `hypothesis_b` | follows only from the High / Low reading |
| `premise_a` | High-attachment control premise, with `hypothesis_aa` (entailed) and `hypothesis_ab` (not entailed) |
| `premise_b` | Low-attachment control premise, with `hypothesis_bb` (entailed) and `hypothesis_ba` (not entailed) |
| `attachment_a` / `attachment_b` | structural label of each reading (`Subject/Verb` / `Object`, `Matrix Verb (V1)` / `Embedded Verb (V2)`, `Flat (Modifies Verb)` / `Nested (Modifies Noun 1)`) |

Each triplet yields 6 NLI pairs: `amb_high`, `amb_low`, `ctrlH_ent`, `ctrlH_non`, `ctrlL_ent`, `ctrlL_non` (see `SLOTS` in `src/config.py`).

## `human_results/`

| File | Content |
|---|---|
| `raw_v1.csv`, `raw_v2.csv` | form exports of the two counterbalanced versions (A/B order of the readings swapped); no names or contact details |
| `key.csv` | per version: question title → item id, which reading was option A, expected answer of the 10 attention checks |
| `v1.csv`, `v2.csv` | cleaned exports, written by `scripts/human_prep.py` (git-ignored) |

`human_prep.py` maps the two sentences that were reworded during collection (T21, T38) to their item ids.

## `archive/`

Earlier dataset versions, kept for reference. No script uses them; `final_dataset.json` is the dataset used in the paper.
