# Exploring PP Attachment Preferences with NLI

Code, data and results for probing which prepositional-phrase attachment (High vs. Low) NLI classifiers and LLMs prefer, using NLI minimal pairs with disambiguated controls and graded human judgments.

## Data

`data/final_dataset.json`: 130 triplets, 780 NLI pairs.

| Category | IDs | Items | Example |
|---|---|---|---|
| Instrument vs. Possession (Ins/Pos) | I1–I50 | 50 | *The security guard is stopping the intruder with the flashlight.* |
| Matrix vs. Embedded (Mtx/Emb) | T1–T50 | 50 | *The knight vowed to defend the king during the ceremony.* |
| Serial vs. Multiple (Ser/Mul) | S1–S30 | 30 | *The dad is calling the kids in the tent at the campfire.* |

Each triplet has an ambiguous premise with a High-reading (`hypothesis_a`) and a Low-reading (`hypothesis_b`) hypothesis, plus two disambiguated controls (`premise_a` with `hypothesis_aa`/`hypothesis_ab`, `premise_b` with `hypothesis_bb`/`hypothesis_ba`). Field details: `data/README.md`.

`data/human_results/`: anonymous form exports from the human study (`raw_v1.csv`, `raw_v2.csv`, two counterbalanced versions) and `key.csv` (question → item, which reading was option A, attention-check answers).

## Reproduce

```bash
pip install -r requirements.txt
make          # build results/human, results/analysis, results/paper
make clean    # delete them again (keeps the model scores)
```

`make` rebuilds every table and figure from `results/nli_scores_letters.csv` and the human responses (CPU only, under a minute). Single stages: `make human`, `make analysis`, `make paper`. No `make`? Run the commands under `human`, `analysis` and `paper` in the `Makefile`, in that order.

Re-scoring the models is a separate step. It needs a GPU for the local models and API keys for GPT-4.1, Llama-3.3-70B and DeepSeek-V4-Pro (copy `.env.example` to `.env`):

```bash
pip install -r requirements-scoring.txt
make smoke    # a few API calls to check keys and log-prob coverage
make scores   # overwrites results/nli_scores_letters.csv
```

Models, prompt format and the sections to score are set in `src/config.py`.

## Layout

```
data/            final_dataset.json, human_results/, archive/ (earlier dataset versions, not used)
src/             config, prompt templates, model scoring, item metrics
scripts/
  eval_*         model scoring
  human_*        clean form exports, item/participant statistics, human figures
  model_*        analysis steps 1-5 (controls, contradiction, bias, JSD, Kendall), RQ2 counts
  paper_*        paper figures and LaTeX tables
results/
  nli_scores_letters.csv   model scores: model x item x slot x prompt variant (the only file in git)
  human/         per-item human lean with 95% CI, participants and exclusions, human figures
  analysis/      step CSVs and diagnostic figures
  paper/         figures and LaTeX tables used in the paper
docs/            dataset design, validation criteria; archive/ = notes from an earlier version
```

## Paper ↔ files

| Paper | File in `results/paper/` | Numbers from |
|---|---|---|
| Fig. 2 | `fig2_bias.pdf` | `analysis/step3_bias*.csv`, `analysis/step1_controls.csv` |
| Table 1 | `tab1_main.tex` | `analysis/step1_controls.csv`, `step2_contradiction.csv`, `step5_model_vs_human.csv` |
| Controls per category | `tabA_controls.tex` | `analysis/step1_controls.csv` |
| Bias: all / ambiguous / other | `tabA_bias.tex` | `analysis/step3_all_vs_ns.csv` |
| RQ2 per category | `tabA_rq2.tex` | `analysis/rq2_both_entailed.csv`, `step2_contradiction.csv` |
| Clear preferences | `tabA_clear.tex` | model scores (`paper_tab_clear.py 0.2` / `0.4` for robustness) |
| Kendall per category | `tabA_kendall.tex` | `analysis/step5_model_vs_human.csv` |
| Human summary | `tabA_human.tex` | `human/items.csv` |
| Robustness figure | `figA_bias_balanced.pdf` | `analysis/step3_all_vs_balanced.csv` |
| P(E) vs. P(C) | `fig_scatter_EC.pdf` | model scores |
| JSD, clustermaps | `figA_jsd.png`, `figA_cluster_*.png` | `analysis/step4_jsd.csv`, step 5 |

## Definitions

- **Bias** per item: `b = P(E | premise → Low reading) − P(E | premise → High reading)`; positive = Low. LLM scores are averaged over 3 templates × 3 option orders.
- **Human lean**: −2 (clearly High) to +2 (clearly Low). Participants failing more than 2 of 10 attention checks are excluded (2 of 19).
- **Human-ambiguous items**: the 95% CI of the item's mean lean includes 0 (main analysis); |mean lean| < 0.5 (robustness).
- **Control accuracy**: share of items where the entailed control hypothesis gets the higher P(E); flagged below 0.90.
- **Alignment**: Kendall τ-b between model bias and human lean, computed within each category.

## License

Code: MIT (`LICENSE`). Data (`data/`): CC BY 4.0.
