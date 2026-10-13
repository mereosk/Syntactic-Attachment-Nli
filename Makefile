# Rebuild all results from the model scores and the human responses (CPU only, under a minute).
#   make            human -> analysis -> paper
#   make clean      delete everything make builds (keeps the model scores)
#   make scores     re-score all models (GPU + API keys in .env); overwrites $(SCORES)

PY     := python
SCORES := results/nli_scores_letters.csv

.PHONY: all human analysis paper scores smoke clean

all: paper

human:                                   # data/human_results/v1.csv, v2.csv and results/human/
	$(PY) scripts/human_prep.py
	$(PY) scripts/human_analysis.py --models $(SCORES)
	$(PY) scripts/human_figures.py

analysis: human                          # results/analysis/: steps 1-5, RQ2 counts
	$(PY) scripts/model_analysis.py
	$(PY) scripts/model_rq2_both.py

paper: analysis                          # results/paper/: figures and LaTeX tables used in the paper
	$(PY) scripts/paper_fig_bias.py ns
	$(PY) scripts/paper_fig_bias.py balanced
	$(PY) scripts/paper_tables.py
	$(PY) scripts/paper_tab_clear.py
	$(PY) scripts/paper_tables_extra.py  # after paper_tables.py: rewrites tabA_bias.tex

scores:
	$(PY) -m scripts.eval_models

smoke:                                   # a few API calls to check keys and log-prob coverage
	$(PY) -m scripts.eval_smoke_test

clean:
	rm -rf results/human results/analysis results/paper
	rm -f data/human_results/v1.csv data/human_results/v2.csv
