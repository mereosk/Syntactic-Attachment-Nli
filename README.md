# Syntactic Attachment Ambiguity in NLI Models

Evaluation of how NLI models handle syntactic attachment ambiguities across three linguistic phenomena.

## 📊 Dataset

60 minimal pair triplets (180 premise-hypothesis pairs):
- **Temporal Attachment** (20): High vs. Low attachment  
- **Instrument vs. Possession** (20): Prepositional phrase attachment
- **Serial/Multiple Attachment** (20): Nested vs. Flat modification

### Data Generation
Data was generated through:
1. Manual seed creation (10 per phenomenon)
2. LLM generation with Gemini Pro 1.5
3. Ensemble validation (Claude Sonnet 4 + GPT-4o)
4. Filtering (both hypotheses ≥60, difference ≤20)

See `docs/methodology.md` for details.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/syntactic-attachment-nli.git
cd syntactic-attachment-nli
pip install -r requirements.txt
```

> ⚠️ Note: This project uses package-based imports.
> Always run scripts using `python -m ...` from the repository root.


### Run Analysis

**Option 1: Command Line**
```bash
python -m scripts.run_evaluation
```

**Option 2: Jupyter Notebook**
```bash
jupyter notebook notebooks/model_evaluation.ipynb
```

**Option 3: Import as Module**
```python
from src.utils import load_data
from src.analysis import analyze_model_by_phenomenon

data = load_data()
results = analyze_model_by_phenomenon("roberta-large-mnli", data)
```

## 📈 Results

Evaluated models:
- cross-encoder/nli-deberta-v3-base
- DeBERTa-v3-large (robust)
- roberta-large-mnli
- facebook/bart-large-mnli

### Key Findings
- **Low JS distance** (0.22-0.31): Models don't treat baselines as ambiguous
- **Strong biases** (58-68%): Systematic preferences for specific attachments
- **Asymmetric accuracy** (88% vs 53%): Better at one disambiguation type

See `results/` for detailed outputs.

## 📁 Repository Structure
```
syntactic-attachment-nli/
├── data/                  # Dataset (60 triplets)
├── src/                   # Core package
│   ├── __init__.py
│   ├── analysis.py
│   ├── visualization.py
│   ├── utils.py
│   └── config.py
├── scripts/               # Entry points
│   ├── __init__.py
│   └── run_evaluation.py
├── notebooks/             # Jupyter notebook
├── results/               # Generated figures and tables
├── docs/                  # Documentation
└── requirements.txt
```

## 🔧 Dependencies

- transformers
- torch  
- pandas
- matplotlib
- seaborn
- scipy

See `requirements.txt` for versions.

## 📝 Citation
```bibtex
@misc{yourname2025syntactic,
  title        = {Syntactic Attachment Ambiguity in NLI Models: Dataset and Evaluation},
  author       = {Bloemendaal, Jelle and Mereos, Konstantinos},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/mereosk/syntactic-attachment-nli}},
  note         = {Unpublished manuscript/Project repository}
}
```