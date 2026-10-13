"""Configuration constants."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "final_dataset.json"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Local runs are CPU-only; large models are queried through APIs.
DEVICE = "cpu"
SEED = 0

PROMPT_FORMAT = "letters"        # "letters" (main) or "words" (old yes/maybe/no)
SECTIONS      = ["S", "I", "T"]     # item sections to (re)score this run: any of "S", "I", "T"
MODELS_TO_RUN = None             # None = all models, or e.g. ["GPT-4.1", "DeepSeek-V4-Pro"]
REPLACE       = True             # True : overwrite existing rows for these models x sections
                                 # False: skip model x section combos already in the results file
                                 #        (use this to resume after a crash)

RESULTS = RESULTS_DIR / f"nli_scores_{PROMPT_FORMAT}.csv"

# optional: seed this session with earlier results / API cache
SEED_RESULTS = None   # e.g. RESULTS_DIR / "nli_scores_long.csv"  (old run = words format)
SEED_CACHE   = None   # e.g. RESULTS_DIR / "openrouter_cache.json"
API_CACHE_PATH = RESULTS_DIR / "api_cache.json"

CATEGORY = {
    "Verb Instrument vs Object Attribute": "Ins/Pos",
    "Matrix vs Embedded Verb Attachment":  "Mtx/Emb",
    "Serial Attachment":                   "Ser/Mul",
}
# hypothesis_a = High attachment, hypothesis_b = Low attachment
SLOTS = {
    "amb_high":  ("premise",   "hypothesis_a"),
    "amb_low":   ("premise",   "hypothesis_b"),
    "ctrlH_ent": ("premise_a", "hypothesis_aa"),
    "ctrlH_non": ("premise_a", "hypothesis_ab"),
    "ctrlL_ent": ("premise_b", "hypothesis_bb"),
    "ctrlL_non": ("premise_b", "hypothesis_ba"),
}



# ---------- API models (top-20 logprobs) ---------------------------------------
API_MODELS = {
    "meta-llama/llama-3.3-70b-instruct": {
        "backend": "openrouter", "workers": 4,
        "provider": {"order": ["AkashML"], "allow_fallbacks": False, "require_parameters": True}},
    "deepseek/deepseek-v4-pro": {
        "backend": "openrouter", "workers": 2,
        "provider": {"order": ["StreamLake"], "allow_fallbacks": False, "require_parameters": True},
        "extra": {"reasoning": {"enabled": False}}},
    "gpt-4.1": {"backend": "openai", "workers": 4},
}
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL     = "https://api.openai.com/v1/chat/completions"


MODELS = [
    # kind,        HF id / API slug,                                           short name
    # ("classifier", "cross-encoder/nli-deberta-v3-base",                         "DeBERTa-v3-base"),
    # ("classifier", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",  "DeBERTa-v3-large"),
    # ("classifier", "roberta-large-mnli",                                        "RoBERTa-large"),
    # ("classifier", "facebook/bart-large-mnli",                                  "BART-large"),
    # ("seq2seq",    "google/flan-t5-large",                                      "Flan-T5-large"),
    # ("causal",     "meta-llama/Llama-3.2-3B-Instruct",                          "Llama-3.2-3B"),
    # ("causal",     "meta-llama/Meta-Llama-3-8B-Instruct",                       "Llama-3-8B"),
    ("api",        "meta-llama/llama-3.3-70b-instruct",                         "Llama-3.3-70B"),
    # ("causal",     "Qwen/Qwen2.5-1.5B-Instruct",                                "Qwen2.5-1.5B"),
    # ("causal",     "Qwen/Qwen2.5-7B-Instruct",                                  "Qwen2.5-7B"),
    # ("causal",     "mistralai/Mistral-7B-Instruct-v0.2",                        "Mistral-7B"),
    ("api",        "deepseek/deepseek-v4-pro",                                  "DeepSeek-V4-Pro"),
    ("api",        "gpt-4.1",                                                   "GPT-4.1"),
]
MODEL_ORDER = [m[2] for m in MODELS]
FAMILY = {m[2]: ("classifier" if m[0] == "classifier" else "generative") for m in MODELS}
