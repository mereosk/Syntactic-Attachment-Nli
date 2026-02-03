import json
from .config import DATA_FILE


def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)