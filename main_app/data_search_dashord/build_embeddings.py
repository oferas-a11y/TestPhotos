import csv
import json
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer  # type: ignore
import numpy as np


RESULTS_DIR = Path("main_app_outputs") / "results"
DATA_TEXT = RESULTS_DIR / "data_text.csv"
EMB_DIR = Path("main_app_outputs") / "data_search_dashord"
EMB_NPZ = EMB_DIR / "embeddings_minilm_l6.npz"
META_JSON = EMB_DIR / "embeddings_meta.json"


def load_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(DATA_TEXT, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v or "").strip() for k, v in r.items()})
    return rows


def build() -> None:
    rows = load_rows()
    if not rows:
        print("No rows in data_text.csv. Run the main app first.")
        return
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    texts = [r.get('description', '') for r in rows]
    embs = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    np.savez_compressed(EMB_NPZ, embeddings=embs)
    # Save meta for mapping back
    meta = [{
        'original_path': r.get('original_path', ''),
        'colorized_path': r.get('colorized_path', ''),
        'full_results_path': r.get('full_results_path', ''),
        'llm_json_path': r.get('llm_json_path', ''),
        'description': r.get('description', '')
    } for r in rows]
    with open(META_JSON, 'w', encoding="utf-8") as f:
        json.dump({'rows': meta}, f, indent=2)
    print(f"Wrote embeddings: {EMB_NPZ}")
    print(f"Wrote meta: {META_JSON}")


if __name__ == "__main__":
    build()


