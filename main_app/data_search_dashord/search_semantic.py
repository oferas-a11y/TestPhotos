import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore


EMB_DIR = Path("main_app_outputs") / "data_search_dashord"
EMB_NPZ = EMB_DIR / "embeddings_minilm_l6.npz"
META_JSON = EMB_DIR / "embeddings_meta.json"


def load_embeddings() -> Tuple[np.ndarray, List[Dict[str, str]]]:
    if not EMB_NPZ.exists() or not META_JSON.exists():
        raise FileNotFoundError("Embeddings or meta not found. Run build_embeddings.py first.")
    data = np.load(EMB_NPZ)
    embs = data['embeddings']
    meta = json.loads(META_JSON.read_text()).get('rows', [])
    return embs, meta


def search(query: str, k: int = 5, threshold: float = 0.35) -> Tuple[List[Dict[str, str]], int]:
    embs, meta = load_embeddings()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = (embs @ q)
    num_above = int((sims >= threshold).sum())
    idxs = np.argsort(-sims)[:max(1, k)]
    return [meta[int(i)] for i in idxs], num_above


def write_report(query: str, rows: List[Dict[str, str]]) -> Path:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = EMB_DIR / f"semantic_{ts}.txt"
    lines: List[str] = [f"Query: {query}", f"Total returned: {len(rows)}", ""]
    for i, r in enumerate(rows, 1):
        orig = r.get('original_path', '')
        desc = r.get('description', '')
        lines.append(f"{i}. {orig}")
        if desc:
            lines.append(f"   {desc}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_gallery(query: str, rows: List[Dict[str, str]]) -> Path:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = EMB_DIR / f"semantic_{ts}.html"
    def esc(t: str) -> str:
        return (t or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    parts: List[str] = []
    parts.append("<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Semantic Search Results</title>")
    parts.append("<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px} .item{margin-bottom:30px;display:flex;gap:16px} img{max-width:360px;height:auto;border:1px solid #ddd} .meta{max-width:800px}</style>")
    parts.append("</head><body>")
    parts.append(f"<h2>Query: {esc(query)}</h2>")
    parts.append(f"<p>Total returned: {len(rows)}</p>")
    for i, r in enumerate(rows, 1):
        orig = r.get('original_path', '')
        desc = r.get('description', '')
        parts.append('<div class="item">')
        parts.append(f"<div><img src=\"{esc(orig)}\" alt=\"{esc(Path(orig).name)}\"></div>")
        parts.append("<div class=\"meta\">")
        parts.append(f"<h3>{i}. {esc(orig)}</h3>")
        if desc:
            parts.append(f"<p>{esc(desc)}</p>")
        parts.append("</div>")
        parts.append("</div>")
    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def main() -> None:
    print("Semantic search (MiniLM-L6). Type your query.")
    q = input("Query: ").strip()
    if not q:
        print("Empty query")
        return
    k = input("Top K (1/5/10): ").strip() or '5'
    try:
        k_int = int(k)
    except Exception:
        k_int = 5
    results, num_above = search(q, k=k_int)
    print(f"\nFound {num_above} items above similarity threshold. Top {len(results)} matches:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.get('original_path','')}")
        desc = r.get('description','')
        if desc:
            print(f"   {desc}")
        print("")
    out_txt = write_report(q, results)
    out_html = write_gallery(q, results)
    print(f"Wrote report: {out_txt}")
    print(f"Wrote gallery: {out_html}")


if __name__ == "__main__":
    main()


