#!/usr/bin/env python3
"""
Seed the Pinecone index directly from data_text.csv using Sentence-Transformers.

Use this if your Pinecone index is empty in deployment and you cannot
run the Chroma migration. It reads main_app_outputs/results/data_text.csv and
upserts 384-D text embeddings (MiniLM-L6-v2) with metadata including original_path.

Requirements:
- PINECONE_API_KEY in environment
- sentence-transformers and pinecone installed (see requirements.txt)
"""

import os
import csv
import time
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

# Project imports
from main_app.modules.pinecone_handler import create_pinecone_handler


def read_rows(csv_path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not csv_path.exists():
        print(f"❌ Not found: {csv_path}")
        return rows
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            rows.append(row)
            if limit and i >= limit:
                break
    print(f"📄 Loaded {len(rows)} rows from {csv_path}")
    return rows


def main():
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        print("❌ PINECONE_API_KEY missing. Add it to .env or environment.")
        return

    csv_path = Path('main_app_outputs/results/data_text.csv')
    rows = read_rows(csv_path)
    if not rows:
        print("⚠️ No rows to index. Build data_text.csv first.")
        return

    print("🔤 Loading Sentence-Transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    print("🔌 Connecting to Pinecone...")
    handler = create_pinecone_handler()
    if handler is None:
        print("❌ Failed to create Pinecone handler")
        return

    print("🚀 Upserting embeddings in batches...")
    batch: List[Dict[str, Any]] = []
    stored = 0
    for row in rows:
        original_path = row.get('original_path', '').strip()
        text = row.get('comprehensive_text', '').strip() or row.get('text', '').strip() or original_path

        # Create a stable-ish ID using the file name + timestamp suffix to avoid collisions
        base = Path(original_path).name or "photo"
        pid = f"photo_{int(time.time()*1000)%10**10:010d}_{base}"

        vec = model.encode([text])[0].tolist()
        rec = {
            'id': pid,
            'embedding': vec,
            'metadata': {
                'original_path': original_path,
                'source': 'seed_csv',
            },
            'document': text,
        }
        batch.append(rec)

        if len(batch) >= 100:
            handler.store_batch_analysis(batch)
            stored += len(batch)
            print(f"✅ Upserted {stored} records...")
            batch = []

    if batch:
        handler.store_batch_analysis(batch)
        stored += len(batch)
        print(f"✅ Upserted {stored} records total")

    stats = handler.get_collection_stats()
    print(f"📊 Pinecone stats: {stats}")
    print("🎉 Done. Your /api/photos should now return items.")


if __name__ == '__main__':
    main()

