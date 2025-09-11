import csv, time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from main_app.modules.pinecone_handler import create_pinecone_handler

load_dotenv()
csv_path = Path('main_app_outputs/results/data_text.csv')
assert csv_path.exists(), 'data_text.csv not found'

# Read first meaningful row
row = None
with open(csv_path, 'r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r.get('original_path') and (r.get('comprehensive_text') or r.get('text')):
            row = r
            break
assert row, 'No suitable row found in CSV'

print('Using original_path:', row.get('original_path'))
text = (row.get('comprehensive_text') or row.get('text') or '').strip()
print('Text length:', len(text))

# Encode
print('Loading MiniLM model...')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
vec = model.encode([text])[0].tolist()
print('Vector dim:', len(vec))

# Upsert
print('Connecting Pinecone...')
handler = create_pinecone_handler()
base = Path(row.get('original_path')).name or 'photo'
pid = f"photo_{int(time.time()*1000)%10**10:010d}_{base}"
rec = {
    'id': pid,
    'embedding': vec,
    'metadata': {
        'original_path': row.get('original_path'),
        'source': 'seed_one'
    },
    'document': text,
}
handler.store_batch_analysis([rec])
print('Seeded one record with id:', pid)

# Verify by quick query
results = handler.search_photos(vec, top_k=1)
print('Query returned:', len(results))
if results:
    print('Top id:', results[0].get('id'))
