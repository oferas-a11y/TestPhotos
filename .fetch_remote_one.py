import os, json, sys
from urllib.request import urlopen, Request
from urllib.parse import urlencode

BASE = os.getenv('BASE_URL', 'https://testphotos-production.up.railway.app')

def get_json(url):
    req = Request(url, headers={'User-Agent':'curl/8'})
    with urlopen(req, timeout=60) as r:
        data = r.read()
    try:
        return json.loads(data.decode('utf-8'))
    except Exception:
        print('Non-JSON:', data[:200])
        raise

# Try /api/photos first
try:
    url = f"{BASE}/api/photos?" + urlencode({'limit':1})
    data = get_json(url)
    results = data.get('results', [])
except Exception as e:
    print('Failed /api/photos:', e)
    results = []

# Fallback to semantic search if empty
if not results:
    try:
        url = f"{BASE}/api/search/semantic?" + urlencode({'q':'people','limit':1,'gemini':'false'})
        data = get_json(url)
        results = data.get('results', [])
    except Exception as e:
        print('Failed /api/search/semantic:', e)
        results = []

if not results:
    print('No results from server. Base:', BASE)
    sys.exit(2)

item = results[0]
thumb_url = item.get('thumbnail_url')
image_url = item.get('image_url')
print('ID:', item.get('id'))
print('thumb:', thumb_url)
print('image:', image_url)

# Download
for name, url in [('remote_thumb.jpg', thumb_url), ('remote_full.jpg', image_url)]:
    req = Request(url, headers={'User-Agent':'curl/8'})
    with urlopen(req, timeout=120) as r:
        data = r.read()
    with open(name,'wb') as f:
        f.write(data)
    print('saved', name, len(data), 'bytes')
