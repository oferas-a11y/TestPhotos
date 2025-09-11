import os, json
from urllib.request import urlopen, Request
BASE = os.getenv('BASE_URL', 'https://testphotos-production.up.railway.app')

def get_json(url):
    req = Request(url, headers={'User-Agent':'curl/8'})
    with urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode('utf-8'))

data = get_json(f"{BASE}/api/search/semantic?q=people&limit=1&gemini=false")
item = data['results'][0]
pid = item['id']
print('ID:', pid)
meta = get_json(f"{BASE}/api/photo/{pid}")
print(json.dumps(meta, indent=2))
