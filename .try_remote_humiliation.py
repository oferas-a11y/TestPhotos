import json
from urllib.request import urlopen, Request, urlretrieve
BASE='https://testphotos-production.up.railway.app'
q='HUMILIATION and USHMM'
url=f"{BASE}/api/search/semantic?q="+q.replace(' ','%20')+"&limit=3&gemini=false"
req=Request(url, headers={'User-Agent':'curl/8'})
with urlopen(req, timeout=60) as r:
    data=json.loads(r.read().decode('utf-8'))
print('results:', len(data.get('results', [])))
if not data.get('results'):
    raise SystemExit('no results')
item=data['results'][0]
print('Choosing id', item['id'])
thumb=item['thumbnail_url']
if thumb.startswith('/'):
    thumb=BASE+thumb
urlretrieve(thumb,'remote_try_thumb.jpg')
print('saved thumb')
