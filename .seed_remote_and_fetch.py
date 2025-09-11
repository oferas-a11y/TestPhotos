import urllib.parse, json
from urllib.request import urlopen, Request, urlretrieve
BASE='https://testphotos-production.up.railway.app'
path='photo_collections/project Photography/HUMILIATION and USHMM/789182.jpeg'
url=f"{BASE}/api/admin/seed_one?path="+urllib.parse.quote(path)
req=Request(url, headers={'User-Agent':'curl/8'})
with urlopen(req, timeout=120) as r:
    data=r.read().decode('utf-8')
print(data)
js=json.loads(data)
thumb=BASE+js['thumbnail_url'] if js['thumbnail_url'].startswith('/') else js['thumbnail_url']
image=BASE+js['image_url'] if js['image_url'].startswith('/') else js['image_url']
print('thumb:',thumb)
print('image:',image)
urlretrieve(thumb,'remote_seed_thumb.jpg')
urlretrieve(image,'remote_seed_full.jpg')
print('saved')
