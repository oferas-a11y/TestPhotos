from app import app
import json

with app.test_client() as c:
    data = c.get('/api/photos?limit=1').get_json()
    item = data['results'][0]
    pid = item['id']
    with open('out_first.json','w') as f:
        json.dump(item, f, indent=2)
    t = c.get(f"/api/photo/{pid}/thumbnail")
    with open('out_first_thumb.jpg','wb') as f:
        f.write(t.data)
    i = c.get(f"/api/photo/{pid}/image")
    with open('out_first_full.jpg','wb') as f:
        f.write(i.data)
    print('saved:', pid)
