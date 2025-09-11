from app import app
with app.test_client() as c:
    r = c.get('/debug/first')
    print('first:', r.status_code, len(r.data))
    g = c.get('/debug/gallery?limit=6')
    print('gallery:', g.status_code, len(g.data))
    print('has <img> in gallery:', b'<img' in g.data)
