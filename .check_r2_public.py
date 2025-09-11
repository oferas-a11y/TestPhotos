import urllib.request
url='https://0c6e85d3cb9e92105b01f4108662ac71.r2.cloudflarestorage.com/ofers/photo_collections/project%20Photography/HUMILIATION%20and%20USHMM/789182.jpeg'
try:
    with urllib.request.urlopen(url, timeout=30) as r:
        data=r.read()
        print('HTTP', r.status, 'Content-Type', r.headers.get('Content-Type'), 'Bytes', len(data))
except Exception as e:
    print('Fetch failed:', e)
