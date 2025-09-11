#!/usr/bin/env python3
"""
Check that a deployed server returns a real photo (not the placeholder).

Usage:
  BASE_URL=https://<your-railway> python scripts/test_server_photo.py

Pass criteria:
- Finds at least one result via /api/search/semantic (q=people) or /api/photos.
- GET /api/photo/{id}/thumbnail returns image/*, and likely not the placeholder:
  - If Content-Type is image/png AND body < 5KB, treat as placeholder → FAIL
  - Otherwise PASS (most real images are JPEGs and > 10KB)
"""

import os
import sys
import json
from urllib.request import urlopen, Request
from urllib.parse import urlencode

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BASE = os.getenv('BASE_URL')
if not BASE:
    print('Error: set BASE_URL to your server, e.g. https://testphotos-production.up.railway.app')
    sys.exit(2)

def get_json(url: str) -> dict:
    req = Request(url, headers={'User-Agent': 'curl/8'})
    with urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode('utf-8'))

def fetch(url: str):
    req = Request(url, headers={'User-Agent': 'curl/8'})
    resp = urlopen(req, timeout=60)
    data = resp.read()
    ctype = resp.headers.get('Content-Type', '')
    return ctype, data

def main():
    # Try /api/photos first
    results = []
    try:
        data = get_json(f"{BASE}/api/photos?" + urlencode({'limit': 1}))
        results = data.get('results', [])
    except Exception:
        pass

    if not results:
        # Fallback to semantic search
        data = get_json(f"{BASE}/api/search/semantic?" + urlencode({'q': 'people', 'limit': 1, 'gemini': 'false'}))
        results = data.get('results', [])

    if not results:
        print('FAIL: server returned no results from /api/photos or /api/search/semantic')
        sys.exit(1)

    item = results[0]
    pid = item.get('id')
    thumb = item.get('thumbnail_url')
    if thumb.startswith('/'):
        thumb = f"{BASE}{thumb}"

    ctype, data = fetch(thumb)
    print('Thumbnail:', thumb)
    print('Status: OK')
    print('Content-Type:', ctype)
    print('Bytes:', len(data))

    # Heuristic: placeholder is a small PNG (generated) ~< 5KB
    is_placeholder = (ctype.startswith('image/png') and len(data) < 5000)
    if is_placeholder:
        print('FAIL: placeholder image detected (small PNG). Configure image source (IMAGE_CDN_BASE/PHOTO_ROOT).')
        sys.exit(1)

    print('PASS: server returned a non-placeholder image')
    sys.exit(0)

if __name__ == '__main__':
    main()
