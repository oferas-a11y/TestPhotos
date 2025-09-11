#!/usr/bin/env python3
"""
Railway Dashboard CLI — query your deployed API and preview results.

Features
- Text UI that searches `/api/search/semantic` on your Railway app
- Flags results that return a placeholder thumbnail vs a real image
- Optionally opens a result in your browser

Usage
  # Read BASE_URL from .env or pass --base
  BASE_URL=https://<your-railway> python scripts/railway_dashboard_cli.py --query "love" --limit 15

  # Or explicitly
  python scripts/railway_dashboard_cli.py --base https://<your-railway> -q "love" -l 15 --open 1

Exit codes
  0 = success
  1 = network/HTTP error
  2 = config error (missing BASE_URL)
  3 = no results
"""

from __future__ import annotations

import os
import sys
import json
import textwrap
import argparse
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def http_json(url: str, timeout: int = 45) -> Dict[str, Any]:
    req = Request(url, headers={"User-Agent": "curl/8"})
    with urlopen(req, timeout=timeout) as r:
        data = r.read()
    return json.loads(data.decode("utf-8", errors="replace"))


def fetch_head_or_thumb(url: str, timeout: int = 30) -> Dict[str, Any]:
    """Fetch thumbnail and return basic info (content-type, bytes)."""
    req = Request(url, headers={"User-Agent": "curl/8"})
    with urlopen(req, timeout=timeout) as r:
        data = r.read()
        ctype = r.headers.get("Content-Type", "")
    return {"content_type": ctype, "bytes": len(data)}


def is_placeholder_thumb(info: Dict[str, Any]) -> bool:
    # Heuristic: generated placeholder is a small PNG (< ~5KB)
    ctype = (info.get("content_type") or "").lower()
    size = int(info.get("bytes") or 0)
    return ctype.startswith("image/png") and size < 5000


def truncate(s: str, n: int = 120) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def search(base: str, query: str, limit: int = 15, gemini: bool = False) -> List[Dict[str, Any]]:
    params = urlencode({"q": query, "limit": str(limit), "gemini": str(gemini).lower()})
    url = f"{base.rstrip('/')}/api/search/semantic?{params}"
    data = http_json(url)
    return data.get("results", [])


def main() -> int:
    ap = argparse.ArgumentParser(description="Railway Dashboard CLI")
    ap.add_argument("--base", help="Railway base URL (e.g., https://app.up.railway.app)")
    ap.add_argument("-q", "--query", required=True, help="Search query text")
    ap.add_argument("-l", "--limit", type=int, default=15, help="Max results (1-50)")
    ap.add_argument("--gemini", action="store_true", help="Use Gemini reranking if enabled on server")
    ap.add_argument("--open", type=int, default=0, metavar="N", help="Open result N in browser (1-based)")
    args = ap.parse_args()

    base = args.base or os.getenv("BASE_URL")
    if not base:
        print("Missing BASE_URL. Set env or pass --base.")
        return 2

    try:
        results = search(base, args.query, args.limit, args.gemini)
    except Exception as e:
        print(f"HTTP error: {e}")
        return 1

    if not results:
        print("No results.")
        return 3

    print(f"Results for: {args.query!r} (base={base})\n")
    rows: List[Dict[str, Any]] = []
    for i, r in enumerate(results, 1):
        pid = r.get("id")
        desc = r.get("description") or r.get("document") or ""
        thumb = r.get("thumbnail_url")
        image = r.get("image_url")
        # Make absolute URLs if needed
        if thumb and thumb.startswith("/"):
            thumb = f"{base.rstrip('/')}{thumb}"
        if image and image.startswith("/"):
            image = f"{base.rstrip('/')}{image}"

        # Probe thumbnail
        info = {"content_type": "", "bytes": 0}
        try:
            info = fetch_head_or_thumb(thumb)
        except Exception as e:
            info = {"error": str(e), "content_type": "", "bytes": 0}

        placeholder = is_placeholder_thumb(info)
        status = "PLACEHOLDER" if placeholder else "REAL"
        rows.append({
            "n": i,
            "id": pid,
            "thumb": thumb,
            "image": image,
            "status": status,
            "ctype": info.get("content_type"),
            "bytes": info.get("bytes"),
            "desc": truncate(desc, 120),
        })

    # Print concise dashboard
    for row in rows:
        print(f"{row['n']:02d}. {row['id']} [{row['status']}] {row['ctype']} {row['bytes']}B")
        print(f"    thumb: {row['thumb']}")
        print(f"    image: {row['image']}")
        if row['desc']:
            print(f"    {row['desc']}")

    # Optional open in browser
    if args.open:
        idx = args.open - 1
        if 0 <= idx < len(rows):
            try:
                import webbrowser
                webbrowser.open(rows[idx]["image"])  # open full image
                print(f"Opened: {rows[idx]['image']}")
            except Exception as e:
                print(f"Open failed: {e}")

    print("\nTip: If you see PLACEHOLDER, set IMAGE_CDN_BASE/IMAGE_PATH_MODE on the server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

