#!/usr/bin/env python3
"""
Flask API Server for Historical Photos Search
Deployable on Render with Pinecone cloud database
"""

import sys
import os
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64

from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
import numpy as np
import mimetypes
from PIL import Image, ImageDraw, ImageFont
import io
import socket
from urllib.parse import quote, urlencode
from urllib.request import urlopen, Request

# Add project paths
project_root = str(Path(__file__).parent)
main_app_path = str(Path(__file__).parent / "main_app")
sys.path.insert(0, project_root)
sys.path.insert(1, main_app_path)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import modules
try:
    from modules.pinecone_handler import create_pinecone_handler
    from dashboard_pipeline import PineconeSearch, CategorySearch, DataLoader, GemmaReranker
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Serve local demo page
@app.route('/')
@app.route('/demo')
def serve_demo():
    try:
        demo_path = Path(project_root) / 'static' / 'demo.html'
        if demo_path.exists():
            return send_file(demo_path, mimetype='text/html')
        return "<h1>Demo page missing</h1>", 404
    except Exception as e:
        return f"<h1>Error</h1><pre>{e}</pre>", 500

# Global handlers (initialized on first request)
pinecone_handler = None
pinecone_search = None
category_search = None
text_model = None
gemini_reranker = None

def initialize_handlers():
    """Initialize all handlers and models."""
    global pinecone_handler, pinecone_search, category_search, text_model, gemini_reranker
    
    if pinecone_handler is None:
        print("🔌 Initializing Pinecone connection...")
        pinecone_handler = create_pinecone_handler()
        if not pinecone_handler:
            raise RuntimeError("Failed to initialize Pinecone handler")
    
    if pinecone_search is None:
        print("🔍 Initializing Pinecone search...")
        data_loader = DataLoader()  # Mock data loader for API
        pinecone_search = PineconeSearch(data_loader)
    
    if category_search is None:
        print("🏷️ Initializing category search...")
        data_loader = DataLoader()
        category_search = CategorySearch(data_loader)
    
    if text_model is None:
        print("🔤 Loading text embedding model...")
        text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    if gemini_reranker is None:
        try:
            print("🤖 Initializing Gemini reranker...")
            # Check if GEMINI_API_KEY is available
            gemini_key = os.getenv('GEMINI_API_KEY')
            if gemini_key:
                print(f"   ✅ GEMINI_API_KEY found (ending: ...{gemini_key[-4:]})")
                gemini_reranker = GemmaReranker()
                print("   ✅ Gemini reranker initialized successfully")
            else:
                print("   ⚠️  GEMINI_API_KEY not found - Gemini reranking disabled")
                gemini_reranker = None
        except Exception as e:
            print(f"   ❌ Gemini initialization failed: {e}")
            print("   🔄 Continuing without Gemini reranking")
            gemini_reranker = None

def _is_lfs_pointer(file_path: Path) -> bool:
    """Check if a file is a Git LFS pointer file."""
    try:
        with open(file_path, 'rb') as f:
            head = f.read(200)
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False

def _find_image_file(image_path: str) -> Optional[Path]:
    """Try to resolve an actual image file for a given metadata path.

    Skips Git LFS pointer files and searches common locations.
    """
    # Optional external root for production (e.g., Railway volume)
    photo_root = os.getenv('PHOTO_ROOT')
    root_candidates = []
    if photo_root:
        pr = Path(photo_root)
        # Try full relative path under PHOTO_ROOT
        root_candidates.append(pr / image_path)
        # Try basename only under PHOTO_ROOT
        root_candidates.append(pr / Path(image_path).name)

    upload_root = Path(project_root) / "uploaded_photos"
    candidates = [
        *root_candidates,
        Path(image_path),
        Path(project_root) / image_path,
        Path(project_root) / "photos" / image_path,
        Path(project_root) / Path(image_path).name,
        upload_root / image_path,
        upload_root / Path(image_path).name,
    ]

    # Prefer any non-LFS actual file among direct candidates
    for p in candidates:
        if p.exists() and p.is_file() and not _is_lfs_pointer(p):
            return p

    # As a last resort, try to find by basename under photo_collections
    try:
        base = Path(image_path).name
        root = Path(project_root) / "photo_collections"
        if root.exists():
            for p in root.rglob(base):
                if p.is_file() and not _is_lfs_pointer(p):
                    return p
    except Exception:
        pass

    return None

def _placeholder_image(text: str, size=(300, 300)) -> io.BytesIO:
    """Generate a simple placeholder image with text."""
    img = Image.new('RGB', size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Attempt to load a default font; fall back to basic if unavailable
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    msg = text
    # Wrap text manually for small images
    lines = []
    words = msg.split()
    line = ''
    for w in words:
        test = f"{line} {w}".strip()
        if font and draw.textlength(test, font=font) > size[0] - 20:
            lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
    y = (size[1] - (len(lines) * 14)) // 2
    for ln in lines:
        w = draw.textlength(ln, font=font) if font else len(ln) * 6
        x = (size[0] - int(w)) // 2
        draw.text((x, y), ln, fill=(80, 80, 80), font=font)
        y += 16
    bio = io.BytesIO()
    img.save(bio, format='PNG')
    bio.seek(0)
    return bio

def _candidate_remote_urls(image_path: str, is_thumb: bool = False) -> List[str]:
    """Build candidate remote URLs for an image based on env config.

    Supported env vars:
    - IMAGE_URL_TEMPLATE: e.g., "https://cdn.example.com/photos/{basename}"
    - IMAGE_CDN_BASE: e.g., "https://cdn.example.com/photos" (joined with basename or path)
    - THUMB_URL_TEMPLATE / THUMB_CDN_BASE: overrides for thumbnails
    - IMAGE_PATH_MODE: "basename" (default) or "fullpath"
    """
    urls: List[str] = []
    basename = Path(image_path).name
    rel_path = str(image_path)
    path_mode = os.getenv('IMAGE_PATH_MODE', 'basename').lower()

    if is_thumb:
        tpl = os.getenv('THUMB_URL_TEMPLATE')
        base = os.getenv('THUMB_CDN_BASE')
    else:
        tpl = os.getenv('IMAGE_URL_TEMPLATE')
        base = os.getenv('IMAGE_CDN_BASE')

    # Template takes precedence
    if tpl:
        try:
            urls.append(tpl.format(basename=quote(basename), path=quote(rel_path)))
        except Exception:
            pass

    # Base URL join
    if base:
        base = base.rstrip('/')
        if path_mode == 'fullpath':
            urls.append(f"{base}/{quote(rel_path)}")
        # Always also try basename for robustness
        urls.append(f"{base}/{quote(basename)}")

    return [u for u in urls if u]

def _fetch_remote_bytes(url: str, timeout: int = 15) -> Optional[bytes]:
    """Fetch remote URL bytes using urllib. Returns None on failure."""
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None

def _fetch_remote_json(url: str, timeout: int = 20) -> Optional[Dict[str, Any]]:
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode('utf-8', errors='replace')
        return json.loads(data)
    except Exception:
        return None

@app.route('/api/admin/seed_one', methods=['POST', 'GET'])
def admin_seed_one():
    """Seed a single Pinecone record pointing to a local image file on the server.

    Security: requires `ADMIN_TOKEN` env var and matching `token` query/body param.

    Params:
    - token: must match ADMIN_TOKEN
    - path: relative path to image file (e.g., photo_collections/..../file.jpg)
    - text (optional): description text; defaults to filename
    - id (optional): custom id; defaults to generated
    """
    try:
        admin_token = os.getenv('ADMIN_TOKEN')
        provided = request.values.get('token')
        if admin_token and provided != admin_token:
            return jsonify({"error": "Unauthorized"}), 401

        initialize_handlers()

        rel_path = request.values.get('path', '').strip()
        if not rel_path:
            return jsonify({"error": "Missing 'path' parameter"}), 400

        img_path = Path(rel_path)
        if not img_path.is_absolute():
            img_path = Path(project_root) / img_path

        if not img_path.exists() or not img_path.is_file() or _is_lfs_pointer(img_path):
            return jsonify({"error": f"Image not available on server: {rel_path}"}), 400

        text = request.values.get('text', '')
        if not text:
            text = Path(rel_path).stem.replace('_', ' ').replace('-', ' ')

        custom_id = request.values.get('id')
        if not custom_id:
            # Stable-ish id from path
            custom_id = f"photo_{abs(hash(rel_path)) % (10**10):010d}"

        # Encode with sentence-transformers (384D)
        vec = text_model.encode([text])[0].tolist()

        record = {
            'id': custom_id,
            'embedding': vec,
            'metadata': {
                'original_path': str(Path(rel_path)),
                'source': 'admin_seed_one'
            },
            'document': text
        }

        # Upsert
        pinecone_handler.store_batch_analysis([record])

        return jsonify({
            'seeded': True,
            'id': custom_id,
            'image_available': True,
            'image_url': f"/api/photo/{custom_id}/image",
            'thumbnail_url': f"/api/photo/{custom_id}/thumbnail",
            'original_path': str(Path(rel_path))
        })

    except Exception as e:
        print(f"❌ admin_seed_one error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to seed record: {str(e)}"}), 500

@app.route('/api/photos', methods=['GET'])
def list_photos():
    """Return a simple paginated list of photos, prioritizing entries with real images.

    Query params:
    - limit (int, default 15, max 50): number of items to return
    - offset (int, default 0): number of items to skip (best-effort)

    Note: This uses a neutral vector query to sample photos from the index.
    Results order is not guaranteed stable; intended for initial gallery loads.
    """
    try:
        # Lightweight init: only Pinecone handler for browsing
        global pinecone_handler
        if pinecone_handler is None:
            pinecone_handler = create_pinecone_handler()
            if pinecone_handler is None:
                return jsonify({"error": "Failed to initialize Pinecone handler"}), 500

        # Parse params
        limit = int(request.args.get('limit', 15))
        offset = int(request.args.get('offset', 0))
        if limit < 1 or limit > 50:
            return jsonify({"error": "Limit must be between 1 and 50"}), 400
        if offset < 0:
            return jsonify({"error": "Offset must be >= 0"}), 400

        # Use a dummy vector to fetch a pool of candidates
        # Fallback to 384 dims like other endpoints if dimension is unknown
        dim = getattr(pinecone_handler, 'dimension', None) or 384
        dummy_vector = [0.0] * dim

        # Fetch a generous pool, then prioritize those with real images
        pool_size = min(max(limit * 5, 50), 1000)
        raw_results = pinecone_handler.search_photos(dummy_vector, top_k=pool_size)

        if not raw_results:
            return jsonify({
                "results": [],
                "total": 0,
                "offset": offset,
                "limit": limit
            })

        # Split results into those with available image files and those without
        with_images = []
        without_images = []

        for r in raw_results:
            pid = r.get('id')
            md = r.get('metadata', {}) or {}
            img_path = md.get('original_path')
            has_real = False
            if img_path:
                try:
                    has_real = _find_image_file(img_path) is not None
                except Exception:
                    has_real = False

            formatted = {
                "id": pid,
                "score": r.get('score', 0.0),
                "description": r.get('document', ''),
                "metadata": md,
                "image_url": f"{request.host_url.rstrip('/')}/api/photo/{pid}/image",
                "thumbnail_url": f"{request.host_url.rstrip('/')}/api/photo/{pid}/thumbnail"
            }

            if has_real:
                with_images.append(formatted)
            else:
                without_images.append(formatted)

        # Apply offset and limit across the concatenated list, preferring real images first
        ordered = with_images + without_images
        sliced = ordered[offset:offset + limit]

        return jsonify({
            "results": sliced,
            "total": len(ordered),
            "offset": offset,
            "limit": limit
        })

    except Exception as e:
        print(f"❌ list_photos error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to list photos: {str(e)}"}), 500


@app.route('/debug/first')
def debug_first_photo():
    """Simple HTML page that shows one photo inline from the DB.

    Useful for quickly verifying that localhost serves real images without downloading.
    """
    try:
        global pinecone_handler
        if pinecone_handler is None:
            pinecone_handler = create_pinecone_handler()
            if pinecone_handler is None:
                return "<h1>Error</h1><p>Failed to init Pinecone handler.</p>", 500

        dim = getattr(pinecone_handler, 'dimension', None) or 384
        dummy_vector = [0.0] * dim
        raw = pinecone_handler.search_photos(dummy_vector, top_k=200)
        if not raw:
            return "<h1>No photos</h1><p>Pinecone index is empty.</p>", 200

        # Prefer a photo with a resolvable real image
        chosen = None
        for r in raw:
            md = r.get('metadata', {}) or {}
            p = md.get('original_path')
            if p and _find_image_file(p):
                chosen = r
                break
        if chosen is None:
            chosen = raw[0]

        pid = chosen.get('id')
        thumb = f"/api/photo/{pid}/thumbnail"
        full = f"/api/photo/{pid}/image"
        html = f"""
        <!doctype html>
        <html><head><meta charset='utf-8'><title>First Photo</title>
        <style>body{font-family:system-ui,Arial;margin:24px} img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:4px}</style>
        </head><body>
        <h1>First Photo</h1>
        <p>ID: {pid}</p>
        <p><a href='{full}' target='_blank'>Open Full Image</a></p>
        <img src='{thumb}' alt='thumbnail'/>
        </body></html>
        """
        return html
    except Exception as e:
        return f"<h1>Error</h1><pre>{str(e)}</pre>", 500


@app.route('/debug/gallery')
def debug_gallery():
    """Simple HTML gallery of the first N photos (real images prioritized).

    Params: limit (default 15)
    """
    try:
        global pinecone_handler
        if pinecone_handler is None:
            pinecone_handler = create_pinecone_handler()
            if pinecone_handler is None:
                return "<h1>Error</h1><p>Failed to init Pinecone handler.</p>", 500

        limit = int(request.args.get('limit', 15))
        limit = max(1, min(limit, 50))
        dim = getattr(pinecone_handler, 'dimension', None) or 384
        dummy_vector = [0.0] * dim

        pool = pinecone_handler.search_photos(dummy_vector, top_k=min(limit*5, 1000))
        if not pool:
            return "<h1>No photos</h1><p>Pinecone index is empty.</p>", 200

        with_images, without_images = [], []
        for r in pool:
            pid = r.get('id')
            md = r.get('metadata', {}) or {}
            p = md.get('original_path')
            target = with_images if (p and _find_image_file(p)) else without_images
            target.append(pid)

        ordered = (with_images + without_images)[:limit]

        items = []
        for pid in ordered:
            items.append(f"<a href='/api/photo/{pid}/image' target='_blank'><img src='/api/photo/{pid}/thumbnail' alt='{pid}'/></a>")

        grid = "\n".join(f"<div class='item'>{itm}</div>" for itm in items)
        html = f"""
        <!doctype html>
        <html><head><meta charset='utf-8'><title>Gallery</title>
        <style>
        body{{font-family:system-ui,Arial;margin:24px}}
        .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}}
        .item img{{width:100%;height:auto;display:block;border:1px solid #ddd;border-radius:4px}}
        </style>
        </head><body>
        <h1>Gallery (limit {limit})</h1>
        <div class='grid'>
        {grid}
        </div>
        </body></html>
        """
        return html
    except Exception as e:
        return f"<h1>Error</h1><pre>{str(e)}</pre>", 500


@app.route('/debug/remote')
def debug_remote_gallery():
    """Render a gallery in the local browser using results from a remote base URL.

    Query params:
    - base: remote base URL (defaults to BASE_URL env or local host)
    - q: search query (default: people)
    - limit: number of results (default: 15)
    - gemini: true/false (default: false)
    """
    try:
        base = request.args.get('base') or os.getenv('BASE_URL') or request.host_url.rstrip('/')
        query = request.args.get('q', 'people')
        limit = int(request.args.get('limit', 15))
        use_gemini = request.args.get('gemini', 'false')

        params = urlencode({'q': query, 'limit': str(limit), 'gemini': use_gemini})
        api_url = f"{base.rstrip('/')}/api/search/semantic?{params}"
        data = _fetch_remote_json(api_url)
        if not data or 'results' not in data or not data['results']:
            return f"<h1>No results</h1><p>Base: {base}<br/>Query: {query}</p>", 200

        def probe_thumb(url: str) -> tuple[str, int]:
            try:
                req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urlopen(req, timeout=10) as resp:
                    body = resp.read()
                    ctype = resp.headers.get('Content-Type', '')
                    return ctype, len(body)
            except Exception:
                return '', 0

        cards = []
        for r in data['results']:
            pid = r.get('id')
            thumb = r.get('thumbnail_url')
            image = r.get('image_url')
            if thumb and thumb.startswith('/'):
                thumb = f"{base.rstrip('/')}{thumb}"
            if image and image.startswith('/'):
                image = f"{base.rstrip('/')}{image}"
            ctype, nbytes = probe_thumb(thumb)
            placeholder = (ctype.startswith('image/png') and nbytes < 5000)
            badge = 'PLACEHOLDER' if placeholder else 'REAL'
            color = '#c0392b' if placeholder else '#27ae60'
            cards.append(f"""
            <div class='item'>
              <a href='{image}' target='_blank'><img src='{thumb}'/></a>
              <div class='meta'>
                <span class='badge' style='background:{color}'>{badge}</span>
                <small>{ctype or 'unknown'} · {nbytes}B</small>
                <div class='pid'>{pid}</div>
              </div>
            </div>
            """)

        html = f"""
        <!doctype html>
        <html><head><meta charset='utf-8'><title>Remote Gallery</title>
        <style>
        body{{font-family:system-ui,Arial;margin:24px}}
        .hdr small{{color:#666}}
        .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}}
        .item img{{width:100%;height:auto;display:block;border:1px solid #ddd;border-radius:4px;background:#f9f9f9}}
        .meta{{display:flex;gap:8px;align-items:center;margin-top:6px;justify-content:space-between}}
        .badge{{color:#fff;padding:2px 6px;border-radius:4px;font-size:.8em}}
        .pid{{font-size:.8em;color:#999;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:60%}}
        form{{margin-bottom:16px;display:flex;gap:8px;flex-wrap:wrap}}
        input[type=text]{{padding:6px 8px;min-width:220px}}
        button{{padding:6px 10px}}
        </style>
        </head><body>
        <div class='hdr'>
          <h1>Remote Gallery</h1>
          <small>Base: {base} · Query: {query} · Limit: {limit}</small>
        </div>
        <form method='get'>
          <input type='text' name='base' value='{base}' placeholder='Base URL' />
          <input type='text' name='q' value='{query}' placeholder='Query' />
          <input type='text' name='limit' value='{limit}' />
          <button type='submit'>Search</button>
        </form>
        <div class='grid'>
          {''.join(cards)}
        </div>
        </body></html>
        """
        return html
    except Exception as e:
        return f"<h1>Error</h1><pre>{str(e)}</pre>", 500


@app.route('/api/admin/upload', methods=['POST'])
def admin_upload_photo():
    """Upload one image to the server and index it for immediate viewing.

    Security: requires `ADMIN_TOKEN` env var and matching `token` form field.

    Form fields:
    - token: must match ADMIN_TOKEN
    - file: image file to upload
    - text (optional): description text for embedding
    - id (optional): custom photo id; autogenerated if missing
    - original_path (optional): logical path to store in metadata; defaults to uploaded filename
    """
    try:
        admin_token = os.getenv('ADMIN_TOKEN')
        provided = request.form.get('token')
        if admin_token and provided != admin_token:
            return jsonify({"error": "Unauthorized"}), 401

        if 'file' not in request.files:
            return jsonify({"error": "Missing file"}), 400
        file = request.files['file']
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400

        # Save file
        upload_dir = Path(os.getenv('UPLOAD_DIR', Path(project_root) / 'uploaded_photos'))
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file.filename).name
        dest = upload_dir / safe_name
        file.save(dest)

        # Build metadata and vector
        initialize_handlers()
        text = request.form.get('text') or Path(safe_name).stem.replace('_', ' ').replace('-', ' ')
        custom_id = request.form.get('id') or f"photo_{abs(hash(safe_name)) % (10**10):010d}"
        logical_path = request.form.get('original_path') or f"uploaded_photos/{safe_name}"

        vec = text_model.encode([text])[0].tolist()
        record = {
            'id': custom_id,
            'embedding': vec,
            'metadata': {
                'original_path': logical_path,
                'source': 'admin_upload'
            },
            'document': text
        }
        pinecone_handler.store_batch_analysis([record])

        return jsonify({
            'uploaded': True,
            'id': custom_id,
            'original_path': logical_path,
            'image_url': f"/api/photo/{custom_id}/image",
            'thumbnail_url': f"/api/photo/{custom_id}/thumbnail"
        })
    except Exception as e:
        print(f"❌ admin_upload error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Render."""
    return jsonify({
        "status": "healthy",
        "service": "historical-photos-api",
        "version": "1.0.0"
    })

@app.route('/api/status')
def get_status():
    """Get database and service status."""
    try:
        initialize_handlers()
        
        # Get Pinecone stats
        stats = pinecone_handler.get_collection_stats()
        
        return jsonify({
            "status": "operational",
            "pinecone": {
                "connected": True,
                "total_photos": stats.get('total_photos', 0),
                "index_name": stats.get('index_name', 'unknown'),
                "dimension": stats.get('dimension', 0)
            },
            "services": {
                "semantic_search": True,
                "category_search": True,
                "gemini_reranking": True
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "pinecone": {"connected": False}
        }), 500

@app.route('/api/search/semantic', methods=['GET'])
def semantic_search():
    """Perform semantic search using Pinecone and optional Gemini reranking."""
    try:
        initialize_handlers()
        
        # Get query parameters
        query = request.args.get('q', '').strip()
        max_results = int(request.args.get('limit', 15))
        use_gemini = request.args.get('gemini', 'true').lower() == 'true'
        
        # Validation
        if not query:
            return jsonify({"error": "Query parameter 'q' is required"}), 400
        
        if max_results < 1 or max_results > 50:
            return jsonify({"error": "Limit must be between 1 and 50"}), 400
        
        print(f"🔍 Semantic search: '{query}' (limit: {max_results}, gemini: {use_gemini})")
        
        # Create query embedding
        query_vector = text_model.encode([query])[0].tolist()
        
        # Get initial results from Pinecone
        # If Gemini reranking is enabled, fetch at least as many as requested,
        # and at least 30 to give the reranker a meaningful pool.
        initial_k = max_results if not use_gemini else max(30, max_results)
        results = pinecone_handler.search_photos(query_vector, initial_k)
        
        if not results:
            return jsonify({
                "query": query,
                "results": [],
                "total": 0,
                "reranked": False
            })
        
        # Format results
        formatted_results = []
        for result in results:
            photo_id = result.get('id')
            metadata = result.get('metadata', {}) or {}
            # Prefer text stored under metadata['document'] (seeded from data_text.csv)
            desc = metadata.get('document', '') or metadata.get('comprehensive_text', '')
            formatted_result = {
                "id": photo_id,
                "score": result.get('score', 0.0),
                "description": desc,
                "metadata": metadata,
                # Build absolute URLs for client convenience
                "image_url": f"{request.host_url.rstrip('/')}/api/photo/{photo_id}/image",
                "thumbnail_url": f"{request.host_url.rstrip('/')}/api/photo/{photo_id}/thumbnail"
            }

            # Add image path information
            if 'original_path' in metadata:
                formatted_result['image_path'] = metadata['original_path']

            formatted_results.append(formatted_result)
        
        # Apply Gemini reranking if requested and available
        reranked = False
        if use_gemini and len(formatted_results) > 1 and gemini_reranker is not None:
            try:
                print("🤖 Applying Gemini AI reranking...")
                
                # Clean results for Gemini (remove scores) and include caption/items
                clean_results = []
                for result in formatted_results:
                    md = result.get('metadata', {}) or {}
                    clean_result = {
                        'id': result['id'],
                        'description': result['description'],
                        'caption': md.get('llm_caption') or md.get('caption') or '',
                        'items': md.get('yolo_object_counts') or md.get('objects') or md.get('items') or '',
                        'metadata': md,
                    }
                    clean_results.append(clean_result)
                
                # Rerank with Gemini
                reranked_results = gemini_reranker.rerank_results(query, clean_results, max_results)
                if reranked_results:
                    formatted_results = reranked_results[:max_results]
                    reranked = True
                    print(f"✅ Gemini reranking completed: {len(formatted_results)} results")
                else:
                    formatted_results = formatted_results[:max_results]
            except Exception as e:
                print(f"⚠️ Gemini reranking failed: {e}")
                formatted_results = formatted_results[:max_results]
        else:
            formatted_results = formatted_results[:max_results]
            if use_gemini and gemini_reranker is None:
                print("⚠️ Gemini reranking requested but not available")
        
        return jsonify({
            "query": query,
            "results": formatted_results,
            "total": len(formatted_results),
            "reranked": reranked,
            "gemini_enabled": use_gemini
        })
        
    except Exception as e:
        print(f"❌ Semantic search error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

@app.route('/api/search/category', methods=['GET'])
def category_search_api():
    """Perform category-based search."""
    try:
        initialize_handlers()
        
        # Get query parameters
        category = request.args.get('category', '').strip().lower()
        max_results = int(request.args.get('limit', 20))
        
        # Available categories
        available_categories = {
            "nazi_symbols": "Photos with Nazi symbols",
            "jewish_symbols": "Photos with Jewish symbols", 
            "hebrew_text": "Photos with Hebrew text",
            "german_text": "Photos with German text",
            "violence": "Photos with signs of violence",
            "indoor": "Indoor photos",
            "outdoor": "Outdoor photos"
        }
        
        if not category:
            return jsonify({
                "error": "Category parameter is required",
                "available_categories": available_categories
            }), 400
        
        if category not in available_categories:
            return jsonify({
                "error": f"Invalid category: {category}",
                "available_categories": list(available_categories.keys())
            }), 400
        
        print(f"🏷️ Category search: '{category}' (limit: {max_results})")
        
        # For now, return a placeholder response since category search requires CSV data
        # In production, you'd implement this with your data source
        return jsonify({
            "category": category,
            "category_description": available_categories[category],
            "results": [],
            "total": 0,
            "message": "Category search requires CSV data integration - implement based on your data structure"
        })
        
    except Exception as e:
        print(f"❌ Category search error: {e}")
        return jsonify({"error": f"Category search failed: {str(e)}"}), 500

@app.route('/api/categories')
def get_categories():
    """Get available search categories."""
    categories = {
        "nazi_symbols": "Photos with Nazi symbols",
        "jewish_symbols": "Photos with Jewish symbols",
        "hebrew_text": "Photos with Hebrew text", 
        "german_text": "Photos with German text",
        "violence": "Photos with signs of violence",
        "indoor": "Indoor photos",
        "outdoor": "Outdoor photos"
    }
    
    return jsonify({
        "categories": categories,
        "total_categories": len(categories)
    })

@app.route('/api/photo/<photo_id>')
def get_photo(photo_id):
    """Get detailed information about a specific photo."""
    try:
        initialize_handlers()
        
        # Search for the photo in Pinecone to get metadata
        # This is a simple implementation - in production you'd have a more efficient lookup
        dummy_vector = [0.0] * 384  # MiniLM-L6-v2 dimension
        results = pinecone_handler.search_photos(dummy_vector, top_k=1000)
        
        # Find the specific photo
        photo_info = None
        for result in results:
            if result.get('id') == photo_id:
                photo_info = result
                break
        
        if not photo_info:
            return jsonify({"error": f"Photo {photo_id} not found"}), 404
        
        # Format response with image URL
        metadata = photo_info.get('metadata', {})
        response = {
            "id": photo_id,
            "description": photo_info.get('document', ''),
            "metadata": metadata,
            "image_url": f"/api/photo/{photo_id}/image",
            "thumbnail_url": f"/api/photo/{photo_id}/thumbnail"
        }
        
        if 'original_path' in metadata:
            response['original_path'] = metadata['original_path']
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Failed to get photo: {str(e)}"}), 500

@app.route('/api/photo/<photo_id>/image')
def get_photo_image(photo_id):
    """Serve the actual photo image."""
    try:
        initialize_handlers()
        
        # Get photo metadata to find the image path
        dummy_vector = [0.0] * 384
        results = pinecone_handler.search_photos(dummy_vector, top_k=1000)
        
        photo_info = None
        for result in results:
            if result.get('id') == photo_id:
                photo_info = result
                break
        
        if not photo_info:
            abort(404)
        
        # Get image path from metadata
        metadata = photo_info.get('metadata', {})
        image_path = metadata.get('original_path')
        
        if not image_path:
            abort(404)
        
        # Resolve actual image file (skip Git LFS pointer files)
        image_file = _find_image_file(image_path)
        if not image_file:
            # Try remote sources via CDN/URL templates
            for url in _candidate_remote_urls(image_path, is_thumb=False):
                data = _fetch_remote_bytes(url)
                if data:
                    bio = io.BytesIO(data)
                    # Guess type from URL extension
                    mime_type, _ = mimetypes.guess_type(url)
                    return send_file(bio, mimetype=mime_type or 'image/jpeg')
            # Fallback: return placeholder image
            placeholder = _placeholder_image("Image unavailable", size=(600, 600))
            return send_file(placeholder, mimetype='image/png')

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(image_file))
        if not mime_type:
            mime_type = 'image/jpeg'

        return send_file(image_file, mimetype=mime_type)
        
    except Exception as e:
        return jsonify({"error": f"Failed to serve image: {str(e)}"}), 500

@app.route('/api/photo/<photo_id>/thumbnail')
def get_photo_thumbnail(photo_id):
    """Serve a thumbnail version of the photo."""
    try:
        initialize_handlers()
        
        # Get photo metadata
        dummy_vector = [0.0] * 384
        results = pinecone_handler.search_photos(dummy_vector, top_k=1000)
        
        photo_info = None
        for result in results:
            if result.get('id') == photo_id:
                photo_info = result
                break
        
        if not photo_info:
            abort(404)
        
        metadata = photo_info.get('metadata', {})
        image_path = metadata.get('original_path')
        
        if not image_path:
            abort(404)
        
        # Find actual image file (skip LFS pointer files)
        image_file = _find_image_file(image_path)
        if not image_file:
            # Try remote sources
            for url in _candidate_remote_urls(image_path, is_thumb=True):
                data = _fetch_remote_bytes(url)
                if data:
                    try:
                        with Image.open(io.BytesIO(data)) as img:
                            img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                            img_io = io.BytesIO()
                            fmt = 'JPEG' if img.mode != 'RGBA' else 'PNG'
                            img.save(img_io, format=fmt, quality=85)
                            img_io.seek(0)
                            return send_file(img_io, mimetype='image/jpeg' if fmt == 'JPEG' else 'image/png')
                    except Exception:
                        pass
            # Return a thumbnail-sized placeholder
            placeholder = _placeholder_image("Image unavailable", size=(300, 300))
            return send_file(placeholder, mimetype='image/png')
        
        # Create thumbnail
        try:
            with Image.open(image_file) as img:
                # Create thumbnail (max 300x300)
                img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                
                # Save to memory
                img_io = io.BytesIO()
                format = 'JPEG' if img.mode != 'RGBA' else 'PNG'
                img.save(img_io, format=format, quality=85)
                img_io.seek(0)
                
                mime_type = 'image/jpeg' if format == 'JPEG' else 'image/png'
                
                return send_file(img_io, mimetype=mime_type)
        
        except Exception as e:
            # Fallback: serve original image if thumbnail creation fails
            print(f"Thumbnail creation failed: {e}, serving original")
            mime_type, _ = mimetypes.guess_type(str(image_file))
            if not mime_type:
                mime_type = 'image/jpeg'
            return send_file(image_file, mimetype=mime_type)
        
    except Exception as e:
        return jsonify({"error": f"Failed to serve thumbnail: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    def _choose_free_port(preferred: int, max_tries: int = 10) -> int:
        p = preferred
        for _ in range(max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(('0.0.0.0', p))
                    return p
                except OSError:
                    p += 1
        return preferred

    chosen = _choose_free_port(port)
    if chosen != port:
        print(f"⚠️  Port {port} busy; switching to {chosen}")
    
    print("🚀 Starting Historical Photos API Server...")
    print(f"📍 Port: {chosen}")
    print(f"🔧 Debug: {debug}")
    
    app.run(host='0.0.0.0', port=chosen, debug=debug)
