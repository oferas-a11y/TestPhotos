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
    candidates = [
        Path(image_path),
        Path(project_root) / image_path,
        Path(project_root) / "photos" / image_path,
        Path(project_root) / Path(image_path).name,
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
        
        # Get initial results from Pinecone (30 for Gemini reranking)
        initial_k = 30 if use_gemini else max_results
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
            formatted_result = {
                "id": photo_id,
                "score": result.get('score', 0.0),
                "description": result.get('document', ''),
                "metadata": result.get('metadata', {}),
                # Build absolute URLs for client convenience
                "image_url": f"{request.host_url.rstrip('/')}/api/photo/{photo_id}/image",
                "thumbnail_url": f"{request.host_url.rstrip('/')}/api/photo/{photo_id}/thumbnail"
            }
            
            # Add image path information
            metadata = result.get('metadata', {})
            if 'original_path' in metadata:
                formatted_result['image_path'] = metadata['original_path']
            
            formatted_results.append(formatted_result)
        
        # Apply Gemini reranking if requested and available
        reranked = False
        if use_gemini and len(formatted_results) > 1 and gemini_reranker is not None:
            try:
                print("🤖 Applying Gemini AI reranking...")
                
                # Clean results for Gemini (remove scores)
                clean_results = []
                for result in formatted_results:
                    clean_result = {
                        'id': result['id'],
                        'description': result['description'],
                        'metadata': result['metadata']
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
            # Fallback: return placeholder image so clients can render something
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
    
    print("🚀 Starting Historical Photos API Server...")
    print(f"📍 Port: {port}")
    print(f"🔧 Debug: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
