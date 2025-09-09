"""Dashboard search pipeline (category + semantic search)."""

import base64
import csv
import json
import os
import re
import sys
import unicodedata
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore

# Load environment variables from .env file in project root


def load_env():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


load_env()

# Add project root to path for importing modules
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import after path is set
try:
    # Import similarity search from local modules directory
    from modules.similarity_search import (  # type: ignore
        get_embeddings_status,
        get_visual_similarities,
        get_combined_similarities
    )

    # Try to import gemma_reranker from project root modules
    try:
        from modules.gemma_reranker import GemmaReranker  # type: ignore
    except ImportError:
        # If not found in project root, check current directory
        project_root_alt = os.path.dirname(os.path.dirname(__file__))
        if os.path.exists(os.path.join(project_root_alt, 'modules', 'gemma_reranker.py')):
            sys.path.insert(0, project_root_alt)
            from modules.gemma_reranker import GemmaReranker  # type: ignore
        else:
            raise ImportError("gemma_reranker not found in any location")

    GEMINI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Gemini reranking unavailable: {e}")
    GEMINI_AVAILABLE = False

    # Create dummy class to avoid import errors
    class GemmaReranker:
        """Dummy GemmaReranker class for when module is unavailable."""

        def __init__(self):
            """Initialize dummy reranker."""
            pass

        def rerank_results(self, query, results, top_k=10):
            """Return original results without reranking."""
            return results[:top_k]

# Import ChromaDB handler
try:
    from modules.chroma_handler import (  # type: ignore
        ChromaPhotoHandler,
        is_chromadb_available,
        create_chroma_handler
    )
    CHROMADB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ChromaDB unavailable: {e}")
    CHROMADB_AVAILABLE = False

    # Create dummy handler to avoid import errors
    class ChromaPhotoHandler:
        """Dummy ChromaPhotoHandler class for when module is unavailable."""

        def __init__(self, *args, **kwargs):
            """Initialize dummy handler."""

        def search_photos(self, *args, **kwargs):
            """Return empty search results."""
            return {
                'documents': [[]],
                'metadatas': [[]],
                'ids': [[]],
                'distances': [[]]
            }

        def search_by_category(self, *args, **kwargs):
            """Return empty category results."""
            return {
                'documents': [[]],
                'metadatas': [[]],
                'ids': [[]],
                'distances': [[]]
            }

        def get_collection_stats(self):
            """Return empty stats."""
            return {'total_photos': 0, 'error': 'ChromaDB not available'}

    def is_chromadb_available():
        """Return False for dummy implementation."""
        return False

    def create_chroma_handler(*args, **kwargs):
        """Return None for dummy implementation."""
        return None

# Import Pinecone handler
try:
    from modules.pinecone_handler import (  # type: ignore
        PineconePhotoHandler,
        create_pinecone_handler
    )
    PINECONE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Pinecone unavailable: {e}")
    PINECONE_AVAILABLE = False

    # Create dummy handler to avoid import errors
    class PineconePhotoHandler:
        """Dummy PineconePhotoHandler class for when module is unavailable."""

        def __init__(self, *args, **kwargs):
            """Initialize dummy handler."""

        def search_photos(self, *args, **kwargs):
            """Return empty search results."""
            return []

        def get_collection_stats(self):
            """Return empty stats."""
            return {'total_photos': 0, 'error': 'Pinecone not available'}

    def create_pinecone_handler(*args, **kwargs):
        """Return None for dummy implementation."""
        return None


class DataLoader:
    """Loads and manages CSV data for searching."""

    def __init__(self, results_dir: Optional[Path] = None):
        if results_dir is None:
            results_dir = Path("main_app_outputs") / "results"

        self.results_dir = results_dir
        self.data_full = results_dir / "data_full.csv"
        self.data_text = results_dir / "data_text.csv"
        self.output_dir = Path("main_app_outputs") / "data_search_dashord"

    def load_csv(self, path: Path) -> List[Dict[str, str]]:
        """Load CSV file into list of dictionaries."""
        rows: List[Dict[str, str]] = []
        if not path.exists():
            return rows

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: (v or "").strip() for k, v in row.items()})

        return rows

    def build_index(self) -> Dict[str, Dict[str, str]]:
        """Build unified index from both CSV files."""
        full_rows = self.load_csv(self.data_full)
        text_rows = self.load_csv(self.data_text)

        # Index by original_path
        idx: Dict[str, Dict[str, str]] = {}

        for r in full_rows:
            idx[r.get("original_path", "")] = r

        for r in text_rows:
            k = r.get("original_path", "")
            comprehensive_text = r.get("comprehensive_text", "") or r.get("description", "")
            if k in idx:
                idx[k]["text_description"] = comprehensive_text
            else:
                idx[k] = {"text_description": comprehensive_text}

        return idx


class CategorySearch:
    """Handles category-based filtering and search."""

    CATEGORIES = {
        "nazi_symbols": "Photos with Nazi symbols",
        "jewish_symbols": "Photos with Jewish symbols",
        "hebrew_text": "Photos with Hebrew text",
        "german_text": "Photos with German text",
        "violence": "Photos with signs of violence",
        "indoor": "Indoor photos",
        "outdoor": "Outdoor photos"
    }

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def compute_category_counts(self, idx: Dict[str, Dict[str, str]]) -> Dict[str, int]:
        """Compute counts for each category."""
        counts = {category: 0 for category in self.CATEGORIES}

        for _, row in idx.items():
            if (row.get("llm_nazi_symbols") or "").strip():
                counts["nazi_symbols"] += 1
            if (row.get("llm_jewish_symbols") or "").strip():
                counts["jewish_symbols"] += 1
            if row.get("hebrew_present", "").lower() == "true":
                counts["hebrew_text"] += 1
            if row.get("german_present", "").lower() == "true":
                counts["german_text"] += 1
            if row.get("violence", "").lower() == "true":
                counts["violence"] += 1

            io = (row.get("indoor_outdoor") or "").lower()
            if io == "indoor":
                counts["indoor"] += 1
            elif io == "outdoor":
                counts["outdoor"] += 1

        return counts

    def filter_rows(self, idx: Dict[str, Dict[str, str]], category: str) -> List[Dict[str, str]]:
        """Filter rows by category."""
        rows: List[Dict[str, str]] = []

        for _, r in idx.items():
            if self._matches_category(r, category):
                rows.append(r)

        return rows

    def _matches_category(self, row: Dict[str, str], category: str) -> bool:
        """Check if row matches the given category."""
        if category == "nazi_symbols":
            return bool((row.get("llm_nazi_symbols") or "").strip())
        elif category == "jewish_symbols":
            return bool((row.get("llm_jewish_symbols") or "").strip())
        elif category == "hebrew_text":
            return row.get("hebrew_present", "").lower() == "true"
        elif category == "german_text":
            return row.get("german_present", "").lower() == "true"
        elif category == "violence":
            return row.get("violence", "").lower() == "true"
        elif category == "indoor":
            return (row.get("indoor_outdoor") or "").lower() == "indoor"
        elif category == "outdoor":
            return (row.get("indoor_outdoor") or "").lower() == "outdoor"

        return False

    def select_category(self) -> str:
        """Interactive category selection."""
        print("\nSearch by categories (type a number):")
        categories = list(self.CATEGORIES.keys())

        for i, (key, title) in enumerate(self.CATEGORIES.items(), 1):
            print(f"{i}) {title}")

        choice = input("Enter 1-7: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(categories):
                return categories[idx]
        except ValueError:
            pass

        return ""

    def write_report(self, category: str, rows: List[Dict[str, str]]) -> Path:
        """Write category search report."""
        self.data_loader.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.data_loader.output_dir / f"search_{category}_{ts}.txt"

        lines: List[str] = []
        title = self.CATEGORIES.get(category, category)
        lines.append(title)
        lines.append(f"Total: {len(rows)}")
        lines.append("")

        for r in rows:
            self._add_row_details(lines, r)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return out_path

    def write_gallery(self, category: str, rows: List[Dict[str, str]]) -> Path:
        """Write category search HTML gallery."""
        self.data_loader.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.data_loader.output_dir / f"category_{category}_{ts}.html"

        def esc(t: str) -> str:
            return (t or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        def get_image_url(row: Dict[str, str]) -> str:
            """Get base64 data URL for original image from CSV path."""
            orig_path = row.get('original_path', '')
            if not orig_path:
                return ""

            # Get absolute path from CSV relative path
            project_root = self.data_loader.output_dir.parent.parent
            source_path = project_root / orig_path

            if not source_path.exists():
                return ""

            # Read image and convert to base64 data URL
            try:
                with open(source_path, 'rb') as f:
                    image_data = f.read()
                b64_data = base64.b64encode(image_data).decode()
                ext = source_path.suffix.lower()
                mime = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
                return f"data:{mime};base64,{b64_data}"
            except Exception:
                return ""

        parts: List[str] = []
        parts.append("<!DOCTYPE html><html><head><meta charset=\"utf-8\">")
        title = self.CATEGORIES.get(category, category)
        parts.append(f"<title>Category Search: {esc(title)}</title>")
        parts.append("<style>")
        parts.append("body{font-family:Arial,Helvetica,sans-serif;margin:20px}")
        parts.append(".item{margin-bottom:30px;display:flex;gap:16px;align-items:flex-start}")
        parts.append(".image-container{flex-shrink:0}")
        parts.append("img{max-width:360px;height:auto;border:1px solid #ddd;border-radius:4px}")
        parts.append("img.missing{background:#f0f0f0;padding:20px;text-align:center;color:#666}")
        parts.append(".meta{max-width:800px;padding-left:10px}")
        parts.append(".path{font-size:0.9em;color:#666;margin-bottom:10px}")
        parts.append("</style>")
        parts.append("</head><body>")
        parts.append(f"<h2>Category: {esc(title)}</h2>")
        parts.append(f"<p>Total found: {len(rows)}</p>")

        for i, r in enumerate(rows, 1):
            orig_path = r.get('original_path', '')
            desc = r.get('text_description', '') or r.get('description', '')

            # Get original B&W image URL
            img_path = get_image_url(r)

            parts.append('<div class="item">')
            parts.append('<div class="image-container">')

            if img_path:
                filename = Path(orig_path).name if orig_path else f"Image {i}"
                parts.append(
                    f'<img src="{esc(img_path)}" alt="{esc(filename)}" '
                    f'onerror="this.style.display=\'none\'; '
                    f'this.nextElementSibling.style.display=\'flex\'">')
                parts.append(
                    '<div style="display:none;width:360px;height:240px;'
                    'background:#f0f0f0;border:1px solid #ddd;border-radius:4px;'
                    'align-items:center;justify-content:center;color:#666;'
                    'font-size:14px;flex-direction:column">'
                    'Original image not found</div>')
            else:
                parts.append(
                    '<div style="width:360px;height:240px;background:#f0f0f0;'
                    'border:1px solid #ddd;border-radius:4px;display:flex;'
                    'align-items:center;justify-content:center;color:#666;'
                    'font-size:14px">Original image not available</div>')

            parts.append("</div>")
            parts.append('<div class="meta">')

            # Show original filename prominently
            if orig_path:
                filename = Path(orig_path).name
                parts.append(
                    f'<div class="path"><strong>📸 {esc(filename)}</strong></div>')

            parts.append(f"<h3>{i}. Historical Photo Analysis</h3>")

            if desc:
                parts.append(f"<p>{esc(desc)}</p>")

            parts.append(
                '<p style="font-size:0.9em;color:#888;margin-top:15px;">'
                '<em>Showing original black & white historical photograph</em></p>')
            parts.append("</div>")
            parts.append("</div>")

        parts.append("</body></html>")

        with open(out, "w", encoding="utf-8") as f:
            f.write("".join(parts))

        return out

    def _add_row_details(self, lines: List[str], r: Dict[str, str]) -> None:
        """Add detailed information for a single row."""
        orig = r.get("original_path", "")
        desc = r.get("text_description", "")

        lines.append(f"Original: {orig}")

        if isinstance(desc, str) and desc.strip():
            lines.append(f"Description: {desc}")

        # Location info
        io = r.get("indoor_outdoor", "")
        bg = r.get("background", "") or r.get("background_top", "")
        if io or bg:
            location = io
            if bg:
                location += f" - {bg}" if location else bg
            lines.append(f"Location: {location}")

        # Objects
        oc = r.get("yolo_object_counts", "")
        if oc:
            lines.append(f"Objects: {oc}")

        # Caption
        cap = r.get("llm_caption", "")
        if cap:
            lines.append(f"Caption: {cap}")

        # Symbols
        if r.get("llm_jewish_symbols", ""):
            lines.append(f"Jewish: {r.get('llm_jewish_symbols')}")
        if r.get("llm_nazi_symbols", ""):
            lines.append(f"Nazi: {r.get('llm_nazi_symbols')}")

        # Texts
        if r.get("hebrew_present", "").lower() == "true":
            ht = r.get("hebrew_text", "")
            tr = r.get("hebrew_translation", "")
            show = " - ".join([s for s in [ht, tr] if s])
            if show:
                lines.append(f"Hebrew: {show}")

        if r.get("german_present", "").lower() == "true":
            gt = r.get("german_text", "")
            tr2 = r.get("german_translation", "")
            show2 = " - ".join([s for s in [gt, tr2] if s])
            if show2:
                lines.append(f"German: {show2}")

        lines.append("")


class SemanticSearch:
    """Handles semantic search using sentence transformers."""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.emb_npz = data_loader.output_dir / "embeddings_minilm_l6.npz"
        self.meta_json = data_loader.output_dir / "embeddings_meta.json"
        self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'

    def build_embeddings(self) -> None:
        """Build embeddings from text descriptions."""
        rows = self.data_loader.load_csv(self.data_loader.data_text)
        if not rows:
            print("No rows in data_text.csv. Run the main app first.")
            return

        self.data_loader.output_dir.mkdir(parents=True, exist_ok=True)
        model = SentenceTransformer(self.model_name)

        texts = [r.get('comprehensive_text', '') or r.get('description', '') for r in rows]
        embs = model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Save embeddings
        np.savez_compressed(self.emb_npz, embeddings=embs)

        # Save metadata
        meta = [{
            'original_path': r.get('original_path', ''),
            'colorized_path': r.get('colorized_path', ''),
            'full_results_path': r.get('full_results_path', ''),
            'llm_json_path': r.get('llm_json_path', ''),
            'description': r.get('comprehensive_text', '') or r.get('description', '')
        } for r in rows]

        with open(self.meta_json, 'w', encoding="utf-8") as f:
            json.dump({'rows': meta}, f, indent=2)

        print(f"Wrote embeddings: {self.emb_npz}")
        print(f"Wrote metadata: {self.meta_json}")

    def load_embeddings(self) -> Tuple[np.ndarray, List[Dict[str, str]]]:
        """Load embeddings and metadata."""
        if not self.emb_npz.exists() or not self.meta_json.exists():
            raise FileNotFoundError("Embeddings or metadata not found. Run build_embeddings() first.")

        data = np.load(self.emb_npz)
        embs = data['embeddings']

        with open(self.meta_json, 'r', encoding="utf-8") as f:
            meta_data = json.load(f)

        meta = meta_data.get('rows', [])
        return embs, meta

    def search(self, query: str, k: int = 5, threshold: float = 0.35,
               use_gemma_reranking: bool = True) -> Tuple[List[Dict[str, str]], int]:
        """Perform semantic search with optional Gemini reranking."""
        embs, meta = self.load_embeddings()
        model = SentenceTransformer(self.model_name)

        # Encode query
        q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        # Calculate similarities
        sims = embs @ q
        num_above = int((sims >= threshold).sum())

        # Get top 30 results for Gemini reranking (or requested k if less than 30)
        initial_k = (min(30, len(meta)) if use_gemma_reranking
                     else max(1, k))
        idxs = np.argsort(-sims)[:initial_k]
        initial_results = [meta[int(i)] for i in idxs]

        # Apply Gemini reranking if enabled and available
        if use_gemma_reranking and len(initial_results) > k and GEMINI_AVAILABLE:
            try:
                print("🤖 [SEMANTIC DEBUG] Enhancing results with Gemini AI reranking...")
                print(f"🔍 [SEMANTIC DEBUG] Initial MiniLM results: {len(initial_results)}")
                print(f"🔍 [SEMANTIC DEBUG] Requesting top {k} from Gemini")
                print(f"🔍 [SEMANTIC DEBUG] Query: '{query}'")

                reranker = GemmaReranker()
                reranked_results = reranker.rerank_results(query, initial_results, top_k=k)

                print(f"✅ [SEMANTIC DEBUG] Gemini reranking complete: "
                      f"{len(reranked_results)} results returned")

                # Show comparison between MiniLM and Gemini ranking
                print("📊 [SEMANTIC DEBUG] Ranking comparison:")
                for i in range(min(5, len(reranked_results))):
                    filename = reranked_results[i].get('original_path', '').split('/')[-1]
                    gemma_rank = reranked_results[i].get('gemma_rank', 'N/A')
                    print(f"  Final #{i + 1}: {filename} (was Gemini rank #{gemma_rank})")

                return reranked_results, num_above
            except ValueError as e:
                if "GEMINI_API_KEY" in str(e):
                    print("❌ [SEMANTIC DEBUG] Gemini reranking disabled: "
                          "Missing GEMINI_API_KEY in .env file")
                else:
                    print(f"❌ [SEMANTIC DEBUG] Gemini reranking error: {e}")
                print("📋 [SEMANTIC DEBUG] Continuing with MiniLM results only")
            except Exception as e:
                print(f"❌ [SEMANTIC DEBUG] Gemini reranking failed: {e}")
                print(f"❌ [SEMANTIC DEBUG] Exception type: {type(e).__name__}")
                import traceback
                print("❌ [SEMANTIC DEBUG] Full traceback:")
                traceback.print_exc()
                print("📋 [SEMANTIC DEBUG] Continuing with MiniLM results only")
        elif use_gemma_reranking and not GEMINI_AVAILABLE:
            print("⚠️  Gemini reranking module not available - using MiniLM results only")

        # Fallback to original MiniLM ranking
        results = initial_results[:k]
        return results, num_above

    def write_report(self, query: str, rows: List[Dict[str, str]]) -> Path:
        """Write semantic search text report."""
        self.data_loader.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.data_loader.output_dir / f"semantic_{ts}.txt"

        lines: List[str] = [
            f"Query: {query}",
            f"Total returned: {len(rows)}",
            ""
        ]

        for i, r in enumerate(rows, 1):
            orig = r.get('original_path', '')
            desc = r.get('description', '')
            lines.append(f"{i}. {orig}")
            if desc:
                lines.append(f"   {desc}")
            lines.append("")

        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def write_gallery(self, query: str, rows: List[Dict[str, str]]) -> Path:
        """Write semantic search HTML gallery."""
        self.data_loader.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.data_loader.output_dir / f"semantic_{ts}.html"

        def esc(t: str) -> str:
            return (t or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        def make_safe_filename(filename: str) -> str:
            """Create a web-safe filename from any string."""
            # Normalize unicode characters
            filename = unicodedata.normalize('NFKD', filename)
            # Remove non-ASCII characters
            filename = filename.encode('ascii', 'ignore').decode('ascii')
            # Replace spaces and special chars with underscores
            filename = re.sub(r'[^\w\.-]', '_', filename)
            # Remove multiple underscores
            filename = re.sub(r'_+', '_', filename)
            # Remove leading/trailing underscores
            filename = filename.strip('_')
            return filename or 'unnamed'

        def get_image_url(row: Dict[str, str]) -> str:
            """Get base64 data URL for original image from CSV path."""
            orig_path = row.get('original_path', '')
            if not orig_path:
                return ""

            # Get absolute path from CSV relative path
            project_root = self.data_loader.output_dir.parent.parent
            source_path = project_root / orig_path

            if not source_path.exists():
                return ""

            # Read image and convert to base64 data URL
            try:
                with open(source_path, 'rb') as f:
                    image_data = f.read()
                b64_data = base64.b64encode(image_data).decode()
                ext = source_path.suffix.lower()
                mime = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
                return f"data:{mime};base64,{b64_data}"
            except Exception:
                return ""

        parts: List[str] = []
        parts.append("<!DOCTYPE html><html><head><meta charset=\"utf-8\">")
        parts.append("<title>Semantic Search Results</title>")
        parts.append("<style>")
        parts.append("body{font-family:Arial,Helvetica,sans-serif;margin:20px}")
        parts.append(".item{margin-bottom:30px;display:flex;gap:16px;align-items:flex-start}")
        parts.append(".image-container{flex-shrink:0}")
        parts.append("img{max-width:360px;height:auto;border:1px solid #ddd;border-radius:4px}")
        parts.append("img.missing{background:#f0f0f0;padding:20px;text-align:center;color:#666}")
        parts.append(".meta{max-width:800px;padding-left:10px}")
        parts.append(".path{font-size:0.9em;color:#666;margin-bottom:10px}")
        parts.append(".similarity-section{margin-top:20px;border-top:1px solid #eee;"
                     "padding-top:15px}")
        parts.append(".similarity-row{display:flex;gap:10px;margin-bottom:15px;"
                     "align-items:flex-start}")
        parts.append(".similarity-img{width:80px;height:60px;object-fit:cover;border-radius:3px}")
        parts.append(".similarity-info{flex:1}")
        parts.append(".similarity-score{font-size:0.8em;color:#666;font-weight:bold}")
        parts.append("</style>")
        parts.append("</head><body>")
        parts.append(f"<h2>Query: {esc(query)}</h2>")
        parts.append(f"<p>Total returned: {len(rows)}</p>")

        for i, r in enumerate(rows, 1):
            orig_path = r.get('original_path', '')
            desc = r.get('description', '')

            # Get original B&W image URL
            img_path = get_image_url(r)

            parts.append('<div class="item">')
            parts.append('<div class="image-container">')

            if img_path:
                original_filename = Path(orig_path).name if orig_path else f"Image {i}"
                parts.append(f'<img src="{esc(img_path)}" alt="{esc(original_filename)}" '
                           f'onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'flex\'">')
                parts.append('<div style="display:none;width:360px;height:240px;background:#f0f0f0;'
                           'border:1px solid #ddd;border-radius:4px;align-items:center;'
                           'justify-content:center;color:#666;font-size:14px;flex-direction:column">Original image not found</div>')
            else:
                parts.append('<div style="width:360px;height:240px;background:#f0f0f0;'
                           'border:1px solid #ddd;border-radius:4px;display:flex;align-items:center;'
                           'justify-content:center;color:#666;font-size:14px">Original image not available</div>')

            parts.append("</div>")
            parts.append('<div class="meta">')

            # Show original filename prominently
            if orig_path:
                parts.append(f'<div class="path"><strong>📸 {esc(Path(orig_path).name)}</strong></div>')

            # Show title with Gemma ranking if available
            gemma_info = f" (Gemma Rank: #{r.get('gemma_rank')})" if r.get('gemma_rank') else ""
            parts.append(f"<h3>{i}. Historical Photo Analysis{esc(gemma_info)}</h3>")

            if desc:
                parts.append(f"<p>{esc(desc)}</p>")

            # Add ranking info
            if r.get('gemma_rank'):
                parts.append('<p style="font-size:0.85em;color:#0066cc;margin-top:10px;">'
                           f'<strong>🤖 AI Relevance Ranking: #{r.get("gemma_rank")} most relevant</strong></p>')

            # Add note about original B&W photos
            parts.append('<p style="font-size:0.9em;color:#888;margin-top:15px;">'
                        '<em>Showing original black & white historical photograph</em></p>')

            parts.append("</div>")
            parts.append("</div>")

        parts.append("</body></html>")
        out.write_text("\n".join(parts), encoding="utf-8")
        return out


class ChromaSearch:
    """ChromaDB-based search functionality."""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.chroma_handler = None
        self._initialize_chroma()

    def _initialize_chroma(self) -> None:
        """Initialize ChromaDB handler if available."""
        if not is_chromadb_available():
            print("⚠️  ChromaDB not available. Install with: pip install chromadb")
            return

        try:
            self.chroma_handler = create_chroma_handler()
            if self.chroma_handler:
                print("✅ ChromaDB search initialized")
            else:
                print("❌ Failed to initialize ChromaDB handler")
        except Exception as e:
            print(f"❌ ChromaDB initialization error: {e}")

    def is_available(self) -> bool:
        """Check if ChromaDB search is available."""
        return self.chroma_handler is not None

    def semantic_search(self, query: str, n_results: int = 10,
                        use_gemma_reranking: bool = True) -> List[Dict[str, Any]]:
        """Perform semantic search using ChromaDB with optional Gemini reranking."""
        if not self.chroma_handler:
            print("❌ ChromaDB not available for search")
            return []

        try:
            # Get more results for Gemini reranking (30) or requested amount
            initial_k = min(
                            30,
                            670) if use_gemma_reranking else n_results  # 670 is total photos in ChromaD)
            results = self.chroma_handler.search_photos(query, initial_k)
            initial_results = self._format_chroma_results(results)

            # Apply Gemini reranking if enabled and available
            if use_gemma_reranking and len(initial_results) > n_results and GEMINI_AVAILABLE:
                try:
                    print("🤖 [CHROMADB DEBUG] Enhancing ChromaDB results with Gemini AI reranking...")
                    print(f"🔍 [CHROMADB DEBUG] Initial ChromaDB results: {len(initial_results)}")
                    print(f"🔍 [CHROMADB DEBUG] Requesting top {n_results} from Gemini")
                    print(f"🔍 [CHROMADB DEBUG] Query: '{query}'")

                    # Remove similarity scores and ChromaDB-specific fields for Gemini
                    clean_results = []
                    for result in initial_results:
                        clean_result = result.copy()
                        # Remove fields that might indicate ranking/scoring
                        keys_to_remove = ['similarity_score', 'chromadb_result', 'id', 'metadata']
                        for key in keys_to_remove:
                            clean_result.pop(key, None)
                        clean_results.append(clean_result)

                    reranker = GemmaReranker()
                    reranked_results = reranker.rerank_results(query, clean_results,
                                                               top_k=n_results)

                    print(f"✅ [CHROMADB DEBUG] Gemini reranking complete: {len(reranked_results)} results returned")

                    # Show comparison between ChromaDB and Gemini ranking
                    print("📊 [CHROMADB DEBUG] Ranking comparison:")
                    for i in range(min(5, len(reranked_results))):
                        filename = reranked_results[i].get('original_path', '').split('/')[-1]
                        gemma_rank = reranked_results[i].get('gemma_rank', 'N/A')
                        print(f"  Final #{i + 1}: {filename} (was Gemini rank #{gemma_rank})")

                    return reranked_results

                except ValueError as e:
                    if "GEMINI_API_KEY" in str(e):
                        print("❌ [CHROMADB DEBUG] Gemini reranking disabled: Missing GEMINI_API_KEY in .env file")
                    else:
                        print(f"❌ [CHROMADB DEBUG] Gemini reranking error: {e}")
                    print("📋 [CHROMADB DEBUG] Continuing with ChromaDB results only")
                except Exception as e:
                    print(f"❌ [CHROMADB DEBUG] Gemini reranking failed: {e}")
                    print(f"❌ [CHROMADB DEBUG] Exception type: {type(e).__name__}")
                    import traceback
                    print("❌ [CHROMADB DEBUG] Full traceback:")
                    traceback.print_exc()
                    print("📋 [CHROMADB DEBUG] Continuing with ChromaDB results only")
            elif use_gemma_reranking and not GEMINI_AVAILABLE:
                print("⚠️  [CHROMADB DEBUG] Gemini reranking module not available - using ChromaDB results only")

            # Fallback to original ChromaDB ranking
            return initial_results[:n_results]

        except Exception as e:
            print(f"❌ ChromaDB semantic search failed: {e}")
            return []

    def category_search(self, category: str, value: Any = True,
                        n_results: int = 50) -> List[Dict[str, Any]]:
        """Search by category using ChromaDB metadata filtering."""
        if not self.chroma_handler:
            print("❌ ChromaDB not available for search")
            return []

        try:
            results = self.chroma_handler.search_by_category(category, value, n_results)
            return self._format_chroma_results(results)
        except Exception as e:
            print(f"❌ ChromaDB category search failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB collection statistics."""
        if not self.chroma_handler:
            return {'total_photos': 0, 'error': 'ChromaDB not available', 'available': False}

        try:
            stats = self.chroma_handler.get_collection_stats()
            stats['available'] = True
            return stats
        except Exception as e:
            return {'total_photos': 0, 'error': str(e), 'available': False}

    def _format_chroma_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format ChromaDB results into a consistent structure."""
        formatted_results: List[Dict[str, Any]] = []

        if not results.get('documents') or not results['documents'][0]:
            return formatted_results

        documents = results['documents'][0]
        metadatas = results.get('metadatas', [[]])[0]
        ids = results.get('ids', [[]])[0]
        distances = (results.get('distances', [[]])[0] if results.get('distances')
                     else [0.0] * len(documents))

        for _i, (doc, metadata, result_id, distance) in enumerate(
                zip(documents, metadatas, ids, distances)):
            formatted_result = {
                'id': result_id,
                'description': doc,
                'original_path': metadata.get('original_path', ''),
                'text_description': doc,  # For compatibility with existing gallery code
                'similarity_score': 1 - distance if distance is not None else 1.0,
                'metadata': metadata,
                'chromadb_result': True  # Flag to identify ChromaDB results
            }
            formatted_results.append(formatted_result)

        return formatted_results

    def write_chroma_gallery(self, query: str,
                             results: List[Dict[str, Any]],
                             search_type: str = "semantic") -> Path:
        """Write ChromaDB search results as HTML gallery."""
        self.data_loader.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.data_loader.output_dir / f"chroma_{search_type}_{ts}.html"

        def esc(t: str) -> str:
            return (t or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        def get_image_url(row: Dict[str, str]) -> str:
            """Get base64 data URL for original image from CSV path."""
            orig_path = row.get('original_path', '')
            if not orig_path:
                return ""

            # Get absolute path from CSV relative path
            project_root = self.data_loader.output_dir.parent.parent
            source_path = project_root / orig_path

            if not source_path.exists():
                return ""

            # Read image and convert to base64 data URL
            try:
                with open(source_path, 'rb') as f:
                    image_data = f.read()
                b64_data = base64.b64encode(image_data).decode()
                ext = source_path.suffix.lower()
                mime = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
                return f"data:{mime};base64,{b64_data}"
            except Exception:
                return ""

        parts: List[str] = []
        parts.append("<!DOCTYPE html><html><head><meta charset=\"utf-8\">")
        title = f"ChromaDB {search_type.title()} Search Results"
        parts.append(f"<title>{esc(title)}</title>")
        parts.append("<style>")
        parts.append("body{font-family:Arial,Helvetica,sans-serif;margin:20px}")
        parts.append(".header{background:#f5f5f5;padding:15px;border-radius:8px;margin-bottom:20px}")
        parts.append(".item{margin-bottom:30px;display:flex;gap:16px;align-items:flex-start;border:1px solid #eee;border-radius:8px;padding:15px}")
        parts.append(".image-container{flex-shrink:0}")
        parts.append("img{max-width:360px;height:auto;border:1px solid #ddd;border-radius:4px}")
        parts.append(".meta{max-width:800px;padding-left:10px}")
        parts.append(".path{font-size:0.9em;color:#666;margin-bottom:10px}")
        parts.append(".similarity-score{color:#0066cc;font-weight:bold;margin-top:10px}")
        parts.append(".chroma-badge{background:#4CAF50;color:white;padding:3px 8px;border-radius:12px;font-size:0.8em;display:inline-block}")
        parts.append("</style>")
        parts.append("</head><body>")

        # Header with statistics
        parts.append('<div class="header">')
        parts.append(f"<h2>{esc(title)}</h2>")
        if search_type == "semantic":
            parts.append(f"<p><strong>Query:</strong> {esc(query)}</p>")
        parts.append(f"<p><strong>Results:</strong> {len(results)} photos found</p>")
        parts.append('<span class="chroma-badge">🔍 ChromaDB Vector Search</span>')
        parts.append('</div>')

        for i, r in enumerate(results, 1):
            orig_path = r.get('original_path', '')
            desc = r.get('description', '')
            similarity = r.get('similarity_score', 0.0)

            # Get original image URL
            img_path = get_image_url(r)

            parts.append('<div class="item">')
            parts.append('<div class="image-container">')

            if img_path:
                filename = Path(orig_path).name if orig_path else f"Image {i}"
                parts.append(f'<img src="{esc(img_path)}" alt="{esc(filename)}" '
                           f'onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'flex\'">')
                parts.append('<div style="display:none;width:360px;height:240px;background:#f0f0f0;'
                           'border:1px solid #ddd;border-radius:4px;align-items:center;'
                           'justify-content:center;color:#666;font-size:14px;flex-direction:column">'
                           'Original image not found</div>')
            else:
                parts.append('<div style="width:360px;height:240px;background:#f0f0f0;'
                           'border:1px solid #ddd;border-radius:4px;display:flex;align-items:center;'
                           'justify-content:center;color:#666;font-size:14px">Original image not available</div>')

            parts.append("</div>")
            parts.append('<div class="meta">')

            # Show original filename prominently
            if orig_path:
                filename = Path(orig_path).name
                parts.append(f'<div class="path"><strong>📸 {esc(filename)}</strong></div>')

            parts.append(f"<h3>{i}. Historical Photo Analysis</h3>")

            if desc:
                parts.append(f"<p>{esc(desc)}</p>")

            # Show similarity score
            if search_type == "semantic":
                parts.append(f'<div class="similarity-score">🎯 Similarity: {similarity:.3f}</div>')

            # Show metadata if available
            metadata = r.get('metadata', {})
            if metadata:
                meta_info = []
                if metadata.get('has_jewish_symbols'):
                    meta_info.append("✡️ Jewish symbols")
                if metadata.get('has_nazi_symbols'):
                    meta_info.append("🚩 Nazi symbols")
                if metadata.get('has_hebrew_text'):
                    meta_info.append("📜 Hebrew text")
                if metadata.get('has_german_text'):
                    meta_info.append("📄 German text")
                if metadata.get('signs_of_violence'):
                    meta_info.append("⚠️ Violence indicators")

                if meta_info:
                    parts.append(f'<p><strong>Content:</strong> {", ".join(meta_info)}</p>')

                # Show people counts
                total_people = metadata.get('total_people', 0)
                if total_people > 0:
                    men = metadata.get('men_count', 0)
                    women = metadata.get('women_count', 0)
                    parts.append(
                        f'<p><strong>People:</strong> {total_people} total '
                        f'({men} men, {women} women)</p>')

            parts.append('<p style="font-size:0.9em;color:#888;margin-top:15px;">'
                        '<em>Showing original historical photograph from ChromaDB vector search</em></p>')
            parts.append("</div>")
            parts.append("</div>")

        parts.append("</body></html>")

        with open(out, "w", encoding="utf-8") as f:
            f.write("".join(parts))

        return out


class PineconeSearch:
    """Pinecone cloud-based search functionality."""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.pinecone_handler = None
        self._initialize_pinecone()

    def _initialize_pinecone(self) -> None:
        """Initialize Pinecone handler if available."""
        if not PINECONE_AVAILABLE:
            print("⚠️  Pinecone not available. Install with: pip install pinecone-client")
            return

        try:
            self.pinecone_handler = create_pinecone_handler()
            if self.pinecone_handler:
                print("✅ Pinecone search initialized")
            else:
                print("❌ Failed to initialize Pinecone handler")
        except Exception as e:
            print(f"❌ Pinecone initialization error: {e}")

    def is_available(self) -> bool:
        """Check if Pinecone search is available."""
        return self.pinecone_handler is not None

    def semantic_search(self, query: str, n_results: int = 10,
                        use_gemma_reranking: bool = True) -> List[Dict[str, Any]]:
        """Perform semantic search using Pinecone with optional Gemini reranking."""
        if not self.pinecone_handler:
            print("❌ Pinecone not available for search")
            return []

        try:
            # First, we need to create an embedding for the query
            from sentence_transformers import SentenceTransformer
            
            # Load the same model used for creating embeddings
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            query_vector = model.encode([query])[0].tolist()
            
            # Get more results for Gemini reranking (30) or requested amount
            initial_k = min(30, 100) if use_gemma_reranking else n_results
            results = self.pinecone_handler.search_photos(query_vector, initial_k)
            initial_results = self._format_pinecone_results(results)

            # Apply Gemini reranking if enabled and available
            if use_gemma_reranking and len(initial_results) > n_results and GEMINI_AVAILABLE:
                try:
                    print("🤖 [PINECONE DEBUG] Enhancing Pinecone results with Gemini AI reranking...")
                    print(f"🔍 [PINECONE DEBUG] Initial Pinecone results: {len(initial_results)}")
                    print(f"🔍 [PINECONE DEBUG] Requesting top {n_results} from Gemini")
                    print(f"🔍 [PINECONE DEBUG] Query: '{query}'")

                    # Clean results for Gemini reranking
                    clean_results = []
                    for result in initial_results:
                        clean_result = result.copy()
                        # Remove fields that might indicate ranking/scoring
                        keys_to_remove = ['similarity_score', 'pinecone_result', 'id', 'metadata']
                        for key in keys_to_remove:
                            clean_result.pop(key, None)
                        clean_results.append(clean_result)

                    reranker = GemmaReranker()
                    reranked_results = reranker.rerank_results(query, clean_results, top_k=n_results)

                    print(f"✅ [PINECONE DEBUG] Gemini reranking complete: {len(reranked_results)} results returned")
                    return reranked_results

                except Exception as e:
                    print(f"❌ [PINECONE DEBUG] Gemini reranking failed: {e}")
                    print("📋 [PINECONE DEBUG] Continuing with Pinecone results only")

            # Fallback to original Pinecone ranking
            return initial_results[:n_results]

        except Exception as e:
            print(f"❌ Pinecone semantic search failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        if not self.pinecone_handler:
            return {'total_photos': 0, 'error': 'Pinecone not available', 'available': False}

        try:
            stats = self.pinecone_handler.get_collection_stats()
            stats['available'] = True
            return stats
        except Exception as e:
            return {'total_photos': 0, 'error': str(e), 'available': False}

    def _format_pinecone_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format Pinecone search results to match expected structure."""
        formatted_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            
            # Convert Pinecone result to expected format
            formatted_result = {
                'id': result.get('id', ''),
                'similarity_score': result.get('score', 0.0),
                'original_path': metadata.get('original_path', ''),
                'filename': metadata.get('filename', ''),
                'file_stem': metadata.get('file_stem', ''),
                'processed_path': metadata.get('processed_path', ''),
                'colorized_path': metadata.get('colorized_path', ''),
                'collection_folder': metadata.get('collection_folder', ''),
                'indoor_outdoor': metadata.get('indoor_outdoor', ''),
                'total_people': int(metadata.get('total_people', 0)) if metadata.get('total_people') else 0,
                'men_count': int(metadata.get('men_count', 0)) if metadata.get('men_count') else 0,
                'women_count': int(metadata.get('women_count', 0)) if metadata.get('women_count') else 0,
                'people_under_18': metadata.get('people_under_18', '').lower() == 'true',
                'has_jewish_symbols': metadata.get('has_jewish_symbols', '').lower() == 'true',
                'has_nazi_symbols': metadata.get('has_nazi_symbols', '').lower() == 'true',
                'signs_of_violence': metadata.get('signs_of_violence', '').lower() == 'true',
                'has_ocr_text': metadata.get('has_ocr_text', '').lower() == 'true',
                'has_hebrew_text': metadata.get('has_hebrew_text', '').lower() == 'true',
                'has_german_text': metadata.get('has_german_text', '').lower() == 'true',
                'total_objects': int(metadata.get('total_objects', 0)) if metadata.get('total_objects') else 0,
                'document': metadata.get('document', ''),
                'pinecone_result': result  # Store original result for debugging
            }
            formatted_results.append(formatted_result)
        
        return formatted_results


class DashboardPipeline:
    """Main dashboard pipeline coordinator."""

    def __init__(self, results_dir: Optional[Path] = None):
        self.data_loader = DataLoader(results_dir)
        self.category_search = CategorySearch(self.data_loader)
        self.semantic_search = SemanticSearch(self.data_loader)
        self.chroma_search = ChromaSearch(self.data_loader)
        self.pinecone_search = PineconeSearch(self.data_loader)

    def run_category_search(self) -> None:
        """Run interactive category search."""
        if not self.data_loader.data_full.exists():
            print(f"Missing {self.data_loader.data_full}. Run the main app first.")
            return

        idx = self.data_loader.build_index()
        counts = self.category_search.compute_category_counts(idx)

        print("\nAvailable in data:")
        for category, title in self.category_search.CATEGORIES.items():
            print(f"- {title}: {counts[category]}")

        cat = self.category_search.select_category()
        if not cat:
            print("Invalid choice")
            return

        rows = self.category_search.filter_rows(idx, cat)
        out_txt = self.category_search.write_report(cat, rows)
        out_html = self.category_search.write_gallery(cat, rows)
        print(f"\nWrote report: {out_txt}")
        print(f"Wrote gallery: {out_html}")

        # Open HTML gallery in browser
        try:
            webbrowser.open(f"file://{out_html.absolute()}")
            print("✅ Gallery opened in browser")
        except Exception as e:
            print(f"⚠️  Could not open browser: {e}")

    def run_semantic_search(self) -> None:
        """Run interactive semantic search."""
        print("Semantic search (MiniLM-L6 + Gemini AI reranking). Type your query.")
        query = input("Query: ").strip()
        if not query:
            print("Empty query")
            return

        k_input = input("Top K (1/5/10): ").strip() or '5'
        try:
            k_int = int(k_input)
        except ValueError:
            k_int = 5

        # Automatically use Gemini reranking (will fallback if not available)
        use_gemma = True

        try:
            results, num_above = self.semantic_search.search(
                query, k=k_int, use_gemma_reranking=use_gemma)

            # Show results with ranking info
            ranking_info = " (Gemma AI reranked)" if use_gemma and any('gemma_rank' in r for r in results) else " (MiniLM ranking)"
            print(f"\nFound {num_above} items above similarity threshold. Top {len(results)} matches{ranking_info}:\n")

            for i, r in enumerate(results, 1):
                gemma_rank = f" [Gemma: #{r['gemma_rank']}]" if r.get('gemma_rank') else ""
                print(f"{i}. {r.get('original_path', '')}{gemma_rank}")
                desc = r.get('description', '')
                if desc:
                    print(f"   {desc}")
                print("")

            out_txt = self.semantic_search.write_report(query, results)
            out_html = self.semantic_search.write_gallery(query, results)
            print(f"Wrote report: {out_txt}")
            print(f"Wrote gallery: {out_html}")

            # Open HTML gallery in browser
            try:
                webbrowser.open(f"file://{out_html.absolute()}")
                print("✅ Gallery opened in browser")
            except Exception as e:
                print(f"⚠️  Could not open browser: {e}")

            # Ask if user wants to see similar photos using CLIP
            self._ask_for_clip_similarities(results)

        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Building embeddings first...")
            self.semantic_search.build_embeddings()
            print("Now you can run semantic search again.")

    def build_embeddings(self) -> None:
        """Build embeddings for semantic search."""
        self.semantic_search.build_embeddings()

    def _ask_for_clip_similarities(self, results: List[Dict[str, str]]) -> None:
        """Ask user if they want to see CLIP-based similar photos"""
        if not results:
            return

        # Check if CLIP embeddings are available
        status = get_embeddings_status()
        if not status['embeddings_ready']:
            print("\n🔍 CLIP similarity search not available (run generate_clip_embeddings.py first)")
            return

        print("\n🔍 Do you want to see similar photos using CLIP AI analysis?")
        choice = input("See similar photos? (y/n): ").strip().lower()

        if choice in ['y', 'yes']:
            self._run_clip_similarity_search(results)

    def _run_clip_similarity_search(self, results: List[Dict[str, str]]) -> None:
        """Run CLIP similarity search on semantic search results"""
        print("\nCLIP Similarity Options:")
        print("1) Visual similarity only (based on image content)")
        print("2) Combined visual + text similarity (best of both)")

        sim_choice = input("Choose similarity type (1-2): ").strip()

        # Ask for number of similar photos
        num_input = input("Number of similar photos per result (1-10, default 5): ").strip() or '5'
        try:
            num_similar = min(10, max(1, int(num_input)))
        except ValueError:
            num_similar = 5

        if sim_choice == '1':
            self._generate_visual_similarity_gallery(results, num_similar)
        elif sim_choice == '2':
            self._generate_combined_similarity_gallery(results, num_similar)
        else:
            print("Invalid choice")

    def _generate_visual_similarity_gallery(
                                            self,
                                            results: List[Dict[str,
                                            str]],
                                            num_similar: int) -> None:
        """Generate gallery showing visual similarities"""
        print(f"\n🖼️ Generating visual similarity gallery with {num_similar} similar photos per result...")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.data_loader.output_dir / f"visual_similarity_{ts}.html"

        # Generate the HTML with visual similarities
        html_content = self._build_similarity_html(
                                                   results,
                                                   num_similar,
                                                   "visual",
                                                   "Visual Similarity")

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Visual similarity gallery: {out_path}")

        # Open in browser
        try:
            webbrowser.open(f"file://{out_path.absolute()}")
            print("✅ Gallery opened in browser")
        except Exception as e:
            print(f"⚠️  Could not open browser: {e}")

    def _generate_combined_similarity_gallery(
                                              self,
                                              results: List[Dict[str,
                                              str]],
                                              num_similar: int) -> None:
        """Generate gallery showing combined visual+text similarities"""
        print(f"\n🎯 Generating combined similarity gallery with {num_similar} similar photos per result...")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.data_loader.output_dir / f"combined_similarity_{ts}.html"

        # Generate the HTML with combined similarities
        html_content = self._build_similarity_html(
                                                   results,
                                                   num_similar,
                                                   "combined",
                                                   "Combined Visual + Text Similarity")

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Combined similarity gallery: {out_path}")

        # Open in browser
        try:
            webbrowser.open(f"file://{out_path.absolute()}")
            print("✅ Gallery opened in browser")
        except Exception as e:
            print(f"⚠️  Could not open browser: {e}")

    def _build_similarity_html(
                               self,
                               results: List[Dict[str,
                               str]],
                               num_similar: int,
                               similarity_type: str,
                               title: str) -> str:
        """Build HTML content for similarity galleries"""
        def esc(t: str) -> str:
            return (t or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        def get_image_url(row_path: str) -> str:
            """Get base64 data URL for image"""
            if not row_path:
                return ""

            # Get absolute path from CSV relative path
            project_root = self.data_loader.output_dir.parent.parent
            source_path = project_root / row_path

            if not source_path.exists():
                return ""

            try:
                with open(source_path, 'rb') as f:
                    image_data = f.read()
                b64_data = base64.b64encode(image_data).decode()
                ext = source_path.suffix.lower()
                mime = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
                return f"data:{mime};base64,{b64_data}"
            except Exception:
                return ""

        parts = [
            "<!DOCTYPE html><html><head><meta charset=\"utf-8\">",
            f"<title>{esc(title)}</title>",
            "<style>",
            "body{font-family:Arial,Helvetica,sans-serif;margin:20px}",
            ".main-result{margin-bottom:50px;border-bottom:2px solid #eee;padding-bottom:30px}",
            ".original-photo{margin-bottom:20px}",
            ".similarities{margin-left:20px}",
            ".similarity-grid{display:grid;grid-template-columns:"
            "repeat(auto-fit,minmax(200px,1fr));gap:15px}",
            ".similarity-item{border:1px solid #ddd;border-radius:8px;padding:10px;background:#f9f9f9}",
            "img{max-width:100%;height:auto;border-radius:4px}",
            ".score{font-weight:bold;color:#0066cc;font-size:0.9em}",
            ".filename{font-size:0.8em;color:#666;margin-top:5px}",
            "h2{color:#333;border-bottom:1px solid #ccc;padding-bottom:5px}",
            "h3{color:#666;margin-top:20px}",
            "</style>",
            "</head><body>",
            f"<h1>{esc(title)}</h1>",
            f"<p>Showing {num_similar} most similar photos for each search result.</p>"
        ]

        for i, result in enumerate(results, 1):
            original_path = result.get('original_path', '')
            desc = result.get('description', '')

            parts.append('<div class="main-result">')
            parts.append('<div class="original-photo">')
            parts.append(f"<h2>{i}. Original: {esc(Path(original_path).name if original_path else 'Unknown')}</h2>")

            # Show original image
            img_url = get_image_url(original_path)
            if img_url:
                parts.append(f'<img src="{esc(img_url)}" alt="Original" style="max-width:400px">')
            else:
                parts.append('<div style="width:400px;height:300px;background:#f0f0f0;border:1px solid #ddd;display:flex;align-items:center;justify-content:center;color:#666">Original image not available</div>')

            if desc:
                parts.append(f"<p><strong>Description:</strong> {esc(desc)}</p>")

            parts.append("</div>")

            # Get similar photos
            if similarity_type == "visual":
                similar_photos = get_visual_similarities(original_path, num_similar)
                parts.append(f"<h3>🖼️ Visually Similar Photos (Top {num_similar})</h3>")
            else:  # combined
                similar_photos = get_combined_similarities(original_path, num_similar)
                parts.append(f"<h3>🎯 Combined Visual + "
                             f"Text Similar Photos (Top {num_similar})</h3>")

            if similar_photos:
                parts.append('<div class="similarities">')
                parts.append('<div class="similarity-grid">')

                for _j, sim in enumerate(similar_photos, 1):
                    sim_path = sim['path']
                    sim_score = sim['similarity_score']
                    sim_filename = sim['filename']
                    sim_img_url = get_image_url(sim_path)

                    parts.append('<div class="similarity-item">')
                    if sim_img_url:
                        parts.append(f'<img src="{esc(sim_img_url)}" alt="{esc(sim_filename)}">')
                    else:
                        parts.append('<div style="width:100%;height:150px;background:#f0f0f0;border:1px solid #ddd;display:flex;align-items:center;justify-content:center;color:#666;font-size:12px">Image not available</div>')

                    parts.append(f'<div class="score">Similarity: {sim_score:.3f}</div>')
                    parts.append(f'<div class="filename">{esc(sim_filename)}</div>')
                    parts.append('</div>')

                parts.append('</div>')
                parts.append('</div>')
            else:
                parts.append('<p><em>No similar photos found.</em></p>')

            parts.append('</div>')

        parts.append("</body></html>")
        return "".join(parts)

    def run_chroma_semantic_search(self) -> None:
        """Run ChromaDB-based semantic search."""
        if not self.chroma_search.is_available():
            print("❌ ChromaDB search is not available.")
            print("   Make sure ChromaDB is installed: pip install chromadb")
            print("   And that you have processed photos with ChromaDB storage enabled.")
            return

        stats = self.chroma_search.get_stats()
        print(f"📊 ChromaDB contains {stats.get('total_photos', 0)} photos")

        print("\n🔍 ChromaDB Semantic Search (Vector Database)")
        query = input("Enter search query: ").strip()
        if not query:
            print("Empty query")
            return

        k_input = input("Top K results (1/5/10/20): ").strip() or '10'
        try:
            k_int = min(50, max(1, int(k_input)))
        except ValueError:
            k_int = 10

        print(f"\n🔍 Searching ChromaDB for: '{query}'...")
        results = self.chroma_search.semantic_search(query, k_int)

        if not results:
            print("❌ No results found")
            return

        print(f"\n🎯 Found {len(results)} results from ChromaDB:")
        for i, r in enumerate(results, 1):
            similarity = r.get('similarity_score', 0.0)
            orig_path = r.get('original_path', 'Unknown')
            filename = Path(orig_path).name if orig_path else f"Result {i}"
            print(f"{i}. {filename} (similarity: {similarity:.3f})")

            desc = r.get('description', '')
            if desc and len(desc) > 100:
                print(f"   {desc[:100]}...")
            elif desc:
                print(f"   {desc}")

        # Generate and open gallery
        try:
            out_html = self.chroma_search.write_chroma_gallery(query, results, "semantic")
            print(f"\n📄 Gallery saved: {out_html}")

            # Open in browser
            webbrowser.open(f"file://{out_html.absolute()}")
            print("✅ Gallery opened in browser")
        except Exception as e:
            print(f"❌ Failed to generate gallery: {e}")

        # Ask if user wants to see CLIP-based similar photos
        self._ask_for_chroma_clip_similarities(results, search_type="semantic")

    def run_chroma_category_search(self) -> None:
        """Run ChromaDB-based category search."""
        if not self.chroma_search.is_available():
            print("❌ ChromaDB search is not available.")
            return

        stats = self.chroma_search.get_stats()
        print(f"📊 ChromaDB contains {stats.get('total_photos', 0)} photos")

        print("\n🏷️  ChromaDB Category Search")
        print("Available categories:")
        categories = {
            "1": ("has_jewish_symbols", "Photos with Jewish symbols"),
            "2": ("has_nazi_symbols", "Photos with Nazi symbols"),
            "3": ("has_hebrew_text", "Photos with Hebrew text"),
            "4": ("has_german_text", "Photos with German text"),
            "5": ("signs_of_violence", "Photos with violence indicators"),
            "6": ("indoor_outdoor", "Indoor photos"),
            "7": ("indoor_outdoor", "Outdoor photos")
        }

        for key, (_, desc) in categories.items():
            print(f"{key}) {desc}")

        choice = input("Select category (1-7): ").strip()
        if choice not in categories:
            print("Invalid choice")
            return

        category_field, category_desc = categories[choice]

        # Handle indoor/outdoor special case
        if category_field == "indoor_outdoor":
            value = "indoor" if choice == "6" else "outdoor"
        else:
            value = True

        print(f"\n🔍 Searching for: {category_desc}...")
        results = self.chroma_search.category_search(category_field, value, 50)

        if not results:
            print(f"❌ No results found for {category_desc}")
            return

        print(f"\n🎯 Found {len(results)} photos with {category_desc.lower()}:")
        for i, r in enumerate(results[:10], 1):  # Show first 10 in console
            orig_path = r.get('original_path', 'Unknown')
            filename = Path(orig_path).name if orig_path else f"Result {i}"
            print(f"{i}. {filename}")

        if len(results) > 10:
            print(f"... and {len(results) - 10} more results")

        # Generate and open gallery
        try:
            out_html = self.chroma_search.write_chroma_gallery(category_desc, results, "category")
            print(f"\n📄 Gallery saved: {out_html}")

            # Open in browser
            webbrowser.open(f"file://{out_html.absolute()}")
            print("✅ Gallery opened in browser")
        except Exception as e:
            print(f"❌ Failed to generate gallery: {e}")

        # Ask if user wants to see CLIP-based similar photos
        self._ask_for_chroma_clip_similarities(results, search_type="category")

    def get_chroma_stats(self) -> None:
        """Display ChromaDB statistics."""
        if not self.chroma_search.is_available():
            print("❌ ChromaDB is not available")
            print("   Install with: pip install chromadb")
            return

        stats = self.chroma_search.get_stats()
        print("📊 ChromaDB Statistics:")
        print(f"   Total photos: {stats.get('total_photos', 0)}")
        print(f"   Collection: {stats.get('collection_name', 'unknown')}")
        print(f"   Storage: {stats.get('persist_directory', 'unknown')}")

        if 'error' in stats:
            print(f"   Error: {stats['error']}")
        else:
            print("   Status: ✅ Ready")

    def run_pinecone_semantic_search(self) -> None:
        """Run Pinecone-based semantic search."""
        if not self.pinecone_search.is_available():
            print("❌ Pinecone search is not available.")
            print("   Make sure PINECONE_API_KEY is set in .env file")
            print("   And that you have migrated your data to Pinecone.")
            return

        stats = self.pinecone_search.get_stats()
        print(f"📊 Pinecone contains {stats.get('total_photos', 0)} photos")

        print("\n🔍 Pinecone Semantic Search (Cloud Vector Database)")
        query = input("Enter search query: ").strip()
        if not query:
            print("Empty query")
            return

        k_input = input("Top K results (1/5/10/20): ").strip() or '10'
        try:
            k_int = min(50, max(1, int(k_input)))
        except ValueError:
            k_int = 10

        print(f"\n🔍 Searching Pinecone for: '{query}'...")
        results = self.pinecone_search.semantic_search(query, k_int)

        if not results:
            print("❌ No results found")
            return

        print(f"\n🎯 Found {len(results)} results from Pinecone:")
        for i, r in enumerate(results, 1):
            similarity = r.get('similarity_score', 0.0)
            orig_path = r.get('original_path', 'Unknown')
            filename = Path(orig_path).name if orig_path else f"Result {i}"
            print(f"{i}. {filename} (similarity: {similarity:.3f})")

            desc = r.get('document', '')
            if desc and len(desc) > 100:
                print(f"   {desc[:100]}...")
            elif desc:
                print(f"   {desc}")

        # Generate and open gallery automatically
        try:
            out_html = self._create_html_gallery(
                results, 
                f"Pinecone Semantic Search - {query}",
                "pinecone_semantic"
            )
            print(f"\n📄 Gallery saved: {out_html}")

            # Open in browser automatically
            webbrowser.open(f"file://{out_html.absolute()}")
            print("✅ Gallery opened in browser")
        except Exception as e:
            print(f"❌ Failed to generate gallery: {e}")

        # Ask if user wants to see CLIP-based similar photos (if available)
        self._ask_for_pinecone_clip_similarities(results, search_type="semantic")
                
    def get_pinecone_stats(self) -> None:
        """Display Pinecone statistics."""
        if not self.pinecone_search.is_available():
            print("❌ Pinecone is not available")
            print("   Make sure PINECONE_API_KEY is set in .env file")
            print("   Get your free API key at: https://www.pinecone.io/")
            return

        stats = self.pinecone_search.get_stats()
        print("📊 Pinecone Statistics:")
        print(f"   Total photos: {stats.get('total_photos', 0)}")
        print(f"   Index name: {stats.get('index_name', 'unknown')}")
        print(f"   Vector dimension: {stats.get('dimension', 'unknown')}")
        print(f"   Index fullness: {stats.get('index_fullness', 'unknown')}")
        
        if 'error' in stats:
            print(f"   Error: {stats['error']}")
        else:
            print("   Status: ✅ Ready")
            print("   🌐 Cloud-hosted and accessible from anywhere!")

    def _ask_for_chroma_clip_similarities(
                                          self,
                                          results: List[Dict[str,
                                          Any]],
                                          search_type: str = "semantic") -> None:
        """Ask user if they want to see CLIP-based similar photos for ChromaDB results"""
        if not results:
            return

        # Check if CLIP embeddings are available
        status = get_embeddings_status()
        if not status['embeddings_ready']:
            print("\n🔍 CLIP similarity search not available (run generate_clip_embeddings.py first)")
            return

        print("\n🔍 Do you want to see similar photos using CLIP AI analysis?")
        choice = input("See similar photos? (y/n): ").strip().lower()

        if choice in ['y', 'yes']:
            self._run_chroma_clip_similarity_search(results, search_type)

    def _run_chroma_clip_similarity_search(
                                           self,
                                           results: List[Dict[str,
                                           Any]],
                                           search_type: str) -> None:
        """Run CLIP similarity search on ChromaDB results"""
        print("\nCLIP Similarity Options:")
        print("1) Visual similarity only (based on image content)")
        print("2) Combined visual + text similarity (best of both)")

        sim_choice = input("Choose similarity type (1-2): ").strip()

        # Ask for number of similar photos
        num_input = input("Number of similar photos per result (1-10, default 5): ").strip() or '5'
        try:
            num_similar = min(10, max(1, int(num_input)))
        except ValueError:
            num_similar = 5

        if sim_choice == '1':
            self._generate_chroma_visual_similarity_gallery(results, num_similar, search_type)
        elif sim_choice == '2':
            self._generate_chroma_combined_similarity_gallery(results, num_similar, search_type)
        else:
            print("Invalid choice")

    def _generate_chroma_visual_similarity_gallery(
                                                   self,
                                                   results: List[Dict[str,
                                                   Any]],
                                                   num_similar: int,
                                                   search_type: str) -> None:
        """Generate gallery showing visual similarities for ChromaDB results"""
        print(f"\n🖼️ Generating visual similarity gallery with {num_similar} similar photos per result...")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.data_loader.output_dir / f"chroma_{search_type}_visual_similarity_{ts}.html"

        # Generate the HTML with visual similarities
        html_content = self._build_chroma_similarity_html(results, num_similar, "visual",
                                                         f"ChromaDB {search_type.title()} + Visual Similarity")

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Visual similarity gallery: {out_path}")

        # Open in browser
        try:
            webbrowser.open(f"file://{out_path.absolute()}")
            print("✅ Gallery opened in browser")
        except Exception as e:
            print(f"⚠️  Could not open browser: {e}")

    def _generate_chroma_combined_similarity_gallery(
                                                     self,
                                                     results: List[Dict[str,
                                                     Any]],
                                                     num_similar: int,
                                                     search_type: str) -> None:
        """Generate gallery showing combined visual+text similarities for ChromaDB results"""
        print(f"\n🎯 Generating combined similarity gallery with {num_similar} similar photos per result...")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.data_loader.output_dir / f"chroma_{search_type}_combined_similarity_{ts}.html"

        # Generate the HTML with combined similarities
        html_content = self._build_chroma_similarity_html(results, num_similar, "combined",
                                                         f"ChromaDB {search_type.title()} + Combined Visual + Text Similarity")

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Combined similarity gallery: {out_path}")

        # Open in browser
        try:
            webbrowser.open(f"file://{out_path.absolute()}")
            print("✅ Gallery opened in browser")
        except Exception as e:
            print(f"⚠️  Could not open browser: {e}")

    def _build_chroma_similarity_html(
                                      self,
                                      results: List[Dict[str,
                                      Any]],
                                      num_similar: int,
                                      similarity_type: str,
                                      title: str) -> str:
        """Build HTML content for ChromaDB similarity galleries"""
        def esc(t: str) -> str:
            return (t or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        def get_image_url(row_path: str) -> str:
            """Get base64 data URL for image"""
            if not row_path:
                return ""

            # Get absolute path from CSV relative path
            project_root = self.data_loader.output_dir.parent.parent
            source_path = project_root / row_path

            if not source_path.exists():
                return ""

            try:
                with open(source_path, 'rb') as f:
                    image_data = f.read()
                b64_data = base64.b64encode(image_data).decode()
                ext = source_path.suffix.lower()
                mime = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
                return f"data:{mime};base64,{b64_data}"
            except Exception:
                return ""

        parts = [
            "<!DOCTYPE html><html><head><meta charset=\"utf-8\">",
            f"<title>{esc(title)}</title>",
            "<style>",
            "body{font-family:Arial,Helvetica,sans-serif;margin:20px}",
            ".header{background:#f5f5f5;padding:15px;border-radius:8px;margin-bottom:20px;border:2px solid #4CAF50}",
            ".chroma-badge{background:#4CAF50;color:white;padding:5px 12px;border-radius:15px;font-size:0.9em;display:inline-block;margin-bottom:10px}",
            ".main-result{margin-bottom:50px;border-bottom:2px solid #eee;padding-bottom:30px}",
            ".original-photo{margin-bottom:20px;background:#fafafa;padding:15px;border-radius:8px}",
            ".similarities{margin-left:20px}",
            ".similarity-grid{display:grid;grid-template-columns:"
            "repeat(auto-fit,minmax(200px,1fr));gap:15px}",
            ".similarity-item{border:1px solid #ddd;border-radius:8px;padding:10px;background:#f9f9f9}",
            "img{max-width:100%;height:auto;border-radius:4px}",
            ".score{font-weight:bold;color:#0066cc;font-size:0.9em}",
            ".filename{font-size:0.8em;color:#666;margin-top:5px}",
            ".description{font-size:0.9em;margin:10px 0;color:#444}",
            "h2{color:#333;border-bottom:1px solid #ccc;padding-bottom:5px}",
            "h3{color:#666;margin-top:20px}",
            "</style>",
            "</head><body>",
            '<div class="header">',
            '<span class="chroma-badge">🔍 ChromaDB Vector Search</span>',
            f"<h1>{esc(title)}</h1>",
            f"<p><strong>Showing {num_similar} most similar photos for each ChromaDB search result.</strong></p>",
            '</div>'
        ]

        for i, result in enumerate(results, 1):
            original_path = result.get('original_path', '')
            desc = result.get('description', '') or result.get('text_description', '')
            similarity_score = result.get('similarity_score', 0.0)

            parts.append('<div class="main-result">')
            parts.append('<div class="original-photo">')
            parts.append(f"<h2>{i}. ChromaDB Result: {esc(Path(original_path).name if original_path else 'Unknown')}</h2>")

            # Show ChromaDB similarity score if available
            if similarity_score > 0:
                parts.append(f'<p><strong>ChromaDB Similarity:</strong> {similarity_score:.3f}</p>')

            # Show original image
            img_url = get_image_url(original_path)
            if img_url:
                parts.append(f'<img src="{esc(img_url)}" alt="Original" style="max-width:400px">')
            else:
                parts.append('<div style="width:400px;height:300px;background:#f0f0f0;border:1px solid #ddd;display:flex;align-items:center;justify-content:center;color:#666">Original image not available</div>')

            if desc:
                parts.append(f'<div class="description"><strong>Description:</strong> {esc(desc)}</div>')

            parts.append("</div>")

            # Get similar photos using CLIP
            if similarity_type == "visual":
                similar_photos = get_visual_similarities(original_path, num_similar)
                parts.append(f"<h3>🖼️ Visually Similar Photos (Top {num_similar})</h3>")
            else:  # combined
                similar_photos = get_combined_similarities(original_path, num_similar)
                parts.append(f"<h3>🎯 Combined Visual + "
                             f"Text Similar Photos (Top {num_similar})</h3>")

            if similar_photos:
                parts.append('<div class="similarities">')
                parts.append('<div class="similarity-grid">')

                for _j, sim in enumerate(similar_photos, 1):
                    sim_path = sim['path']
                    sim_score = sim['similarity_score']
                    sim_filename = sim['filename']
                    sim_img_url = get_image_url(sim_path)

                    parts.append('<div class="similarity-item">')
                    if sim_img_url:
                        parts.append(f'<img src="{esc(sim_img_url)}" alt="{esc(sim_filename)}">')
                    else:
                        parts.append('<div style="width:100%;height:150px;background:#f0f0f0;border:1px solid #ddd;display:flex;align-items:center;justify-content:center;color:#666;font-size:12px">Image not available</div>')

                    parts.append(f'<div class="score">CLIP Similarity: {sim_score:.3f}</div>')
                    parts.append(f'<div class="filename">{esc(sim_filename)}</div>')
                    parts.append('</div>')

                parts.append('</div>')
                parts.append('</div>')
            else:
                parts.append('<p><em>No similar photos found via CLIP analysis.</em></p>')

            parts.append('</div>')

        parts.extend([
            '<div style="margin-top:30px;padding:15px;background:#f0f8ff;border-radius:8px;border-left:4px solid #0066cc">',
            '<h3>🤖 AI Analysis Combination</h3>',
            '<p><strong>ChromaDB:</strong> Found initial results using vector semantic search</p>',
            '<p><strong>CLIP:</strong> Found visually similar photos using computer vision analysis</p>',
            '<p>This combination provides both semantically relevant and visually similar historical photographs.</p>',
            '</div>',
            "</body></html>"
        ])

        return "".join(parts)

    def _create_html_gallery(self, results: List[Dict[str, Any]], title: str, search_type: str) -> Path:
        """Create HTML gallery for search results."""
        self.data_loader.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.data_loader.output_dir / f"{search_type}_{ts}.html"

        def esc(t: str) -> str:
            return (t or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        def get_image_url(result: Dict[str, Any]) -> str:
            """Get base64 data URL for original image."""
            orig_path = result.get('original_path', '')
            if not orig_path:
                return ""

            # Get absolute path
            project_root = self.data_loader.output_dir.parent.parent
            source_path = project_root / orig_path

            if source_path.exists():
                try:
                    import base64
                    with open(source_path, 'rb') as f:
                        img_data = f.read()
                    ext = source_path.suffix.lower()
                    if ext == '.jpg' or ext == '.jpeg':
                        mime_type = 'image/jpeg'
                    elif ext == '.png':
                        mime_type = 'image/png'
                    else:
                        mime_type = 'image/jpeg'
                    
                    b64_data = base64.b64encode(img_data).decode('utf-8')
                    return f"data:{mime_type};base64,{b64_data}"
                except Exception:
                    pass
            return ""

        # HTML template
        parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{esc(title)}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }",
            "h1 { color: #333; text-align: center; }",
            ".gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }",
            ".item { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }",
            ".item img { max-width: 100%; height: auto; border-radius: 4px; }",
            ".filename { font-weight: bold; color: #0066cc; margin: 10px 0 5px 0; }",
            ".score { color: #666; font-size: 0.9em; margin-bottom: 10px; }",
            ".metadata { font-size: 0.85em; color: #555; margin-top: 10px; }",
            ".metadata div { margin: 2px 0; }",
            ".no-image { width: 100%; height: 200px; background: #f0f0f0; border: 1px solid #ddd; display: flex; align-items: center; justify-content: center; color: #666; border-radius: 4px; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{esc(title)}</h1>",
            f"<p>Found {len(results)} results</p>",
            "<div class='gallery'>"
        ]

        for i, result in enumerate(results):
            parts.append("<div class='item'>")
            
            # Image
            img_url = get_image_url(result)
            if img_url:
                parts.append(f"<img src='{img_url}' alt='Photo {i+1}'>")
            else:
                parts.append("<div class='no-image'>Image not available</div>")
            
            # Filename and score
            filename = result.get('filename', 'Unknown')
            score = result.get('similarity_score', 0)
            parts.append(f"<div class='filename'>{esc(filename)}</div>")
            parts.append(f"<div class='score'>Similarity Score: {score:.3f}</div>")
            
            # Metadata
            parts.append("<div class='metadata'>")
            
            metadata_fields = [
                ('original_path', 'Path'),
                ('total_people', 'People Count'),
                ('has_jewish_symbols', 'Jewish Symbols'),
                ('has_nazi_symbols', 'Nazi Symbols'),
                ('signs_of_violence', 'Violence'),
                ('has_ocr_text', 'Has Text'),
                ('has_hebrew_text', 'Hebrew Text'),
                ('has_german_text', 'German Text'),
                ('indoor_outdoor', 'Location')
            ]
            
            for field, label in metadata_fields:
                value = result.get(field)
                if value is not None and str(value).strip():
                    if isinstance(value, bool):
                        value_str = "Yes" if value else "No"
                    else:
                        value_str = str(value)
                    parts.append(f"<div><strong>{label}:</strong> {esc(value_str)}</div>")
            
            parts.append("</div>")
            parts.append("</div>")

        parts.extend([
            "</div>",
            f"<p style='text-align: center; margin-top: 30px; color: #666;'>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            "</body>",
            "</html>"
        ])

        # Write file
        with open(out, 'w', encoding='utf-8') as f:
            f.write('\n'.join(parts))

        return out

    def _display_search_results(self, results: List[Dict[str, Any]], title: str) -> None:
        """Display search results in a formatted list."""
        if not results:
            print("No results to display.")
            return

        print(f"\n{title}")
        print("=" * len(title))
        
        for i, result in enumerate(results, 1):
            filename = result.get('filename', 'Unknown')
            score = result.get('similarity_score', 0)
            path = result.get('original_path', '')
            
            print(f"\n{i}. {filename}")
            print(f"   Score: {score:.3f}")
            print(f"   Path: {path}")
            
            # Show key metadata
            people = result.get('total_people', 0)
            if people > 0:
                print(f"   People: {people}")
            
            flags = []
            if result.get('has_jewish_symbols'):
                flags.append("Jewish symbols")
            if result.get('has_nazi_symbols'):
                flags.append("Nazi symbols")
            if result.get('signs_of_violence'):
                flags.append("Violence")
            if result.get('has_hebrew_text'):
                flags.append("Hebrew text")
            if result.get('has_german_text'):
                flags.append("German text")
            
            if flags:
                print(f"   Flags: {', '.join(flags)}")

    def _ask_for_pinecone_clip_similarities(self, results: List[Dict[str, Any]], search_type: str = "semantic") -> None:
        """Ask user if they want to see CLIP-based similar photos for Pinecone results"""
        if not results:
            return

        # Check if CLIP embeddings are available
        status = get_embeddings_status()
        if not status['embeddings_ready']:
            print("\n🔍 CLIP similarity search not available (run generate_clip_embeddings.py first)")
            return

        print(f"\n🔍 CLIP-based similar photo search available!")
        print("This will find visually similar photos for each result using computer vision.")
        
        choice = input("Show CLIP similarities? (y/N): ").strip().lower()
        if choice in ['y', 'yes']:
            # Use existing CLIP similarity functionality
            self._ask_for_clip_similarities(results)


def main() -> None:
    """Main entry point for dashboard pipeline."""
    import sys

    pipeline = DashboardPipeline()

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "category":
            pipeline.run_category_search()
        elif command == "semantic":
            pipeline.run_semantic_search()
        elif command == "build_embeddings":
            pipeline.build_embeddings()
        elif command == "chroma_semantic":
            pipeline.run_chroma_semantic_search()
        elif command == "chroma_category":
            pipeline.run_chroma_category_search()
        elif command == "chroma_stats":
            pipeline.get_chroma_stats()
        else:
            print("Usage: python dashboard_pipeline.py [category|semantic|build_embeddings|chroma_semantic|chroma_category|chroma_stats]")
    else:
        print("Dashboard Pipeline")
        print("Available commands:")
        print("1) category - Run category-based search (CSV)")
        print("2) semantic - Run semantic search (MiniLM)")
        print("3) build_embeddings - Build embeddings for semantic search")
        print("4) chroma_semantic - Run ChromaDB semantic search")
        print("5) chroma_category - Run ChromaDB category search")
        print("6) chroma_stats - Show ChromaDB statistics")
        choice = input("Enter command (1-6): ").strip().lower()

        if choice in ["1", "category"]:
            pipeline.run_category_search()
        elif choice in ["2", "semantic"]:
            pipeline.run_semantic_search()
        elif choice in ["3", "build_embeddings"]:
            pipeline.build_embeddings()
        elif choice in ["4", "chroma_semantic"]:
            pipeline.run_chroma_semantic_search()
        elif choice in ["5", "chroma_category"]:
            pipeline.run_chroma_category_search()
        elif choice in ["6", "chroma_stats"]:
            pipeline.get_chroma_stats()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
