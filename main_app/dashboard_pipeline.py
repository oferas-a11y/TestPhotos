"""Dashboard search pipeline (category + semantic search)."""

import base64
import csv
import json
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import unicodedata
import re

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore


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
            if k in idx:
                idx[k]["text_description"] = r.get("description", "")
            else:
                idx[k] = {"text_description": r.get("description", "")}

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

        texts = [r.get('description', '') for r in rows]
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
            'description': r.get('description', '')
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

    def search(self, query: str, k: int = 5, threshold: float = 0.35) -> Tuple[List[Dict[str, str]], int]:
        """Perform semantic search."""
        embs, meta = self.load_embeddings()
        model = SentenceTransformer(self.model_name)

        # Encode query
        q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        # Calculate similarities
        sims = embs @ q
        num_above = int((sims >= threshold).sum())

        # Get top k results
        idxs = np.argsort(-sims)[:max(1, k)]
        results = [meta[int(i)] for i in idxs]

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
                parts.append(f'<div style="display:none;width:360px;height:240px;background:#f0f0f0;'
                           f'border:1px solid #ddd;border-radius:4px;align-items:center;'
                           f'justify-content:center;color:#666;font-size:14px;flex-direction:column">Original image not found</div>')
            else:
                parts.append('<div style="width:360px;height:240px;background:#f0f0f0;'
                           'border:1px solid #ddd;border-radius:4px;display:flex;align-items:center;'
                           'justify-content:center;color:#666;font-size:14px">Original image not available</div>')
            
            parts.append("</div>")
            parts.append('<div class="meta">')
            
            # Show original filename prominently
            if orig_path:
                parts.append(f'<div class="path"><strong>📸 {esc(Path(orig_path).name)}</strong></div>')
            
            parts.append(f"<h3>{i}. Historical Photo Analysis</h3>")
            if desc:
                parts.append(f"<p>{esc(desc)}</p>")
            
            # Add note about original B&W photos
            parts.append('<p style="font-size:0.9em;color:#888;margin-top:15px;">'
                        '<em>Showing original black & white historical photograph</em></p>')
            
            parts.append("</div>")
            parts.append("</div>")

        parts.append("</body></html>")
        out.write_text("\n".join(parts), encoding="utf-8")
        return out


class DashboardPipeline:
    """Main dashboard pipeline coordinator."""

    def __init__(self, results_dir: Optional[Path] = None):
        self.data_loader = DataLoader(results_dir)
        self.category_search = CategorySearch(self.data_loader)
        self.semantic_search = SemanticSearch(self.data_loader)

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
        print("Semantic search (MiniLM-L6). Type your query.")
        query = input("Query: ").strip()
        if not query:
            print("Empty query")
            return

        k_input = input("Top K (1/5/10): ").strip() or '5'
        try:
            k_int = int(k_input)
        except ValueError:
            k_int = 5

        try:
            results, num_above = self.semantic_search.search(query, k=k_int)

            print(f"\nFound {num_above} items above similarity threshold. Top {len(results)} matches:\n")
            for i, r in enumerate(results, 1):
                print(f"{i}. {r.get('original_path', '')}")
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

        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Building embeddings first...")
            self.semantic_search.build_embeddings()
            print("Now you can run semantic search again.")

    def build_embeddings(self) -> None:
        """Build embeddings for semantic search."""
        self.semantic_search.build_embeddings()


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
        else:
            print("Usage: python dashboard_pipeline.py [category|semantic|build_embeddings]")
    else:
        print("Dashboard Pipeline")
        print("Available commands:")
        print("1) category - Run category-based search")
        print("2) semantic - Run semantic search")
        print("3) build_embeddings - Build embeddings for semantic search")
        choice = input("Enter command: ").strip().lower()

        if choice in ["1", "category"]:
            pipeline.run_category_search()
        elif choice in ["2", "semantic"]:
            pipeline.run_semantic_search()
        elif choice in ["3", "build_embeddings"]:
            pipeline.build_embeddings()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
