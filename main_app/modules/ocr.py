from typing import Any, Dict, List
import os

from PIL import Image


class OCRRunner:
    def __init__(self) -> None:
        # Make OCR optional; don't crash the app if the package is missing
        self.available = True
        try:
            import pytesseract  # type: ignore
            self.pytesseract = __import__('pytesseract')
        except Exception:
            self.available = False
            self.pytesseract = None  # type: ignore[assignment]
            return

        # Try common macOS/Linux locations for the tesseract binary
        possible_bins = [
            "/opt/homebrew/bin/tesseract",  # Apple Silicon Homebrew
            "/usr/local/bin/tesseract",     # Intel Homebrew
            "/usr/bin/tesseract"            # System
        ]
        for path in possible_bins:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                try:
                    self.pytesseract.pytesseract.tesseract_cmd = path  # type: ignore[attr-defined]
                except Exception:
                    pass
                break

    def run_ocr(self, image_path: str, min_confidence: int = 60, max_lines: int = 50) -> Dict[str, Any]:
        """Run OCR on the given image and return text lines and bounding boxes.
        min_confidence: filter out weak OCR tokens (< value from 0-100)
        max_lines: limit number of combined lines to avoid very large outputs
        """
        if not self.available or self.pytesseract is None:  # type: ignore[truthy-bool]
            return {
                'has_text': False,
                'lines': [],
                'items': [],
                'error': 'pytesseract not installed'
            }

        img = Image.open(image_path).convert('RGB')
        try:
            data = self.pytesseract.image_to_data(img, output_type=self.pytesseract.Output.DICT)
        except Exception as exc:
            # Graceful fallback when the binary is missing or misconfigured
            return {
                'has_text': False,
                'lines': [],
                'items': [],
                'error': f"OCR unavailable: {exc}"
            }

        n = int(data.get('level', []) and len(data['level']) or 0)
        items: List[Dict[str, Any]] = []
        for i in range(n):
            try:
                text = data['text'][i].strip()
                conf = int(float(data['conf'][i]))
                if not text or conf < min_confidence:
                    continue
                x, y, w, h = int(data['left'][i]), int(data['top'][i]), int(data['width'][i]), int(data['height'][i])
                items.append({
                    'text': text,
                    'confidence': conf,
                    'bbox': {'x': x, 'y': y, 'w': w, 'h': h}
                })
            except Exception:
                continue

        # Combine tokens into lines by line number and block/page
        lines_map: Dict[str, List[str]] = {}
        for i in range(n):
            try:
                text = data['text'][i].strip()
                conf = int(float(data['conf'][i]))
                if not text or conf < min_confidence:
                    continue
                key = f"p{data['page_num'][i]}_b{data['block_num'][i]}_l{data['line_num'][i]}"
                lines_map.setdefault(key, []).append(text)
            except Exception:
                continue

        lines: List[str] = [" ".join(tokens) for _, tokens in sorted(lines_map.items())]
        if len(lines) > max_lines:
            lines = lines[:max_lines]

        return {
            'has_text': len(items) > 0,
            'lines': lines,
            'items': items
        }


