## TestPhotos - Dev Setup Guide (Cursor-friendly)

This document helps teammates get the project running quickly in Cursor on macOS or Windows.

### 1) Prerequisites
- Python 3.10+ (3.11 recommended)
- Git and Git LFS (`git lfs install` after Git is installed)
- Disk space: ~2–3 GB for models/results

Optional but recommended:
- macOS: Homebrew (`brew`)
- Windows: Tesseract OCR installer

### 2) Clone and environment
```bash
git clone <your-repo-url>
cd TestPhotos
cp .env.example .env
# Edit .env and set GROQ_API_KEY
```

Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install optional packages:
```bash
# OCR (optional)
pip install pytesseract

# Semantic search
pip install sentence-transformers

# If PyTorch is missing on Windows, install a matching wheel from pytorch.org first
```

### 3) Tesseract OCR
- macOS: `brew install tesseract` (binary usually at `/opt/homebrew/bin/tesseract`)
- Windows: Download installer from `https://github.com/tesseract-ocr/tesseract` and set its path in `.env` if needed.

### 4) Git LFS (large files)
```bash
git lfs install
# Embeddings/results can be large; LFS rules are in .gitattributes
```

### 5) Run the main app
```bash
python -m main_app.app
```
You’ll be prompted to process 1 random, 5 random, or all photos.

Outputs go to `main_app_outputs/`:
- `results/summary.json`, `results/summary.txt`
- `results/data_full.csv`, `results/data_text.csv`
- `colorized/` images
- `data_search_dashord/` embeddings and search reports

### 6) Semantic search workflow
Build embeddings from results text:
```bash
python -m main_app.data_search_dashord.build_embeddings
```
Run semantic search:
```bash
python -m main_app.data_search_dashord.search_semantic
```

### 7) Groq LLM
- The app uses `meta-llama/llama-4-scout-17b-16e-instruct`.
- Put your key in `.env` as `GROQ_API_KEY`.
- Multi-photo runs enforce a delay between calls; errors write an error JSON and the app continues.

### 8) Windows notes
- Prefer Python from python.org (x64) or `winget install Python.Python.3.11`.
- If `torch` is missing, install from `https://pytorch.org/` matching your Python/CPU/GPU.
- Tesseract default path: `C:\\Program Files\\Tesseract-OCR\\tesseract.exe` (add to PATH or set in `.env`).

### 9) What not to commit
- `.env` is ignored.
- `main_app_outputs/` is ignored by default except: embeddings files can be committed (`embeddings_minilm_l6.npz`, `embeddings_meta.json`).
- Large models are ignored; use LFS if you must track them.

### 10) Pushing code
We do not push without explicit approval. Create a feature branch and open a PR for review.


