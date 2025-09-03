# TestPhotos Pipelines (Processing + Search)

This document explains, in plain language, how the system analyzes historical photos and how you can explore the results. It is organized in the order things happen, and it names the tool used at each step plus the type of method: pre-trained model, zero-shot model, OCR, or LLM.

## 1) Processing Pipeline (from photos to results)

Entry commands (either one):

```bash
python -m main_app.app --input_dir sample_photos
```

```bash
python -m main_app.main_pipeline --input_dir sample_photos
```

### 1. Select which photos to process

- What happens: The app asks you to pick 1 random photo, 5 random photos, or all photos from the input folder.
- Input folder: `--input_dir` (default is `sample_photos`).
- No model here; it’s just selecting files.

### 2. Colorize the selected photos

- Tool: OpenCV DNN Colorization (Zhang et al.)
- Category: pre-trained model (no training done here; we load published weights)
- Purpose: Creates colorized versions of black-and-white photos for better detection/analysis later.
- Output: Colorized images saved to `main_app_outputs/colorized/`.
- Note: Only the analysis models use the colorized image. All data exports and search refer to the original black-and-white paths.

### 3. Detect objects (people, etc.)

- Tool: YOLOv8 (Ultralytics), wrapped by `main_app/modules/yolo_runner.py` and `yolo_detection/yolo_detector.py`
- Category: pre-trained model
- Purpose: Finds objects and people; counts them; returns bounding boxes and per-person rectangles for later steps.
- Output fields: object counts, full detections (with locations and centers), a list of person bounding boxes.

### 4. Understand the environment and people’s gender

- Tool: CLIP (Contrastive Language-Image Pre-training) via `main_app/modules/clip_runner.py`
- Category: zero-shot model (no fine-tuning; it recognizes concepts from text prompts)
- What it does:
  - Indoor vs. Outdoor classification (binary, zero-shot prompts)
  - Background categories (e.g., street, field, forest, church interior)
  - Gender for each detected person crop (man vs. woman, with clear tie-breaking: ties go to woman; otherwise higher score)
- Output fields: indoor/outdoor label, background categories, and per-person gender scores/labels added to YOLO detections.

### 5. Read any visible text (optional)

- Tool: Tesseract OCR via `pytesseract` in `main_app/modules/ocr.py`
- Category: OCR (text extraction)
- Purpose: Finds text and returns tokens with bounding boxes, plus combined lines.
- Behavior if missing: If OCR isn’t installed, the app continues safely and marks OCR as unavailable for that photo.

### 6. Expert interpretation (LLM) on the original black‑and‑white image

- Tool: Groq LLM in `main_app/modules/llm.py`, model `meta-llama/llama-4-scout-17b-16e-instruct`
- Category: LLM
- Prompt: Fixed, user-provided prompt; the app does not change it.
- Purpose: Produces a structured JSON analysis (caption, minors count, Jewish/Nazi symbols, Hebrew/German texts with translation, up to 5 key objects, and violence assessment).
- Rate limiting: When processing multiple photos, the app calls the LLM at most once every ~45 seconds to avoid rate limits.
- Reliability: Includes retries and saves a small error JSON if the API fails or returns malformed JSON, so the run never stalls.
- Output: LLM JSON saved as `main_app_outputs/results/llm_<photo>_orig.json` and linked into the per-photo record.

### 7. Conditional “army objects” pass (only in certain scenes)

- Tool: Dedicated YOLO model for army-related items in `main_app/modules/yolo_army.py`
- Category: pre-trained model
- When it runs: Only if more than one person is detected AND the background looks like a field or forest.
- Purpose: Detects army-related objects (e.g., uniform, rifle, cannon, camp) to support wartime evidence.

### 8. Results and summaries generated

- Per-photo JSON: `main_app_outputs/results/full_<photo>.json` (all detections, CLIP results, OCR, and attached LLM path/json)
- Summaries:
  - `main_app_outputs/results/summary.json` (machine-readable overview)
  - `main_app_outputs/results/summary.txt` (human-readable)
    - Shows object counts, per-person gender labels and scores, indoor/outdoor, top background categories, OCR presence and a few lines (if any), and LLM highlights (caption, symbol/text flags, selected details). Wartime words in the caption are removed unless there is evidence (army objects, Nazi symbols, or violence) to avoid assumptions.

### 9. CSV datasets for exploration and search

- `main_app_outputs/results/data_text.csv`
  - Purpose: A single fluent “description” string per photo for semantic embeddings (no category labels, just text).
  - Also stores paths to the original image, colorized image, full JSON, and LLM JSON.
- `main_app_outputs/results/data_full.csv`
  - Purpose: A compact, structured table of fields: indoor/outdoor, first background category, YOLO object counts, gender counts, LLM caption, text flags and texts (Hebrew/German), symbol details, violence flags/explanations, and object list.
- Overwrite behavior: If you re-run the pipeline for the same photo, its CSV rows are updated/overwritten by original photo path.

## 2) Search Pipeline (explore what the system found)

Entry command:

```bash
python -m main_app.dashboard_pipeline
```

Or run direct subcommands:

```bash
python -m main_app.dashboard_pipeline category
python -m main_app.dashboard_pipeline semantic
python -m main_app.dashboard_pipeline build_embeddings
```

### A. Category search (counts → pick a category → report + gallery)

- Uses: `data_full.csv` and `data_text.csv`
- Always shows original black-and-white images in the gallery
- Categories include: Nazi symbols, Jewish symbols, Hebrew text, German text, violence, indoor, outdoor
- Output:
  - Text report: `main_app_outputs/data_search_dashord/search_<category>_<timestamp>.txt`
  - HTML gallery: `main_app_outputs/data_search_dashord/category_<category>_<timestamp>.html`

### B. Semantic search (free-text query)

- Tool: Sentence Transformers `all-MiniLM-L6-v2` for embeddings
- Category: pre-trained text embedding model
- What happens:
  1) Build embeddings (once) from `data_text.csv` descriptions → saves `embeddings_minilm_l6.npz` and `embeddings_meta.json`.
  2) You type a query; the system finds the closest photos by text meaning.
  3) It reports how many items cross a similarity threshold, shows top matches, and writes a report plus an HTML gallery (original black-and-white thumbnails).
- Outputs:
  - Text report: `main_app_outputs/data_search_dashord/semantic_<timestamp>.txt`
  - HTML gallery: `main_app_outputs/data_search_dashord/semantic_<timestamp>.html`

## Why colorized for models but black‑and‑white for data and search?

- Detection/classification models sometimes work better on colorized inputs, especially for environmental cues.
- To avoid confusion and to keep archival integrity, all data exports and search operations reference the original black‑and‑white image paths. Only the modeling steps operate on the colorized copies.

## Quick reference: Tools and categories

- OpenCV Colorization (Zhang et al.): pre-trained
- YOLOv8 (Ultralytics): pre-trained
- CLIP: zero-shot (prompts; no fine-tuning)
- OCR (Tesseract via pytesseract): OCR (text extraction)
- Groq LLM (Llama 4 Scout): LLM (fixed prompt, JSON output)
- Sentence Transformers MiniLM-L6: pre-trained text embedding model
