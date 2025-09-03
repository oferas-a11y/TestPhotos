## 2025-08-23

### Added
- `opencv_analysis/colorize_one.py`: Script to colorize a single photo using OpenCV DNN (Zhang et al. model).
- `pipeline/analyze_colorized.py`: Pipeline that runs YOLO (v8m or v9) on colorized images and classifies detected people with CLIP as man/woman/child. Outputs JSON with per-person labels and summary counts.
- `Dockerfile`: Container to run the pipeline with all dependencies installed. Default command runs YOLOv8m on two colorized images.
- `opencv_analysis/models/` directory created to store model files:
  - `colorization_deploy_v2.prototxt`
  - `pts_in_hull.npy`
  - `colorization_release_v2.caffemodel` (must be downloaded manually; see below)

### Why
- Enable quick verification of OpenCV DNN-based colorization on one image so results can be inspected.

### What works
- The script loads the model, injects cluster centers, and colorizes an input image, saving the result to `results/`.
- Verified dependencies are already listed in `requirements.txt` (`opencv-python`, `numpy`).

### What needs attention
- The `colorization_release_v2.caffemodel` file is large and some hosts block automated downloads. If the file size is not >100MB, it is likely an HTML page or LFS pointer. Download manually via a browser from an official source and place it at `opencv_analysis/models/colorization_release_v2.caffemodel`.

Recommended sources (open in a browser):
- Berkeley: `https://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel`
- Mirror: `https://people.eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel`

Expected size: approximately 128MB.

### Usage
Run the YOLO+CLIP pipeline locally on two colorized images:

```bash
python pipeline/analyze_colorized.py \
  --images "results/colorized_Figure_22.jpg" "results/colorized_Figure_1.jpg" \
  --yolo_model yolov8m.pt \
  --output pipeline/colorized_analysis_yolov8m.json
```

Try with YOLOv9:

```bash
python pipeline/analyze_colorized.py \
  --images "results/colorized_Figure_22.jpg" "results/colorized_Figure_1.jpg" \
  --yolo_model yolov9c.pt \
  --output pipeline/colorized_analysis_yolov9c.json
```

Docker build and run:

```bash
docker build -t testphotos-pipeline .
docker run --rm -v "$PWD/results:/app/results" -v "$PWD/pipeline:/app/pipeline" testphotos-pipeline
```

You can override the default command to use YOLOv9:

```bash
docker run --rm -v "$PWD/results:/app/results" -v "$PWD/pipeline:/app/pipeline" testphotos-pipeline \
  python pipeline/analyze_colorized.py --images results/colorized_Figure_22.jpg results/colorized_Figure_1.jpg \
  --yolo_model yolov9c.pt --output pipeline/colorized_analysis_yolov9c.json
```

Colorize one sample image after placing the Caffe model file:

```bash
python opencv_analysis/colorize_one.py \
  --input "sample_photos/Figure 1.jpg" \
  --output "results/colorized_Figure_1.jpg" \
  --models_dir "opencv_analysis/models"
```

You can change `--input` to any path. The script will write the output image to the path given in `--output`.

# Changelog

All notable changes to this project will be documented in this file.

## 2025-09-03

### Added
- `PIPELINE.md`: Clear, non-technical documentation of the end-to-end processing pipeline (colorization → YOLO → CLIP → OCR → LLM → CSVs → summaries) and the search pipeline (category and semantic searches). Includes which tool is used at each step and its category (pre-trained, zero-shot, OCR, LLM), where outputs are written, and how to run both pipelines.

### Why
- Provide a single authoritative, readable guide for teammates and stakeholders that explains what the system does, how it works, and how to run searches on the results without reading code.

### What works
- Processing runs interactively (choose 1/5/all photos), writes per-photo JSONs, summaries, and two CSVs aligned to the original black-and-white photo paths.
- Search pipeline supports category counts/reports/galleries and semantic search with MiniLM-L6 embeddings and galleries.

### What needs attention
- Ensure Tesseract and pytesseract are installed if OCR results are required. Otherwise the pipeline proceeds without OCR.

## 2025-02-03

### YOLO Detection Improvements (no fine-tuning)

- Switched default YOLO model to `yolov8s` for better accuracy on M1 8GB devices.
- Added EXIF-aware preprocessing with CLAHE and mild sharpening to enhance historical scans.
- Tuned inference parameters:
  - `imgsz=1280`, `iou=0.65`, `max_det=300`, and `classes` restricted to relevant COCO IDs.
  - Introduced per-class confidence thresholds (lower for `person`), plus global base confidence `0.4`.
  - Applied post-filter on minimum box area ratio (`0.003` of image area) to reduce tiny false positives.
- Kept optional test-time augmentation flag (`augment`) off by default to stay within M1 8GB memory.
- Updated `yolo_detection/README.md` to document new defaults and configuration knobs.

### Combined YOLO + OpenCV (2025-02-03)
- Integrated OpenCV auxiliary analysis into YOLO pipeline (behind `enable_opencv_aux=True`, default on):
  - Haar-based face and cat face detection
  - HOG+SVM person detection
  - Hough line segments and scene categorization (indoor/outdoor; street/forest/field/park/water)
- Overlays include OpenCV cues alongside YOLO boxes; JSON outputs expose an `opencv_aux` section per photo.

### What works
- Noticeable recall/precision gains on persons and common objects in historical photos without training.
- Stable performance on Apple M1 8GB with default settings.

### What needs attention
- Tiling inference for ultra-high-resolution scans is not yet implemented; could further boost small-object recall.
- Optional multi-scale/WBF ensemble remains a future enhancement if compute allows.

