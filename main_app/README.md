# Main App: Colorize + YOLO + CLIP

This app ingests photos, colorizes them, runs YOLO object detection and CLIP semantic analysis. Outputs are clear about which model produced which result.

## Key points
- YOLO: bounding-box object detection. We report counts and top detections.
- CLIP scene: zero-shot scene categories from the full image.
- CLIP people: gender scores on person crops extracted from YOLO boxes.
 

## Usage
```bash
python -m main_app.app --input_dir sample_photos --output_dir main_app_outputs \
  --models_dir opencv_analysis/models --ab_boost 1.0 --yolo_model_size s \
  --clip_model_name ViT-L/14@336px --confidence_yolo 0.4 --confidence_clip 0.3
```

Outputs:
- main_app_outputs/colorized/ — colorized images
- main_app_outputs/results/summary.json — structured results
- main_app_outputs/results/summary.txt — human-friendly summary
 

## Notes
- Colorization uses OpenCV DNN (Zhang et al.). No prompts.
- YOLO is used strictly for object detection (boxes, classes).
- CLIP scene uses full image; CLIP gender uses YOLO person crops.
 
