# YOLO Object Detection

Advanced object detection for historical photos using pre-trained YOLOv8 models. Detects people, animals, vehicles, and everyday objects without requiring custom training.

## Features

### Object Detection Capabilities
- **People**: Person detection and counting
- **Animals**: Horses, dogs, cats, birds, sheep, cows
- **Vehicles**: Cars, bicycles, motorcycles, buses, trains, trucks
- **Household Items**: Chairs, tables, books, clocks, vases
- **Personal Items**: Bags, umbrellas, cups, utensils

### Advanced Analysis
- **Confidence Scoring**: Adjustable detection thresholds
- **Priority Scoring**: Historical relevance weighting
- **Visual Annotations**: Bounding boxes with labels
- **Batch Processing**: Analyze entire photo collections

## Files

- `yolo_detector.py` - Main YOLO detection script
- `yolo_detections.json` - Detection results (generated)
- `detection_visualizations/` - Annotated images (generated)
- `README.md` - This documentation

## Usage

### Basic Detection
```bash
cd yolo_detection
python3 yolo_detector.py
```

This will:
1. Download YOLOv8n model (first run only)
2. Analyze all photos in `../sample_photos/`
3. Create `yolo_detections.json` with results
4. Generate visualization images in `detection_visualizations/`

### Model Sizes Available
Edit the `model_size` variable in the script:
- `'n'` - Nano (fastest, ~6MB)
- `'s'` - Small (balanced, ~22MB)
- `'m'` - Medium (more accurate, ~52MB)
- `'l'` - Large (high accuracy, ~131MB)
- `'x'` - Extra Large (best accuracy, ~264MB)

## Output Format

### JSON Results Structure
```json
{
  "model_info": {
    "model_name": "YOLOv8n",
    "confidence_threshold": 0.5,
    "total_photos_analyzed": 68
  },
  "summary": {
    "photos_with_objects": 45,
    "photos_without_objects": 23,
    "total_objects_detected": 127,
    "average_objects_per_photo": 1.87,
    "unique_object_types": 12,
    "most_common_objects": {
      "person": 89,
      "chair": 15,
      "car": 8,
      "horse": 6,
      "dog": 4
    }
  },
  "photos": [
    {
      "filename": "photo.jpg",
      "total_objects": 3,
      "object_counts": {"person": 2, "horse": 1},
      "priority_score": 15,
      "detections": [
        {
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.85,
          "bbox": {
            "x1": 150.5,
            "y1": 200.3,
            "x2": 250.7,
            "y2": 400.9,
            "width": 100.2,
            "height": 200.6
          }
        }
      ]
    }
  ]
}
```

### Visual Outputs
- **Annotated Images**: Photos with colored bounding boxes
- **Confidence Labels**: Object names with detection confidence
- **Color Coding**: Different colors for different object types

## Historical Photo Specific Features

### Priority Objects for Historical Photos
1. **Person** - Highest priority for genealogy/family history
2. **Horse** - Common in historical transportation
3. **Car** - Dating photos by vehicle types
4. **Bicycle** - Early transportation methods
5. **Dog/Cat** - Family pets in historical context
6. **Bird** - Wildlife and domestic animals
7. **Chair** - Furniture styles for period dating
8. **Book** - Educational/cultural context

### Detection Accuracy Notes
- **Modern objects** may have lower accuracy in historical photos
- **Vintage vehicles** might be detected as generic "car"
- **Period clothing** doesn't affect person detection
- **Black & white photos** work well with YOLO
- **Photo quality** affects detection confidence

## Technical Details

### Model Information
- **Architecture**: YOLOv8 (You Only Look Once)
- **Training Data**: COCO dataset (80 object classes)
- **Input Format**: RGB images, any resolution
- **Output**: Bounding boxes with confidence scores

### Performance Characteristics
| Model Size | Speed | Accuracy | File Size | Memory |
|------------|-------|----------|-----------|---------|
| Nano (n)   | Fast  | Good     | ~6MB      | Low     |
| Small (s)  | Medium| Better   | ~22MB     | Medium  |
| Medium (m) | Slower| High     | ~52MB     | High    |

### Supported Object Classes (Relevant to Historical Photos)
```python
Classes = {
    0: 'person',       14: 'bird',        57: 'couch',
    1: 'bicycle',      15: 'cat',         58: 'potted plant',
    2: 'car',          16: 'dog',         59: 'bed',
    3: 'motorcycle',   17: 'horse',       60: 'dining table',
    5: 'bus',          18: 'sheep',       62: 'tv',
    6: 'train',        19: 'cow',         73: 'book',
    7: 'truck',        56: 'chair',       74: 'clock'
    # ... and more
}
```

## Customization

### Adjust Confidence Threshold
```python
# In main() function
confidence_threshold = 0.3  # Lower = more detections, more false positives
confidence_threshold = 0.7  # Higher = fewer detections, more accurate
```

### Change Model Size
```python
# In main() function
model_size = 's'  # small model for better accuracy
model_size = 'm'  # medium model for high accuracy
```

### Modify Priority Objects
```python
# In YOLODetector class
self.priority_objects = ['person', 'horse', 'car', 'dog', 'book']
```

### Custom Object Filter
```python
# In YOLODetector class - modify relevant_classes dictionary
self.relevant_classes = {
    0: 'person',
    17: 'horse',
    2: 'car'
    # Add only objects you want to detect
}
```

## Installation Requirements

The script will automatically download the YOLO model on first run:
- YOLOv8n: ~6MB download
- YOLOv8s: ~22MB download
- Internet connection required for initial setup

## Troubleshooting

### Common Issues

**Model download fails**
- Check internet connection
- Ensure sufficient disk space
- Try smaller model size first

**No objects detected**
- Lower confidence threshold (0.3-0.4)
- Check if objects are in supported classes
- Verify image quality and resolution

**Memory errors**
- Use smaller model size ('n' instead of 'l')
- Process images individually
- Reduce image resolution before processing

**Slow processing**
- Use nano model ('n') for speed
- Disable visualizations: `create_visualizations=False`
- Process smaller batches

### Performance Tips
- **Batch processing**: Analyze multiple photos efficiently
- **Model caching**: YOLO model loads once per session
- **GPU acceleration**: Automatically used if available
- **Memory management**: Processes one image at a time

## Integration with Other Modules

### With Photo Clustering
```python
# Use YOLO results to enhance clustering features
# Filter photos by object types before clustering
# Group photos by detected objects
```

### With Quality Analysis
```python
# Combine quality grades with object detection
# Prioritize high-quality photos with important objects
# Filter low-quality photos before YOLO processing
```

## Historical Photo Use Cases

1. **Family History**: Find all photos with people
2. **Transportation History**: Track vehicles across time periods
3. **Animal Documentation**: Locate photos with horses, pets
4. **Lifestyle Analysis**: Identify furniture, household items
5. **Location Context**: Detect outdoor vs indoor scenes
6. **Period Dating**: Use objects to estimate photo age
7. **Collection Organization**: Sort by object types automatically