# OpenCV Photo Analysis

This module analyzes historical photos for quality metrics and face detection using OpenCV computer vision techniques.

## Features

### Photo Quality Grading (A-D)
Comprehensive quality assessment based on:

- **Sharpness**: Laplacian variance to detect blur and focus quality
- **Brightness**: Average luminance levels for proper exposure
- **Contrast**: Standard deviation analysis for image clarity
- **Noise Level**: High-frequency content detection for image cleanliness
- **Exposure Analysis**: Detection of over/under-exposed regions

### Face Detection
- Uses OpenCV Haar Cascade classifier for face detection
- Counts total faces per image
- Provides bounding box coordinates for each detected face
- Works with historical photos and group photos

## Files

- `photo_analyzer.py` - Main analysis script
- `photo_analysis.json` - Output results (generated)
- `README.md` - This documentation

## Usage

### Basic Analysis
```bash
cd opencv_analysis
python3 photo_analyzer.py
```

This will:
1. Analyze all photos in `../sample_photos/` directory
2. Generate quality grades (A, B, C, D) for each photo
3. Count faces in each image
4. Save results to `photo_analysis.json`

### Output Format

The script creates a JSON file with detailed analysis:

```json
{
  "summary": {
    "total_photos": 68,
    "grade_distribution": {"A": 31, "B": 20, "C": 15, "D": 2},
    "average_face_count": 4.87
  },
  "photos": [
    {
      "filename": "photo.jpg",
      "quality_grade": "A",
      "quality_score": 92,
      "quality_metrics": {
        "sharpness": 1719.02,
        "brightness": 178.5,
        "contrast": 67.84,
        "noise_level": 6.99,
        "exposure": {
          "overexposed_ratio": 0.38,
          "underexposed_ratio": 0.009,
          "well_exposed_ratio": 0.49
        }
      },
      "face_count": 3,
      "face_locations": [[x, y, width, height], ...]
    }
  ]
}
```

## Quality Grading System

### Grade A (85-100 points)
- Excellent sharpness (>500)
- Good contrast (30-80)
- Proper brightness (80-180)
- Low noise (<5)
- Well-balanced exposure

### Grade B (70-84 points)
- Good sharpness (200-500)
- Decent contrast (20-100)
- Acceptable brightness (60-200)
- Moderate noise (5-10)
- Reasonable exposure

### Grade C (55-69 points)
- Fair sharpness (100-200)
- Limited contrast (15-120)
- Sub-optimal brightness (40-220)
- Higher noise (10-15)
- Poor exposure balance

### Grade D (<55 points)
- Poor sharpness (<100)
- Very low contrast (<15)
- Extreme brightness issues
- High noise (>15)
- Severe exposure problems

## Face Detection

- **Algorithm**: Haar Cascade Classifier (frontalface_default)
- **Minimum Face Size**: 30x30 pixels
- **Scale Factor**: 1.1 (multi-scale detection)
- **Min Neighbors**: 5 (reduces false positives)

### Face Detection Accuracy
- Works well with frontal faces
- May detect false positives in textured backgrounds
- Historical photos may have varying accuracy due to image quality
- Group photos typically show higher face counts

## Technical Details

### Image Processing Pipeline
1. Load image using OpenCV
2. Convert color spaces (BGR→RGB, RGB→Grayscale)
3. Apply various filters and transformations
4. Extract numerical features
5. Apply scoring algorithms
6. Generate letter grades

### Dependencies
- **OpenCV**: Computer vision and image processing
- **NumPy**: Numerical computations
- **JSON**: Output formatting
- **Pathlib**: File system operations

## Customization

### Modify Grading Thresholds
Edit the `grade_quality()` method scoring system:
```python
# Sharpness scoring (40% weight)
if sharpness > 500:
    score += 40
# ... modify thresholds as needed
```

### Face Detection Sensitivity
Adjust parameters in `count_faces()` method:
```python
faces = self.face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,     # Increase for faster, less accurate
    minNeighbors=5,      # Increase to reduce false positives
    minSize=(30, 30),    # Minimum face size
)
```

### Change Input Directory
Modify the default path in `main()`:
```python
results = analyzer.analyze_directory('../sample_photos', 'photo_analysis.json')
```

## Troubleshooting

### Common Issues

**No images found**
- Ensure photos are in `../sample_photos/` directory
- Check supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif

**Face cascade not found**
- OpenCV installation may be incomplete
- Haar cascade files should be included with opencv-python

**Memory issues with large images**
- Script automatically processes images as-is
- Consider resizing very large images beforehand

**False positive faces**
- High face counts may indicate false detections
- Adjust `minNeighbors` parameter to reduce sensitivity

### Performance Notes
- Processing time: ~1-2 seconds per image
- Memory usage: Minimal (processes one image at a time)
- Output file size: ~2-10KB per analyzed photo

## Integration

This module works alongside:
- **Photo Clustering** (`../photo_clustering/`) - Use quality grades to filter photos before clustering
- **Main Project** - Quality analysis can inform photo selection and organization