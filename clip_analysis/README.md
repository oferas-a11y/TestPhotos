# CLIP Semantic Analysis for Historical Photos

This tool uses OpenAI's CLIP (Contrastive Language-Image Pre-training) model to perform semantic analysis on historical photographs from 20th century Europe. Unlike traditional object detection, CLIP understands the semantic meaning and context of images through natural language descriptions.

## Features

- **Semantic Understanding**: Analyzes photos using natural language descriptions instead of rigid object categories
- **Historical Context**: Specialized categories for 20th century European historical content
- **Interactive Analysis**: Multiple modes for analyzing single photos, random samples, or entire directories
- **Custom Queries**: Compare photos against your own text descriptions
- **Visualization**: Creates semantic analysis charts showing confidence scores
- **GPU Support**: Automatically uses CUDA if available for faster processing

## Installation

Ensure you have the CLIP dependencies installed:

```bash
pip install torch torchvision clip-by-openai
# Or install all requirements
pip install -r ../requirements.txt
```

## Usage

### Interactive Mode

Run the script for an interactive menu:

```bash
cd clip_analysis
python clip_detector.py
```

**Menu Options:**
1. **Analyze all photos** - Process entire directory with semantic analysis
2. **Analyze 5 random photos** - Quick sample analysis with visualizations
3. **Analyze specific photo** - Deep dive into a single image
4. **Custom text queries** - Compare photo against your own descriptions
5. **Exit**

### Historical Categories

The tool analyzes photos for these semantic categories:

**People & Social Context:**
- Person in formal clothing
- Person in military uniform
- Family portrait
- Wedding photograph
- Children playing

**Transportation & Technology:**
- Horse and carriage
- Steam locomotive
- Early automobile
- Vintage car
- Railway station

**Architecture & Buildings:**
- European building
- Church or cathedral
- Town square
- Residential house

**Daily Life Objects:**
- Period furniture
- Antique chair
- Vintage clock
- Old books
- Traditional tableware

**Cultural & Historical:**
- Military equipment
- Religious ceremony
- Traditional clothing
- Cultural celebration
- Musical instruments

## Output Files

### JSON Results
- `clip_analysis.json` - Complete directory analysis
- `single_photo_clip_[filename].json` - Individual photo results
- `random_5_photos_clip.json` - Random sample analysis

### Visualizations
- `semantic_visualizations/` directory contains analysis charts
- Shows top semantic categories with confidence scores
- Original image alongside results

## Example Results

```json
{
  "filename": "family_portrait_1920.jpg",
  "total_detections": 8,
  "priority_score": 15.432,
  "top_categories": [
    "family portrait",
    "a person in formal clothing", 
    "traditional clothing",
    "European building",
    "period furniture"
  ],
  "detections": [
    {
      "category": "family portrait",
      "confidence": 0.856,
      "historical_context": "Family relationships, social customs, formal photography practices"
    }
  ]
}
```

## Model Information

**Default Model**: ViT-B/32
- Balanced speed and accuracy
- Good for general historical photo analysis
- ~400MB download on first use

**Alternative Models** (change in code):
- `ViT-B/16` - Higher accuracy, slower
- `ViT-L/14` - Best accuracy, much slower
- `RN50` - ResNet-based, faster but less accurate

## Comparison with YOLO

| Feature | YOLO Detection | CLIP Semantic |
|---------|----------------|---------------|
| **Approach** | Object bounding boxes | Semantic understanding |
| **Categories** | Fixed object classes | Natural language descriptions |
| **Historical Context** | Objects present | Cultural/historical meaning |
| **Flexibility** | Limited to trained objects | Any text description |
| **Visualization** | Bounding boxes | Confidence charts |
| **Best For** | Counting specific objects | Understanding photo context |

## Use Cases

1. **Historical Research**: Understand the cultural and social context of photos
2. **Photo Organization**: Categorize photos by semantic content and historical period
3. **Genealogy**: Identify formal occasions, military service, social status
4. **Digital Archives**: Tag photos with meaningful historical descriptors
5. **Academic Research**: Analyze visual culture and social history

## Tips for Best Results

1. **Confidence Threshold**: Start with 0.3, adjust based on results
   - Lower (0.2): More detections, some false positives
   - Higher (0.5): Fewer but more confident detections

2. **Custom Queries**: Use specific historical terms
   - "Edwardian era clothing" vs "old clothes"
   - "Habsburg military uniform" vs "soldier"

3. **Model Selection**: Choose based on your needs
   - ViT-B/32: General use, good speed
   - ViT-L/14: Research requiring high accuracy

## Troubleshooting

**GPU Issues**: If CUDA errors occur, the model automatically falls back to CPU
**Memory Errors**: Use smaller model (RN50) or process fewer images at once
**No Detections**: Lower confidence threshold or check image quality

## Technical Details

- **Input**: RGB images (automatically converted)
- **Processing**: 224x224 pixel patches for ViT models
- **Output**: Cosine similarity scores between image and text embeddings
- **Performance**: ~1-3 seconds per image on modern GPU