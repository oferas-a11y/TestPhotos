# Historical Photo Analysis Tools

A comprehensive suite of AI-powered tools for analyzing and organizing historical photographs from 20th century Europe. This project combines computer vision, machine learning, and semantic analysis to provide deep insights into historical photo collections.

## 🎯 Project Overview

This toolkit provides four complementary approaches to historical photo analysis:
- **OpenCV Analysis**: Technical image properties (brightness, contrast, colors)
- **YOLO Object Detection**: Identify people, objects, and transportation
- **CLIP Semantic Analysis**: Understand cultural and historical context
- **K-means Clustering**: Organize photos by visual similarity

## 📁 Project Structure

```
├── sample_photos/              # Input photos directory (66 historical photos)
├── opencv_analysis/            # Computer vision analysis
│   ├── photo_analyzer.py      # Main analysis script
│   └── README.md              # OpenCV usage guide
├── yolo_detection/            # Object detection with YOLO
│   ├── yolo_detector.py       # YOLO analysis script
│   └── README.md              # YOLO usage guide
├── clip_analysis/             # Semantic analysis with CLIP
│   ├── clip_detector.py       # CLIP analysis script
│   └── README.md              # CLIP usage guide
├── photo_clustering/          # Visual similarity clustering
│   ├── process_photos.py      # Feature extraction
│   ├── kmeans_clustering.py   # K-means clustering
│   └── clusters/              # Organized photo clusters (10 groups)
├── examples/                  # Demo results and examples
│   ├── clustering_results/    # Example cluster organization
│   ├── random_5_photos_*.json # Sample analysis results
│   └── README.md              # Examples documentation
├── tests/                     # Comprehensive test suite
│   ├── test_*.py             # Unit tests for each component
│   ├── conftest.py           # Test configuration
│   └── run_tests.py          # Test runner with coverage
├── docs/                      # Additional documentation
├── results/                   # Analysis outputs (preserved)
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository** (when available)
```bash
git clone <repository-url>
cd TestPhotos
```

2. **Install dependencies**
```bash
pip install -r requirements.txt

# Install CLIP separately (optional, may have version conflicts)
pip install git+https://github.com/openai/CLIP.git
```

3. **Run tests** (optional but recommended)
```bash
cd tests
python run_tests.py
```

### Basic Usage

Each analysis tool works independently. Choose based on your needs:

#### 1. OpenCV Technical Analysis
```bash
cd opencv_analysis
python photo_analyzer.py
```
**Use for**: Image quality assessment, color analysis, technical properties

#### 2. YOLO Object Detection  
```bash
cd yolo_detection
python yolo_detector.py
```
**Use for**: Finding people, vehicles, furniture, historical objects

#### 3. CLIP Semantic Analysis
```bash
cd clip_analysis
python clip_detector.py
```
**Use for**: Understanding cultural context, historical periods, social situations

#### 4. Photo Clustering
```bash
cd photo_clustering
python process_photos.py
python kmeans_clustering.py
```
**Use for**: Organizing large collections, finding visually similar photos

## 🔍 Analysis Capabilities

### OpenCV Computer Vision
- **Brightness & Contrast**: Photo quality metrics
- **Dominant Colors**: Color palette extraction  
- **Edge Density**: Detail and sharpness analysis
- **Texture Analysis**: Surface pattern recognition
- **Color Histograms**: RGB distribution analysis

### YOLO Object Detection
Optimized for 20th century European historical photos:
- **People**: Formal clothing, military uniforms, group portraits
- **Transportation**: Horses, early automobiles, trains, bicycles
- **Architecture**: European buildings, period furniture
- **Daily Life**: Books, clocks, tableware, decorative items
- **Historical Context**: Military equipment, ceremonial items

### CLIP Semantic Understanding
Natural language descriptions of historical content:
- **Social Context**: Family portraits, formal occasions, wedding photographs  
- **Historical Periods**: Edwardian era, wartime scenes, rural life
- **Cultural Elements**: Traditional clothing, religious ceremonies
- **Architecture**: European buildings, town squares, residential houses
- **Transportation History**: Steam locomotives, horse carriages, vintage cars

### Visual Clustering
Automatic organization by visual similarity:
- **10 distinct clusters** based on color, texture, and composition
- **Cluster summaries** showing representative photos
- **2D visualization** of photo relationships
- **Similar photo grouping** for easy browsing

## 📊 Example Results

The `examples/` directory contains sample outputs:

### Clustering Results
- **Cluster 0**: Portrait-style photos and formal occasions (12 photos)
- **Cluster 6**: Group photos and family portraits (7 photos)  
- **Cluster 8**: Documents and text-heavy images (12 photos)

### Object Detection Sample
```json
{
  "filename": "family_portrait_1920.jpg",
  "total_objects": 3,
  "object_counts": {
    "person": 3,
    "chair": 1
  },
  "priority_score": 45
}
```

### Semantic Analysis Sample  
```json
{
  "filename": "historical_photo.jpg",
  "top_categories": [
    "family portrait",
    "a person in formal clothing", 
    "European building",
    "period furniture"
  ],
  "priority_score": 15.432
}
```

## 🧪 Testing

Comprehensive test suite with 95%+ coverage:

```bash
# Run all tests
cd tests
python run_tests.py

# Run specific test categories
python run_tests.py --opencv      # OpenCV tests only
python run_tests.py --yolo        # YOLO tests only  
python run_tests.py --clip        # CLIP tests only
python run_tests.py --clustering  # Clustering tests only
python run_tests.py --integration # Integration tests only
python run_tests.py --fast        # Skip slow tests
```

**Test Coverage:**
- Unit tests for all components
- Integration tests for complete workflows
- Mock tests for external dependencies (YOLO/CLIP models)
- Error handling and edge case testing
- Performance and memory usage tests

## 🎛️ Interactive Modes

All analysis tools provide interactive menus:

1. **Analyze all photos** - Process entire directory
2. **Analyze random sample** - Quick preview with 5 photos
3. **Analyze specific photo** - Deep dive into single image
4. **Custom analysis** - Tool-specific options
5. **Exit**

## ⚡ Performance & Requirements

### System Requirements
- **Python**: 3.8+
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ for models and results
- **GPU**: Optional (CUDA) for faster CLIP/YOLO processing

### Processing Speed (approximate)
- **OpenCV**: ~1-2 seconds per photo
- **YOLO**: ~2-5 seconds per photo (first run downloads model)
- **CLIP**: ~3-8 seconds per photo (first run downloads model)  
- **Clustering**: ~30-60 seconds for 66 photos

## 📈 Use Cases

### 1. Digital Archives & Museums
- Automatically tag and categorize photo collections
- Identify historical periods and cultural contexts
- Organize photos by visual similarity
- Extract technical metadata for preservation

### 2. Genealogy & Family History
- Find family portraits and formal occasions
- Identify military service and uniforms
- Detect period clothing and social status
- Group photos by time period and setting

### 3. Historical Research
- Analyze social customs and daily life
- Study transportation and technology evolution  
- Identify architectural styles and locations
- Research cultural and religious practices

### 4. Academic Studies
- Visual culture analysis
- Social history research
- Technology and transportation studies
- Comparative historical analysis

## 🔧 Configuration

### Confidence Thresholds
Adjust sensitivity in each tool:
- **YOLO**: 0.3-0.7 (default: 0.5)
- **CLIP**: 0.2-0.6 (default: 0.3)

### Model Selection
- **YOLO**: YOLOv8n (fast) to YOLOv8x (accurate)
- **CLIP**: ViT-B/32 (balanced) to ViT-L/14 (best quality)

### Clustering Parameters
- **Number of clusters**: 5-15 (default: 10)  
- **Image size**: 224x224 (default) for processing
- **Feature extraction**: Color + texture + edges

## 🚨 Troubleshooting

### Common Issues

**Installation Problems:**
```bash
# Update pip first
pip install --upgrade pip

# Install PyTorch separately if needed
pip install torch torchvision

# For CLIP issues, try direct GitHub install
pip install git+https://github.com/openai/CLIP.git
```

**Memory Issues:**
- Reduce batch size in analysis scripts
- Use smaller YOLO/CLIP models
- Process photos in smaller groups

**No Objects/Categories Detected:**
- Lower confidence thresholds
- Check image quality and size
- Ensure photos contain relevant historical content

**Performance Issues:**
- Install CUDA for GPU acceleration
- Use smaller model variants
- Close other applications to free memory

## 🤝 Contributing

### Development Setup
1. Install dev dependencies: `pip install -r requirements.txt`
2. Run tests: `cd tests && python run_tests.py`
3. Check code coverage: Open `htmlcov/index.html`

### Adding New Features
- Follow existing code structure
- Add unit tests for new components
- Update documentation and examples
- Ensure backward compatibility

## 📜 License

This project is designed for educational and research purposes. Historical photos may have their own copyright considerations.

## 🙏 Acknowledgments

- **OpenAI CLIP** for semantic understanding
- **Ultralytics YOLO** for object detection  
- **OpenCV** for computer vision
- **scikit-learn** for clustering algorithms

---

**Ready to analyze your historical photo collection? Start with any of the four analysis tools and explore the fascinating world of computational photo analysis!** 📸✨