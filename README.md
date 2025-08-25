# TestPhotos - Historical Photo Analysis Tool 📸

An AI-powered tool for analyzing historical photographs with automatic symbol detection, text extraction, and semantic search capabilities.

## 🎯 Quick Start (3 Simple Commands)

### 1. Setup
```bash
git clone [repository-url]
cd TestPhotos
pip install -r requirements.txt
```

### 2. Add Photos
```bash
# Put your photos in the sample_photos/ folder
cp /path/to/your/photos/* sample_photos/
```

### 3. Run Analysis
```bash
python run.py photo_processing  # Process photos with AI
python run.py dashboard         # Search and explore results
python run.py                   # Interactive menu
```

That's it! 🎉

## 🖼️ What This Tool Does

**TestPhotos** analyzes historical photos using multiple AI models to detect:

- 👥 **People & Objects** - YOLO detection of people, clothing, furniture
- 🏛️ **Scenes & Settings** - Indoor/outdoor classification, backgrounds  
- ✡️ **Jewish Symbols** - Stars of David, menorahs, religious items
- 卐 **Nazi Symbols** - Swastikas, Nazi insignia, uniforms (for historical documentation)
- 📝 **Text Extraction** - Hebrew and German text with automatic translation
- ⚠️ **Violence Assessment** - Detection of weapons, violence indicators
- 🎨 **Photo Colorization** - Converts black & white photos to color
- 🔍 **Smart Search** - Category filters and natural language search

## 📁 How It Works

```
Input Photos → AI Analysis → Searchable Database
     ↓              ↓              ↓
sample_photos/ → Processing → main_app_outputs/
```

### What Gets Created:
- **Colorized Photos** - Enhanced versions of B&W images
- **JSON Analysis** - Detailed AI analysis for each photo  
- **CSV Database** - Searchable data for all photos
- **HTML Reports** - Visual search results with thumbnails
- **Text Reports** - Filtered results by category

## 🚀 Installation & Setup

### Option 1: Using Cursor (Recommended for AI Users)
1. **Open Cursor IDE**
2. **Clone/Open this repository**
3. **Ask Cursor AI:**
   ```
   Set up TestPhotos for me - install dependencies and explain how to use it
   ```
4. **Follow the AI instructions**

### Option 2: Manual Installation
```bash
# Clone repository
git clone [repository-url]
cd TestPhotos

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python run.py --help
```

### System Requirements
- **Python 3.8+**
- **4GB+ RAM** (8GB+ recommended)
- **Internet connection** (for downloading AI models)
- **Optional: GPU** for faster processing

## 🎮 Usage Guide

### Command-Line Interface
```bash
python run.py photo_processing  # Analyze photos with AI
python run.py dashboard         # Search processed photos
python run.py                   # Interactive menu with help
```

### Interactive Menu
Run `python run.py` for a user-friendly menu:

```
🖼️  TestPhotos - Historical Photo Analysis Tool
====================================
📊 Current Status:
   📁 Photos in sample_photos/: 25
   ✅ Photos processed: 12
   🆕 New photos: 13

🎮 Available Commands:
1️⃣  Process Photos - Run AI analysis
2️⃣  Search Dashboard - Explore results
3️⃣  Help & Info - Detailed help
0️⃣  Exit
```

### Processing Options
When running `process`, you can choose:
- **1 random photo** - Quick test
- **5 random photos** - Sample analysis  
- **All photos** - Full collection analysis

### Search Dashboard
Access your analyzed photos through:

1. **Category Search** - Filter by:
   - Nazi symbols present
   - Jewish symbols present
   - Hebrew/German text present
   - Violence indicators
   - Indoor/outdoor photos

2. **Semantic Search** - Natural language queries like:
   - "family photos with children"
   - "military uniforms"
   - "religious ceremonies"
   - "documents with text"

## 📊 Example Results

### Processing Output
```
✅ Processing complete!
📊 Processed 25 images
📁 Results saved to: main_app_outputs/
🔍 Run 'python run.py dashboard' to explore
```

### Search Results
```
🏷️ Category Search: Nazi Symbols
📋 Found 3 photos:
   • photo_001.jpg - Nazi uniform, swastika armband
   • photo_015.jpg - Military ceremony with Nazi flags
   • photo_027.jpg - Document with Nazi letterhead
```

## 🗂️ Project Structure

```
TestPhotos/
├── run.py                      # Main entry point
├── sample_photos/              # Your input photos
├── main_app/                   # AI processing modules
│   ├── main_pipeline.py        # Photo processing
│   ├── dashboard_pipeline.py   # Search functionality
│   └── modules/                # AI components
├── main_app_outputs/           # Analysis results
│   ├── colorized/              # Colorized photos
│   ├── results/                # JSON/CSV data
│   └── data_search_dashord/    # Search results
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔧 Configuration Options

### Processing Settings
```bash
# Custom processing (advanced)
python main_app/main_pipeline.py \
  --input_dir sample_photos \
  --output_dir custom_output \
  --yolo_model_size s \
  --confidence_yolo 0.4
```

### Model Options
- **YOLO Model**: `n` (fastest) → `s` → `m` → `l` → `x` (most accurate)
- **Confidence Levels**: 0.1 (sensitive) → 0.5 (balanced) → 0.9 (strict)

## 🤖 AI Models Used

- **🎨 Colorization**: OpenCV deep learning models
- **👁️ Object Detection**: YOLOv8 (Ultralytics)
- **🧠 Scene Analysis**: CLIP (OpenAI)
- **📝 Text Extraction**: Tesseract OCR  
- **💬 Content Analysis**: Groq LLM (GPT-style)
- **🔍 Semantic Search**: Sentence Transformers

## 🚨 Important Notes

### Historical Content Warning
This tool is designed for **historical research and documentation**. It may detect sensitive historical symbols and content. Use responsibly and in appropriate academic/research contexts.

### Privacy & Data
- All processing happens locally on your machine
- No photos are uploaded to external services (except LLM analysis if API key provided)
- Results are stored in your `main_app_outputs/` folder

### Performance Tips
- **First run**: Will download AI models (~2GB total)
- **GPU acceleration**: Install CUDA for 5-10x speedup
- **Memory**: Close other apps for large photo collections
- **Batch processing**: Process photos in groups of 10-20 for best performance

## 🐛 Troubleshooting

### Common Issues

**No photos found:**
```bash
# Make sure photos are in the right place
ls sample_photos/
# Should show your .jpg, .png files
```

**Installation errors:**
```bash
# Update pip first
pip install --upgrade pip
# Install individually if needed
pip install torch torchvision ultralytics opencv-python
```

**Out of memory:**
```bash
# Use smaller model
python run.py photo_processing  # Choose "1 random photo" first
```

**No AI results:**
```bash
# Check internet connection (models need to download)
# Lower confidence thresholds in advanced settings
```

### Get Help

1. **Run interactive help**: `python run.py` → Choose option 3
2. **Check CURSOR.md** for AI-specific instructions
3. **Ask Cursor AI**: "Help me debug TestPhotos issues"

## 🎓 For Cursor/AI Users

This project is optimized for AI assistance. When using Cursor:

1. **Ask for setup help**: "Set up and explain TestPhotos"
2. **Debug issues**: "TestPhotos isn't working, help debug"
3. **Customize analysis**: "Modify TestPhotos to focus on [specific type] photos"
4. **Understand results**: "Explain these TestPhotos analysis results"

See `CURSOR.md` for detailed AI prompts and instructions.

## 📜 License & Ethics

- **Academic/Research Use**: Freely available
- **Historical Documentation**: Appropriate for museums, archives, research
- **Sensitive Content**: Handle Nazi/war imagery responsibly and in proper context
- **Privacy**: Respect privacy of individuals in historical photos

## 🙏 Credits

Built with: OpenAI CLIP, Ultralytics YOLO, OpenCV, Tesseract OCR, Groq AI, Sentence Transformers

---

**Ready to explore your historical photo collection with AI? Start with `python run.py`!** 🚀📸