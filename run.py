#!/usr/bin/env python3
"""
TestPhotos - Historical Photo Analysis Tool

Simple commands:
- python run.py photo_processing  # Process photos with AI analysis
- python run.py dashboard         # Search and explore processed photos  
- python run.py                   # Interactive menu
"""

import sys
import os
from pathlib import Path
import json
import csv
from typing import Set

# Add main_app to path
sys.path.insert(0, str(Path(__file__).parent / "main_app"))

try:
    from main_pipeline import run_main_pipeline
    from dashboard_pipeline import DashboardPipeline
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)


def get_processed_photos() -> Set[str]:
    """Get list of already processed photos from CSV data."""
    processed = set()
    csv_path = Path("main_app_outputs") / "results" / "data_full.csv"
    
    if csv_path.exists():
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    orig_path = row.get('original_path', '')
                    if orig_path:
                        # Extract just the filename
                        processed.add(Path(orig_path).name)
        except Exception:
            pass
    
    return processed


def get_available_photos() -> Set[str]:
    """Get list of available photos in sample_photos directory."""
    photos = set()
    photo_dir = Path("sample_photos")
    
    if photo_dir.exists():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", 
                ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF", ".TIF"}
        for file_path in photo_dir.iterdir():
            if file_path.suffix in exts and file_path.is_file():
                photos.add(file_path.name)
    
    return photos


def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    try:
        import torch  # type: ignore
        import torchvision  # type: ignore
        import ultralytics  # type: ignore
        import cv2  # type: ignore
        import numpy  # type: ignore
        import PIL  # type: ignore
        return True
    except ImportError:
        return False


def run_process_command():
    """Run the photo processing pipeline."""
    print("🔍 TestPhotos - Photo Processing Pipeline")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Missing required dependencies. Please run:")
        print("   pip install -r requirements.txt")
        return
    
    # Check for photos
    available_photos = get_available_photos()
    if not available_photos:
        print("❌ No photos found in sample_photos/ directory")
        print("   Please add photos to the sample_photos/ folder")
        return
    
    print(f"📸 Found {len(available_photos)} photos in sample_photos/")
    
    # Check for already processed photos
    processed_photos = get_processed_photos()
    new_photos = available_photos - processed_photos
    
    if processed_photos:
        print(f"✅ {len(processed_photos)} photos already processed")
        
        if new_photos:
            print(f"🆕 {len(new_photos)} new photos found:")
            for photo in sorted(new_photos):
                print(f"   • {photo}")
        else:
            print("ℹ️  All photos are already processed")
            choice = input("\n🤔 Process photos again? (y/N): ").strip().lower()
            if choice not in ['y', 'yes']:
                print("👋 Skipping processing")
                return
    
    print("\n🚀 Starting photo processing...")
    print("   This may take several minutes depending on the number of photos")
    
    try:
        result = run_main_pipeline(
            input_dir="sample_photos",
            output_dir="main_app_outputs", 
            models_dir=os.path.join("opencv_analysis", "models"),
            ab_boost=1.0,
            yolo_model_size="s",
            clip_model_name="ViT-L/14@336px",
            confidence_yolo=0.4,
            confidence_clip=0.3
        )
        
        print("\n✅ Processing complete!")
        print(f"📊 Processed {result.get('num_images', 0)} images")
        print(f"📁 Results saved to: {result.get('output_dir', 'main_app_outputs')}")
        print("🔍 Run 'python run.py dashboard' to explore results")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        print("Please check the error messages above for details")


def run_dashboard_command():
    """Run the dashboard search interface."""
    print("🔍 TestPhotos - Search Dashboard") 
    print("=" * 50)
    
    # Check if data exists
    csv_path = Path("main_app_outputs") / "results" / "data_full.csv"
    if not csv_path.exists():
        print("❌ No processed data found")
        print("   Run 'python run.py process' first to analyze photos")
        return
    
    processed_photos = get_processed_photos()
    print(f"📊 {len(processed_photos)} processed photos available")
    
    dashboard = DashboardPipeline()
    
    print("\n📋 Search Options:")
    print("1️⃣  Category Search - Filter by symbols, violence, text, etc.")
    print("2️⃣  Semantic Search - Natural language search")
    print("3️⃣  Build Embeddings - Prepare for semantic search")
    print("0️⃣  Exit")
    
    while True:
        choice = input("\n🎯 Choose search type (1-3, 0 to exit): ").strip()
        
        if choice == "1":
            print("\n🏷️  Running Category Search...")
            dashboard.run_category_search()
            break
        elif choice == "2":
            print("\n🔤 Running Semantic Search...")
            dashboard.run_semantic_search()
            break
        elif choice == "3":
            print("\n🧠 Building Embeddings...")
            dashboard.build_embeddings()
            print("✅ Embeddings ready! Now you can use semantic search.")
            break
        elif choice == "0":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 0")


def show_interactive_menu():
    """Show interactive menu for users."""
    print("🖼️  TestPhotos - Historical Photo Analysis Tool")
    print("=" * 60)
    print("📖 This tool analyzes historical photos using AI to detect:")
    print("   • Objects, people, and scenes")
    print("   • Historical symbols and text")
    print("   • Violence indicators")
    print("   • Hebrew and German text with translation")
    print()
    
    # Show status
    available_photos = get_available_photos()
    processed_photos = get_processed_photos()
    
    print("📊 Current Status:")
    print(f"   📁 Photos in sample_photos/: {len(available_photos)}")
    print(f"   ✅ Photos processed: {len(processed_photos)}")
    print(f"   🆕 New photos: {len(available_photos - processed_photos)}")
    print()
    
    print("🎮 Available Commands:")
    print("1️⃣  Process Photos - Run AI analysis on photos")
    print("2️⃣  Search Dashboard - Explore processed photos") 
    print("3️⃣  Help & Info - Show detailed help")
    print("0️⃣  Exit")
    
    while True:
        choice = input("\n🎯 What would you like to do? (1-3, 0 to exit): ").strip()
        
        if choice == "1":
            run_process_command()
            break
        elif choice == "2":
            run_dashboard_command()
            break
        elif choice == "3":
            show_help()
            break
        elif choice == "0":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 0")


def show_help():
    """Show detailed help information."""
    print("\n📚 TestPhotos Help & Information")
    print("=" * 50)
    
    print("\n🎯 Quick Start:")
    print("1. Add photos to the sample_photos/ folder")
    print("2. Run: python run.py process")
    print("3. Run: python run.py dashboard")
    
    print("\n📁 Directory Structure:")
    print("   sample_photos/          - Put your photos here")
    print("   main_app_outputs/       - Analysis results")
    print("   ├── colorized/          - Colorized versions")
    print("   ├── results/            - JSON and CSV data")
    print("   └── data_search_dashord/ - Search results")
    
    print("\n🔧 Commands:")
    print("   python run.py                  - Interactive menu")
    print("   python run.py photo_processing - Process photos")
    print("   python run.py dashboard        - Search interface")
    
    print("\n🤖 AI Analysis Features:")
    print("   • Photo colorization (black & white → color)")
    print("   • Object detection (people, objects, scenes)")
    print("   • Symbol recognition (Jewish, Nazi symbols)")
    print("   • Text extraction and translation")
    print("   • Violence assessment")
    print("   • Historical context analysis")
    
    print("\n🔍 Search Features:")
    print("   • Category filters (symbols, violence, text)")
    print("   • Natural language search")
    print("   • HTML galleries and text reports")
    
    print("\n⚙️  Requirements:")
    print("   • Python 3.8+")
    print("   • GPU recommended (but not required)")
    print("   • Internet connection for some models")
    
    input("\n📖 Press Enter to continue...")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command in ["photo_processing", "process"]:
            run_process_command()
        elif command in ["dashboard", "run_dashboard"]:
            run_dashboard_command()
        elif command in ["help", "-h", "--help"]:
            show_help()
        else:
            print(f"❌ Unknown command: {command}")
            print("📖 Available commands: photo_processing, dashboard, help")
            print("   Or run without arguments for interactive menu")
    else:
        show_interactive_menu()


if __name__ == "__main__":
    main()