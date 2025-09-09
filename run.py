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
import subprocess
from typing import Set

# Default input directory (user collection)
DEFAULT_INPUT_DIR = os.path.join("photo_collections", "project Photography")

# Add project root and main_app to path
project_root = str(Path(__file__).parent)
main_app_path = str(Path(__file__).parent / "main_app")

# Insert project root first, then main_app
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if main_app_path not in sys.path:
    sys.path.insert(1, main_app_path)  # Insert after project root

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
    """Get list of available photos in the default input directory (recursive)."""
    photos = set()
    photo_dir = Path(DEFAULT_INPUT_DIR)
    
    if photo_dir.exists():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
                ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF", ".TIF"}
        for file_path in photo_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in exts:
                photos.add(file_path.name)
    
    return photos


def count_photos_in_directory(dir_path: str) -> int:
    """Count photos recursively under the given directory."""
    p = Path(dir_path)
    if not p.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF", ".TIF"}
    try:
        return sum(1 for fp in p.rglob("*") if fp.is_file() and fp.suffix in exts)
    except Exception:
        return 0


def count_processed_photos() -> int:
    """Count processed photos based on data_full.csv rows (excluding header)."""
    csv_path = Path("main_app_outputs") / "results" / "data_full.csv"
    if not csv_path.exists():
        return 0
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            import csv as _csv
            rdr = _csv.DictReader(f)
            return sum(1 for _ in rdr)
    except Exception:
        return 0


def run_status_command():
    """Print status of total vs processed photos."""
    print("📊 TestPhotos - Status")
    print("=" * 50)
    total = count_photos_in_directory(DEFAULT_INPUT_DIR)
    processed = count_processed_photos()
    print(f"📁 Source directory: {DEFAULT_INPUT_DIR}")
    print(f"🧮 Total photos found (recursive): {total}")
    print(f"✅ Processed photos (in results): {processed}")
    remaining = max(0, total - processed)
    print(f"⏳ Remaining (approx): {remaining}")


def run_clean_plan_command():
    """Create a deletion plan for cleaning outputs without deleting anything."""
    print("🧹 TestPhotos - Clean Plan (no deletion)")
    print("=" * 50)
    root = Path("main_app_outputs")
    targets = [
        root / "colorized",
        root / "processed_photos",
        root / "results",
        root / "data_search_dashord",
        root / "gallery_images"
    ]
    out = root / "files_to_delete.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out, 'w', encoding='utf-8') as f:
        for t in targets:
            if not t.exists():
                continue
            for p in t.rglob("*"):
                if p.is_file():
                    f.write(f"{p}\tReason: user requested clean pipeline reset\n")
                    count += 1
    print(f"📝 Plan written: {out}")
    print(f"🗂️  Files listed for deletion: {count}")


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
        print(f"❌ No photos found in {DEFAULT_INPUT_DIR} directory")
        print(f"   Please add photos under '{DEFAULT_INPUT_DIR}'")
        return
    
    print(f"📸 Found {len(available_photos)} photos in {DEFAULT_INPUT_DIR}")
    
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
            input_dir=DEFAULT_INPUT_DIR,
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
    print("1️⃣  Category Search - Filter by symbols, violence, text, etc. (CSV)")
    print("2️⃣  Semantic Search - Natural language search (MiniLM)")
    print("3️⃣  Build Embeddings - Prepare for semantic search")
    print("4️⃣  ChromaDB Semantic Search - Vector database search")
    print("5️⃣  ChromaDB Category Search - Fast metadata filtering")
    print("6️⃣  ChromaDB Stats - Show vector database statistics")
    print("7️⃣  Pinecone Semantic Search - Cloud vector database search")
    print("8️⃣  Pinecone Stats - Show cloud database statistics")
    print("9️⃣  Migrate to Pinecone - Move data to cloud")
    print("0️⃣  Exit")
    
    while True:
        choice = input("\n🎯 Choose search type (1-9, 0 to exit): ").strip()
        
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
        elif choice == "4":
            print("\n🔍 Running ChromaDB Semantic Search...")
            dashboard.run_chroma_semantic_search()
            break
        elif choice == "5":
            print("\n🏷️  Running ChromaDB Category Search...")
            dashboard.run_chroma_category_search()
            break
        elif choice == "6":
            print("\n📊 ChromaDB Statistics...")
            dashboard.get_chroma_stats()
            break
        elif choice == "7":
            print("\n🔍 Running Pinecone Semantic Search...")
            dashboard.run_pinecone_semantic_search()
            break
        elif choice == "8":
            print("\n📊 Pinecone Statistics...")
            dashboard.get_pinecone_stats()
            break
        elif choice == "9":
            print("\n🔄 Migrating to Pinecone...")
            import subprocess
            result = subprocess.run([sys.executable, "migrate_to_pinecone.py"], 
                                  capture_output=False, text=True)
            break
        elif choice == "0":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-9 or 0")


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
    print(f"   📁 Photos in {DEFAULT_INPUT_DIR}: {len(available_photos)}")
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


def run_chroma_search_command():
    """Run ChromaDB-based search interface."""
    print("🔍 TestPhotos - ChromaDB Vector Search")
    print("=" * 50)
    
    # Import ChromaDB components
    try:
        from main_app.modules.chroma_handler import create_chroma_handler
        from main_app.dashboard_pipeline import DashboardPipeline
    except ImportError as e:
        print(f"❌ ChromaDB components not available: {e}")
        print("   Install ChromaDB with: pip install chromadb")
        return
    
    # Initialize ChromaDB handler
    chroma_handler = create_chroma_handler()
    if not chroma_handler:
        print("❌ ChromaDB not available or not initialized")
        print("   Make sure ChromaDB is installed and you have processed photos")
        return
    
    # Get stats
    stats = chroma_handler.get_collection_stats()
    print(f"📊 ChromaDB Database contains {stats.get('total_photos', 0)} photos")
    
    if stats.get('total_photos', 0) == 0:
        print("⚠️  No photos found in ChromaDB")
        print("   Run photo processing first to populate the database")
        return
    
    # Initialize dashboard for ChromaDB methods
    dashboard = DashboardPipeline()
    
    print("\n🔍 ChromaDB Search Options:")
    print("1️⃣  Semantic Search - Natural language vector search")
    print("2️⃣  Category Search - Fast metadata filtering")
    print("3️⃣  Statistics - Show database information")
    print("0️⃣  Exit")
    
    while True:
        choice = input("\n🎯 Choose search type (1-3, 0 to exit): ").strip()
        
        if choice == "1":
            print("\n🔍 ChromaDB Semantic Search...")
            dashboard.run_chroma_semantic_search()
            break
        elif choice == "2":
            print("\n🏷️  ChromaDB Category Search...")
            dashboard.run_chroma_category_search()
            break
        elif choice == "3":
            print("\n📊 ChromaDB Statistics...")
            dashboard.get_chroma_stats()
            break
        elif choice == "0":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-3 or 0")


def run_migrate_command():
    """Run migration of existing CSV data to ChromaDB."""
    print("🔄 TestPhotos - Migrate Data to ChromaDB")
    print("=" * 50)
    
    # Check if CSV data exists
    csv_path = Path("main_app_outputs") / "results" / "data_full.csv"
    if not csv_path.exists():
        print("❌ No CSV data found to migrate")
        print("   Run photo processing first to create data")
        return
    
    print("⚠️  This will migrate all existing photo analysis data to ChromaDB")
    print("   This process may take a few minutes depending on data size")
    
    confirm = input("\n🤔 Continue with migration? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("👋 Migration cancelled")
        return
    
    # Run migration script
    print("\n🔄 Starting migration...")
    try:
        # Import and run migration
        python_cmd = sys.executable
        script_path = Path("migrate_to_chroma.py")
        if script_path.exists():
            result = subprocess.run([python_cmd, str(script_path)], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Migration completed successfully!")
                print(result.stdout)
            else:
                print("❌ Migration failed:")
                print(result.stderr)
        else:
            print("❌ Migration script not found: migrate_to_chroma.py")
            print("   Please create the migration script first")
    except Exception as e:
        print(f"❌ Migration error: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command in ["photo_processing", "process"]:
            run_process_command()
        elif command in ["dashboard", "run_dashboard"]:
            run_dashboard_command()
        elif command in ["chroma_search", "chroma"]:
            run_chroma_search_command()
        elif command in ["migrate", "migrate_chroma"]:
            run_migrate_command()
        elif command in ["status", "stats"]:
            run_status_command()
        elif command in ["clean", "clean_plan"]:
            run_clean_plan_command()
        elif command in ["help", "-h", "--help"]:
            show_help()
        else:
            print(f"❌ Unknown command: {command}")
            print("📖 Available commands: photo_processing, dashboard, chroma_search, migrate, status, clean, help")
            print("   Or run without arguments for interactive menu")
    else:
        show_interactive_menu()


if __name__ == "__main__":
    main()