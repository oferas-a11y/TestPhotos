#!/usr/bin/env python3
"""
Restore processing status from LLM files only.
Since the full_*.json files were deleted, we'll work with just the LLM files
to determine which photos had successful processing.
"""

import json
import csv
from pathlib import Path
from typing import Set, Dict, List, Any


def extract_successful_llm_photos() -> Set[str]:
    """Extract photos that have successful LLM results (not rate limited)."""
    results_dir = Path("main_app_outputs/results")
    successful_photos = set()
    rate_limited_photos = set()
    
    if not results_dir.exists():
        print("❌ Results directory not found")
        return set()
    
    # Find all LLM result files
    llm_files = list(results_dir.glob("llm_*_orig.json"))
    print(f"🔍 Found {len(llm_files)} LLM result files")
    
    for llm_file in llm_files:
        try:
            # Read LLM result
            with open(llm_file, 'r', encoding='utf-8') as f:
                llm_data = json.load(f)
            
            # Extract original filename from LLM file name
            # Format: llm_<filename>_orig.json -> reconstruct original path
            file_stem = llm_file.name.replace('llm_', '').replace('_orig.json', '')
            
            # Try to reconstruct the original filename
            # This is tricky since we don't have the full path info
            potential_paths = [
                f"photo_collections/project Photography/Anonymous German soldier – Warsaw ghetto/{file_stem}.jpg",
                f"photo_collections/project Photography/Plaszow from USHMM/{file_stem}.jpeg",
                f"photo_collections/project Photography/Grodno ghetto from USHMM/german photografers /PK JUDEN POLEN - GERMAN PROPAGANDA 1940 - 1942/{file_stem}.jpg",
            ]
            
            # Check if it's a valid LLM response (not an error)
            if 'error' in llm_data:
                error_msg = llm_data.get('error', 'unknown')
                if 'rate_limit' in str(error_msg).lower() or '429' in str(error_msg):
                    print(f"  ⚠️  Rate Limited: {file_stem}")
                    # For rate limited, we'll assume the photo was processed up to LLM stage
                    # Try to find the actual file
                    found_path = find_actual_photo_path(file_stem)
                    if found_path:
                        rate_limited_photos.add(found_path)
                else:
                    print(f"  ❌ LLM Error: {file_stem} - {error_msg}")
            else:
                # Valid LLM response
                print(f"  ✅ Success: {file_stem}")
                found_path = find_actual_photo_path(file_stem)
                if found_path:
                    successful_photos.add(found_path)
                    
        except Exception as e:
            print(f"  ❌ Error processing {llm_file}: {e}")
    
    print(f"\n📊 LLM Analysis Results:")
    print(f"   Successful LLM: {len(successful_photos)}")
    print(f"   Rate Limited: {len(rate_limited_photos)}")
    print(f"   Total with some processing: {len(successful_photos | rate_limited_photos)}")
    
    return successful_photos, rate_limited_photos


def find_actual_photo_path(file_stem: str) -> str:
    """Try to find the actual photo file path by searching."""
    base_dir = Path("photo_collections/project Photography")
    
    if not base_dir.exists():
        return ""
    
    # Common extensions
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # Search for files with matching stem
    for ext in extensions:
        # Direct search in subdirectories
        for photo_file in base_dir.rglob(f"{file_stem}{ext}"):
            if photo_file.is_file():
                return str(photo_file)
    
    # Handle special characters in filenames
    # Try to find files that might match
    for photo_file in base_dir.rglob("*"):
        if photo_file.is_file() and photo_file.suffix in extensions:
            if file_stem in photo_file.stem or photo_file.stem in file_stem:
                return str(photo_file)
    
    return ""


def create_basic_processed_index(successful_photos: Set[str], rate_limited_photos: Set[str]) -> None:
    """Create processed index with successful photos only."""
    index_path = Path("main_app_outputs/results/processed_index.csv")
    
    # Only mark truly successful photos as processed
    # Rate limited photos should be reprocessed
    processed_photos = successful_photos
    
    try:
        with open(index_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file_path'])
            writer.writeheader()
            for path in sorted(processed_photos):
                writer.writerow({'file_path': path})
        
        print(f"✅ Updated processed index: {len(processed_photos)} fully completed photos")
        print(f"   (Rate limited photos will be reprocessed)")
        
    except Exception as e:
        print(f"❌ Error updating processed index: {e}")


def create_basic_csv(successful_photos: Set[str]) -> None:
    """Create basic CSV files with minimal data."""
    results_dir = Path("main_app_outputs/results")
    
    if not successful_photos:
        print("❌ No successful photos to create CSV from")
        return
    
    # Basic text CSV
    text_rows = []
    full_rows = []
    
    for photo_path in successful_photos:
        file_stem = Path(photo_path).stem
        
        text_rows.append({
            'original_path': photo_path,
            'colorized_path': '',
            'full_results_path': f"main_app_outputs/results/full_{file_stem}.json",
            'llm_json_path': f"main_app_outputs/results/llm_{file_stem}_orig.json",
            'description': f"Processed photo with LLM analysis: {Path(photo_path).name}"
        })
        
        full_rows.append({
            'original_path': photo_path,
            'colorized_path': '',
            'indoor_outdoor': '',
            'background': '',
            'yolo_object_counts': '',
            'men': '0',
            'women': '0',
            'llm_caption': '',
            'full_results_path': f"main_app_outputs/results/full_{file_stem}.json",
            'llm_json_path': f"main_app_outputs/results/llm_{file_stem}_orig.json"
        })
    
    # Write CSV files
    try:
        # Text CSV
        with open(results_dir / 'data_text.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['original_path', 'colorized_path', 'full_results_path', 'llm_json_path', 'description'])
            writer.writeheader()
            writer.writerows(text_rows)
        print(f"✅ Created data_text.csv: {len(text_rows)} rows")
        
        # Full CSV
        fieldnames = ['original_path', 'colorized_path', 'indoor_outdoor', 'background', 'yolo_object_counts', 
                     'men', 'women', 'llm_caption', 'full_results_path', 'llm_json_path']
        with open(results_dir / 'data_full.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(full_rows)
        print(f"✅ Created data_full.csv: {len(full_rows)} rows")
        
    except Exception as e:
        print(f"❌ Error creating CSV files: {e}")


def main():
    print("🔧 Restoring from LLM files only...")
    print("=" * 50)
    
    # Extract successful photos from LLM files
    successful_photos, rate_limited_photos = extract_successful_llm_photos()
    
    all_photos = successful_photos | rate_limited_photos
    
    if not all_photos:
        print("❌ No photos with LLM processing found")
        return
    
    print(f"\n📝 Creating basic data structures...")
    
    # Create processed index (only successful photos)
    create_basic_processed_index(successful_photos, rate_limited_photos)
    
    # Create basic CSV files (only successful photos)
    create_basic_csv(successful_photos)
    
    # Print summary
    print(f"""
📊 Restoration Summary:
────────────────────────────────
   ✅ Photos with successful LLM: {len(successful_photos)}
   ⚠️  Photos that were rate limited: {len(rate_limited_photos)}
   📇 Processed index updated: {len(successful_photos)} marked complete
   📝 Basic CSV files created
   
💡 Status:
   - {len(successful_photos)} photos are marked as fully processed
   - {len(rate_limited_photos)} photos will be reprocessed (LLM retry)
   - Remaining ~{1068 - len(all_photos)} photos are untouched
   
🎯 Next Steps:
   Run the main pipeline to continue processing
   Use custom number option with small batches to avoid rate limits
""")


if __name__ == "__main__":
    main()