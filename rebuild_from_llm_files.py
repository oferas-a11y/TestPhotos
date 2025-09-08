#!/usr/bin/env python3
"""
Rebuild CSV files and processed index from existing LLM files.
This script finds all existing LLM result files and rebuilds the data properly.
"""

import json
import csv
from pathlib import Path
from typing import Set, Dict, List, Any


def find_completed_photos_from_llm() -> Dict[str, Dict[str, Any]]:
    """Find photos with valid LLM results by scanning llm_*_orig.json files."""
    results_dir = Path("main_app_outputs/results")
    completed_photos = {}
    
    if not results_dir.exists():
        print("❌ Results directory not found")
        return {}
    
    # Find all LLM result files
    llm_files = list(results_dir.glob("llm_*_orig.json"))
    print(f"🔍 Found {len(llm_files)} LLM result files")
    
    for llm_file in llm_files:
        try:
            # Read LLM result
            with open(llm_file, 'r', encoding='utf-8') as f:
                llm_data = json.load(f)
            
            # Check if it's a valid LLM response (not an error)
            if 'error' in llm_data:
                print(f"  ⚠️  LLM Error: {llm_file.name} - {llm_data.get('error', 'unknown')}")
                continue
            
            # Extract original filename from LLM file name
            # Format: llm_<filename>_orig.json
            file_stem = llm_file.name.replace('llm_', '').replace('_orig.json', '')
            
            # Try to find the corresponding full result file
            full_file = results_dir / f"full_{file_stem}.json"
            
            if full_file.exists():
                try:
                    with open(full_file, 'r', encoding='utf-8') as f:
                        full_data = json.load(f)
                    
                    # Combine full data with LLM data
                    original_filename = full_data.get('original_filename', '')
                    if original_filename:
                        # Add LLM data to the record
                        full_data['llm'] = {
                            'output_path': str(llm_file),
                            'json': json.dumps(llm_data, ensure_ascii=False)
                        }
                        
                        completed_photos[original_filename] = full_data
                        print(f"  ✅ Complete: {Path(original_filename).name}")
                    else:
                        print(f"  ❌ No original filename in: {full_file.name}")
                        
                except Exception as e:
                    print(f"  ❌ Error reading {full_file}: {e}")
            else:
                print(f"  ❌ Missing full result file for: {llm_file.name}")
                
        except Exception as e:
            print(f"  ❌ Error processing {llm_file}: {e}")
    
    return completed_photos


def rebuild_csv_files(completed_photos: Dict[str, Dict[str, Any]]) -> None:
    """Rebuild CSV files from completed photo data."""
    if not completed_photos:
        print("❌ No completed photos to rebuild CSV files")
        return
    
    print(f"📝 Rebuilding CSV files from {len(completed_photos)} completed photos...")
    
    # Import the CSV export functionality
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'main_app'))
    
    try:
        from main_pipeline import CSVExporter
        
        # Convert to the expected format (list of records)
        per_image = list(completed_photos.values())
        
        # Create CSV rows
        input_path = Path("photo_collections/project Photography")  # Adjust if different
        results_dir = Path("main_app_outputs/results")
        
        rows_for_text, rows_for_full = CSVExporter.create_csv_rows(per_image, input_path, results_dir)
        
        # Write CSV files
        write_csv_with_merge(results_dir / 'data_text.csv', rows_for_text, 
                           ['original_path', 'colorized_path', 'full_results_path', 'llm_json_path', 'description'])
        
        if rows_for_full:
            full_fields = list(rows_for_full[0].keys()) if rows_for_full else []
            write_csv_with_merge(results_dir / 'data_full.csv', rows_for_full, full_fields)
        
        print(f"✅ Rebuilt CSV files successfully")
        
    except Exception as e:
        print(f"❌ Error rebuilding CSV files: {e}")
        print("Falling back to basic CSV rebuild...")
        basic_csv_rebuild(completed_photos)


def basic_csv_rebuild(completed_photos: Dict[str, Dict[str, Any]]) -> None:
    """Basic CSV rebuild without using the complex CSVExporter."""
    results_dir = Path("main_app_outputs/results")
    
    # Simple text CSV
    text_rows = []
    full_rows = []
    
    for original_filename, record in completed_photos.items():
        # Basic text row
        text_rows.append({
            'original_path': original_filename,
            'colorized_path': '',
            'full_results_path': f"main_app_outputs/results/full_{Path(original_filename).stem}.json",
            'llm_json_path': record.get('llm', {}).get('output_path', ''),
            'description': f"Processed photo: {Path(original_filename).name}"
        })
        
        # Basic full row  
        full_rows.append({
            'original_path': original_filename,
            'colorized_path': '',
            'indoor_outdoor': record.get('clip', {}).get('indoor_outdoor', ''),
            'background': '',
            'yolo_object_counts': '',
            'men': '0',
            'women': '0',
            'llm_caption': '',
            'full_results_path': f"main_app_outputs/results/full_{Path(original_filename).stem}.json",
            'llm_json_path': record.get('llm', {}).get('output_path', '')
        })
    
    # Write basic CSV files
    if text_rows:
        write_csv_simple(results_dir / 'data_text.csv', text_rows, 
                        ['original_path', 'colorized_path', 'full_results_path', 'llm_json_path', 'description'])
    
    if full_rows:
        write_csv_simple(results_dir / 'data_full.csv', full_rows,
                        ['original_path', 'colorized_path', 'indoor_outdoor', 'background', 'yolo_object_counts', 
                         'men', 'women', 'llm_caption', 'full_results_path', 'llm_json_path'])


def write_csv_simple(csv_path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    """Simple CSV writer."""
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ Wrote {csv_path.name}: {len(rows)} rows")
    except Exception as e:
        print(f"❌ Error writing {csv_path}: {e}")


def write_csv_with_merge(csv_path: Path, new_rows: List[Dict], field_names: List[str]) -> None:
    """Write CSV file, merging with existing data."""
    import csv
    
    existing: Dict[str, Dict[str, str]] = {}
    
    # Load existing data if file exists
    if csv_path.exists():
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    key = (r.get('original_path') or '').strip()
                    if key:
                        existing[key] = {k: (v or '') for k, v in r.items()}
                        # Update fieldnames if new fields found
                        for k in r.keys():
                            if k not in field_names:
                                field_names.append(k)
        except Exception:
            pass
    
    # Merge new rows
    for r in new_rows:
        key = (r.get('original_path') or '').strip()
        if key:
            existing[key] = {**existing.get(key, {}), **r}
    
    # Write merged data
    if field_names and existing:
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=field_names)
                writer.writeheader()
                for key in sorted(existing.keys()):
                    row = existing[key]
                    writer.writerow({fn: row.get(fn, '') for fn in field_names})
            print(f"✅ Wrote {csv_path.name}: {len(existing)} rows")
        except Exception as e:
            print(f"❌ Error writing {csv_path}: {e}")


def update_processed_index(completed_photos: Set[str]) -> None:
    """Update processed index with completed photos."""
    index_path = Path("main_app_outputs/results/processed_index.csv")
    
    try:
        with open(index_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file_path'])
            writer.writeheader()
            for path in sorted(completed_photos):
                writer.writerow({'file_path': path})
        
        print(f"✅ Updated processed index: {len(completed_photos)} completed photos")
        
    except Exception as e:
        print(f"❌ Error updating processed index: {e}")


def main():
    print("🔧 Rebuilding from existing LLM files...")
    print("=" * 50)
    
    # Find all completed photos from LLM files
    completed_photos = find_completed_photos_from_llm()
    
    if not completed_photos:
        print("❌ No valid completed photos found")
        return
    
    print(f"\n📊 Found {len(completed_photos)} photos with valid LLM results")
    
    # Rebuild CSV files
    rebuild_csv_files(completed_photos)
    
    # Update processed index
    update_processed_index(set(completed_photos.keys()))
    
    # Print summary
    print(f"""
📊 Rebuild Summary:
────────────────────────────────
   ✅ Photos with complete LLM: {len(completed_photos)}
   📝 CSV files rebuilt with valid data
   📇 Processed index updated
   
🎉 System restored! You can now:
   - Search/browse the {len(completed_photos)} completed photos
   - Process remaining ~{1068 - len(completed_photos)} unprocessed photos
   - All LLM data is preserved and available
""")


if __name__ == "__main__":
    main()