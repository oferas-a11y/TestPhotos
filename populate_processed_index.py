#!/usr/bin/env python3
"""
Utility script to populate the processed index with existing results.
Run this if you have previously processed photos that aren't in the index.
"""

import csv
import json
from pathlib import Path
from typing import Set


def populate_from_existing_csvs() -> Set[str]:
    """Populate from existing CSV files."""
    processed = set()
    
    # Check data_full.csv and data_text.csv for existing entries
    for csv_file in ['data_full.csv', 'data_text.csv']:
        csv_path = Path(f"main_app_outputs/results/{csv_file}")
        if csv_path.exists():
            try:
                with open(csv_path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        path = row.get('original_path', '').strip()
                        if path:
                            processed.add(path)
                            print(f"Found in {csv_file}: {Path(path).name}")
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
    
    return processed


def populate_from_json_files() -> Set[str]:
    """Populate from existing full_*.json result files."""
    processed = set()
    
    results_dir = Path("main_app_outputs/results")
    if not results_dir.exists():
        return processed
    
    # Find all full_*.json files
    json_files = list(results_dir.glob("full_*.json"))
    print(f"Found {len(json_files)} result JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                path = data.get('original_filename', '').strip()
                if path:
                    processed.add(path)
                    print(f"Found in {json_file.name}: {Path(path).name}")
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    return processed


def write_processed_index(processed_files: Set[str]) -> None:
    """Write the processed index CSV."""
    if not processed_files:
        print("No processed files found to add to index.")
        return
    
    index_path = Path("main_app_outputs/results/processed_index.csv")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing entries
    existing = set()
    if index_path.exists():
        try:
            with open(index_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    path = row.get('file_path', '').strip()
                    if path:
                        existing.add(path)
        except Exception:
            pass
    
    # Merge with new entries
    all_processed = existing | processed_files
    
    # Write updated index
    with open(index_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file_path'])
        writer.writeheader()
        for path in sorted(all_processed):
            writer.writerow({'file_path': path})
    
    print(f"✅ Updated processed index with {len(all_processed)} files")
    print(f"   - Previously in index: {len(existing)}")
    print(f"   - Added from results: {len(processed_files - existing)}")


def main():
    print("🔍 Populating processed index from existing results...")
    
    # Gather processed files from multiple sources
    all_processed = set()
    
    # From CSV files
    csv_processed = populate_from_existing_csvs()
    all_processed.update(csv_processed)
    
    # From JSON result files
    json_processed = populate_from_json_files()
    all_processed.update(json_processed)
    
    # Write the index
    write_processed_index(all_processed)
    
    print(f"\n📊 Summary:")
    print(f"   Total processed files found: {len(all_processed)}")
    if all_processed:
        print(f"   Index updated: main_app_outputs/results/processed_index.csv")
    else:
        print("   No previously processed files found - starting fresh!")


if __name__ == "__main__":
    main()