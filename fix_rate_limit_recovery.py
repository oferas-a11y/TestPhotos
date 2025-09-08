#!/usr/bin/env python3
"""
Rate Limit Recovery Script

This script handles recovery from LLM rate limits by:
1. Scanning all results for completed vs incomplete LLM processing
2. Updating processed index to only include fully completed photos
3. Cleaning up unnecessary files from incomplete processing
4. Providing clear status of what needs to be reprocessed
"""

import json
import csv
import os
from pathlib import Path
from typing import Set, Dict, List, Any


def scan_llm_results() -> Dict[str, Any]:
    """Scan all result files to determine LLM completion status."""
    results_dir = Path("main_app_outputs/results")
    
    completed_with_llm = set()
    incomplete_without_llm = set()
    failed_llm = set()
    
    stats = {
        'total_results': 0,
        'completed_with_llm': 0,
        'incomplete_without_llm': 0,
        'failed_llm': 0,
        'files_to_clean': []
    }
    
    if not results_dir.exists():
        print("❌ No results directory found")
        return {**stats, 'completed': completed_with_llm, 'incomplete': incomplete_without_llm, 'failed': failed_llm}
    
    # Scan full_*.json files for LLM completion status
    json_files = list(results_dir.glob("full_*.json"))
    print(f"🔍 Scanning {len(json_files)} result files...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            stats['total_results'] += 1
            photo_path = data.get('original_filename', '')
            
            # Check LLM section
            llm_section = data.get('llm', {})
            llm_json = llm_section.get('json', '') if isinstance(llm_section, dict) else ''
            
            if not llm_json:
                # No LLM response at all
                incomplete_without_llm.add(photo_path)
                stats['incomplete_without_llm'] += 1
                stats['files_to_clean'].append(str(json_file))
                print(f"  ❌ No LLM: {Path(photo_path).name}")
                
            else:
                # Has LLM response - check if it's an error or valid
                try:
                    llm_data = json.loads(llm_json)
                    if 'error' in llm_data:
                        # LLM returned error (rate limit, file not found, etc.)
                        failed_llm.add(photo_path)
                        stats['failed_llm'] += 1
                        stats['files_to_clean'].append(str(json_file))
                        print(f"  ⚠️  LLM Error: {Path(photo_path).name} - {llm_data.get('error', 'unknown error')}")
                    else:
                        # Valid LLM response
                        completed_with_llm.add(photo_path)
                        stats['completed_with_llm'] += 1
                        print(f"  ✅ Complete: {Path(photo_path).name}")
                        
                except json.JSONDecodeError:
                    # Invalid JSON in LLM response
                    failed_llm.add(photo_path)
                    stats['failed_llm'] += 1
                    stats['files_to_clean'].append(str(json_file))
                    print(f"  ❌ Bad JSON: {Path(photo_path).name}")
                    
        except Exception as e:
            print(f"  ❌ Error reading {json_file}: {e}")
            stats['files_to_clean'].append(str(json_file))
    
    return {
        **stats, 
        'completed': completed_with_llm, 
        'incomplete': incomplete_without_llm, 
        'failed': failed_llm
    }


def update_csv_files(completed_photos: Set[str]) -> None:
    """Update CSV files to only include photos with complete LLM processing."""
    results_dir = Path("main_app_outputs/results")
    
    for csv_name in ['data_full.csv', 'data_text.csv']:
        csv_path = results_dir / csv_name
        if not csv_path.exists():
            continue
            
        # Read existing CSV
        existing_rows = []
        fieldnames = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                for row in reader:
                    existing_rows.append(row)
        except Exception as e:
            print(f"❌ Error reading {csv_name}: {e}")
            continue
        
        # Filter to only completed photos
        completed_rows = []
        for row in existing_rows:
            original_path = row.get('original_path', '').strip()
            if original_path in completed_photos:
                completed_rows.append(row)
        
        # Write back filtered CSV
        try:
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                if fieldnames and completed_rows:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(completed_rows)
                elif fieldnames:  # Empty but valid
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
            
            print(f"✅ Updated {csv_name}: {len(existing_rows)} → {len(completed_rows)} rows")
            
        except Exception as e:
            print(f"❌ Error writing {csv_name}: {e}")


def update_processed_index(completed_photos: Set[str]) -> None:
    """Update processed index to only include fully completed photos."""
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


def clean_incomplete_files(files_to_clean: List[str]) -> None:
    """Clean up files from incomplete processing."""
    if not files_to_clean:
        print("✅ No files need cleaning")
        return
    
    print(f"🧹 Cleaning {len(files_to_clean)} incomplete result files...")
    
    cleaned_count = 0
    for file_path in files_to_clean:
        try:
            path = Path(file_path)
            if path.exists():
                # Also clean related LLM files
                llm_file = path.parent / f"llm_{path.stem}_orig.json"
                
                path.unlink()
                cleaned_count += 1
                print(f"  🗑️  Removed: {path.name}")
                
                if llm_file.exists():
                    llm_file.unlink()
                    print(f"  🗑️  Removed: {llm_file.name}")
                    
        except Exception as e:
            print(f"  ❌ Failed to remove {file_path}: {e}")
    
    print(f"✅ Cleaned {cleaned_count} incomplete files")


def print_summary(stats: Dict[str, Any]) -> None:
    """Print recovery summary."""
    print(f"""
📊 Rate Limit Recovery Summary:
────────────────────────────────
   Total result files found: {stats['total_results']}
   ✅ Completed with LLM: {stats['completed_with_llm']}
   ❌ Missing LLM: {stats['incomplete_without_llm']}  
   ⚠️  Failed LLM: {stats['failed_llm']}
   🧹 Files cleaned: {len(stats['files_to_clean'])}

📈 System Status:
   Photos ready for search: {stats['completed_with_llm']}
   Photos need reprocessing: {stats['incomplete_without_llm'] + stats['failed_llm']}
   
💡 Next Steps:
   Run the main pipeline to process remaining {stats['incomplete_without_llm'] + stats['failed_llm']} photos
   Use smaller batch sizes to avoid rate limits
""")


def main():
    print("🔧 Starting Rate Limit Recovery...")
    print("=" * 50)
    
    # Scan all results for LLM completion
    scan_results = scan_llm_results()
    
    completed_photos = scan_results['completed']
    incomplete_photos = scan_results['incomplete'] | scan_results['failed']
    
    print(f"\n📋 Processing Results:")
    print(f"   Completed photos: {len(completed_photos)}")
    print(f"   Incomplete photos: {len(incomplete_photos)}")
    
    if not completed_photos and not incomplete_photos:
        print("❌ No results found to process")
        return
    
    # Update CSV files to only include completed
    if completed_photos:
        print(f"\n📝 Updating CSV files...")
        update_csv_files(completed_photos)
    
    # Update processed index
    print(f"\n📇 Updating processed index...")
    update_processed_index(completed_photos)
    
    # Clean incomplete files
    if scan_results['files_to_clean']:
        print(f"\n🧹 Cleaning incomplete files...")
        clean_incomplete_files(scan_results['files_to_clean'])
    
    # Print summary
    print_summary(scan_results)
    
    print("✅ Recovery complete! System is ready for continued processing.")


if __name__ == "__main__":
    main()