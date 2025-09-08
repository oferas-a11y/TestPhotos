#!/usr/bin/env python3
"""
Migrate existing CSV data to ChromaDB

This script migrates all existing photo analysis data from CSV files to ChromaDB
for improved search performance and vector similarity capabilities.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def load_existing_csv_data() -> List[Dict[str, Any]]:
    """Load and convert existing CSV data to record format."""
    results_dir = Path("main_app_outputs/results")
    data_full_path = results_dir / "data_full.csv"
    
    if not data_full_path.exists():
        print(f"❌ No data_full.csv found at {data_full_path}")
        print("   Run photo processing first to create data")
        return []
    
    records = []
    
    print(f"📖 Loading data from {data_full_path}")
    
    with open(data_full_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader, 1):
            try:
                # Convert CSV row back to record format
                record = convert_csv_row_to_record(row)
                records.append(record)
                
                if i % 10 == 0:
                    print(f"   Loaded {i} records...")
            
            except Exception as e:
                print(f"⚠️  Error processing row {i}: {e}")
                continue
    
    print(f"✅ Loaded {len(records)} records from CSV")
    return records


def convert_csv_row_to_record(row: Dict[str, str]) -> Dict[str, Any]:
    """
    Convert CSV row back to the original record format.
    
    This reconstructs the hierarchical record structure from the flattened CSV data.
    """
    original_path = row.get('original_path', '')
    
    # Base record structure
    record = {
        'original_filename': original_path,
        'colorized_path': row.get('colorized_path', ''),
        'processed_path': row.get('original_path', ''),  # Use original_path as processed_path
        'source_path': row.get('source_path', '') or row.get('original_path', '')
    }
    
    # Reconstruct CLIP section
    indoor_outdoor = row.get('indoor_outdoor', '')
    men_count = int(row.get('men', '0') or '0')
    women_count = int(row.get('women', '0') or '0')
    
    # Reconstruct gender data (simplified)
    people_gender = []
    for _ in range(men_count):
        people_gender.append({'man': 1.0, 'woman': 0.0})
    for _ in range(women_count):
        people_gender.append({'man': 0.0, 'woman': 1.0})
    
    record['clip'] = {
        'indoor_outdoor': indoor_outdoor,
        'background_top': [row.get('background', '')] if row.get('background') else [],
        'background_detections': [],
        'people_gender': people_gender,
        'notes': 'Reconstructed from CSV data'
    }
    
    # Reconstruct YOLO section
    yolo_counts_str = row.get('yolo_object_counts', '')
    object_counts = {}
    detections = []
    
    if yolo_counts_str:
        try:
            # Parse "person:2, chair:1" format
            for item in yolo_counts_str.split(','):
                if ':' in item:
                    obj_type, count_str = item.split(':', 1)
                    obj_type = obj_type.strip()
                    count = int(count_str.strip())
                    object_counts[obj_type] = count
                    
                    # Create simplified detections
                    for i in range(count):
                        detection = {
                            'class_name': obj_type,
                            'confidence': 0.5,  # Default confidence
                            'bbox': {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100, 'center_x': 50, 'center_y': 50}
                        }
                        detections.append(detection)
        except Exception:
            pass
    
    record['yolo'] = {
        'object_counts': object_counts,
        'top_objects': list(object_counts.keys())[:5],
        'detections': detections,
        'notes': 'Reconstructed from CSV data'
    }
    
    # Reconstruct OCR section
    has_ocr = row.get('has_ocr_text', '').lower() == 'true'
    ocr_sample = row.get('ocr_text_sample', '')
    
    record['ocr'] = {
        'has_text': has_ocr,
        'lines': [ocr_sample] if ocr_sample else [],
        'items': []
    }
    
    # Reconstruct LLM section from JSON file if available
    llm_json_path = row.get('llm_json_path', '')
    if llm_json_path and Path(llm_json_path).exists():
        try:
            with open(llm_json_path, 'r', encoding='utf-8') as f:
                llm_json_data = f.read()
                record['llm'] = {
                    'json': llm_json_data,
                    'output_path': llm_json_path
                }
        except Exception as e:
            print(f"⚠️  Could not load LLM data from {llm_json_path}: {e}")
            record['llm'] = create_llm_from_csv(row)
    else:
        # Reconstruct LLM data from CSV fields
        record['llm'] = create_llm_from_csv(row)
    
    return record


def create_llm_from_csv(row: Dict[str, str]) -> Dict[str, Any]:
    """Create LLM section from CSV fields."""
    # Reconstruct LLM JSON from CSV fields
    llm_data = {
        'caption': row.get('llm_caption', ''),
        'people_under_18': int(row.get('llm_people_under_18', '0') or '0'),
        'has_jewish_symbols': row.get('has_jewish_symbols', '').lower() == 'yes',
        'jewish_symbols_details': [],
        'has_nazi_symbols': row.get('has_nazi_symbols', '').lower() == 'yes',
        'nazi_symbols_details': [],
        'text_analysis': {
            'hebrew_text': {
                'present': row.get('has_hebrew_text', '').lower() == 'yes',
                'text_found': row.get('hebrew_text', ''),
                'translation': row.get('hebrew_translation', ''),
                'context': ''
            },
            'german_text': {
                'present': row.get('has_german_text', '').lower() == 'yes', 
                'text_found': row.get('german_text', ''),
                'translation': row.get('german_translation', ''),
                'context': ''
            }
        },
        'main_objects_artifacts_animals': [],
        'violence_assessment': {
            'signs_of_violence': row.get('has_violence', '').lower() == 'yes',
            'explanation': row.get('violence_explanation', '')
        }
    }
    
    # Parse objects from CSV
    objects_str = row.get('llm_objects', '')
    if objects_str:
        try:
            # Parse "item: description; item2: description2" format
            for item in objects_str.split(';'):
                if ':' in item:
                    parts = item.split(':', 1)
                    if len(parts) == 2:
                        llm_data['main_objects_artifacts_animals'].append({
                            'item': parts[0].strip(),
                            'category': 'object',
                            'description': parts[1].strip(),
                            'significance': ''
                        })
        except Exception:
            pass
    
    # Parse symbols
    for symbol_type, csv_field in [('jewish', 'llm_jewish_symbols'), ('nazi', 'llm_nazi_symbols')]:
        symbols_str = row.get(csv_field, '')
        if symbols_str:
            try:
                symbol_details = []
                for item in symbols_str.split(';'):
                    if ' - ' in item:
                        parts = item.split(' - ', 1)
                        symbol_details.append({
                            'symbol_type': parts[0].strip(),
                            'description': parts[1].strip(),
                            'location_in_image': ''
                        })
                llm_data[f'{symbol_type}_symbols_details'] = symbol_details
            except Exception:
                pass
    
    return {
        'json': json.dumps(llm_data, ensure_ascii=False),
        'output_path': ''
    }


def migrate_to_chromadb(records: List[Dict[str, Any]]) -> bool:
    """Migrate records to ChromaDB."""
    try:
        from main_app.modules.chroma_handler import create_chroma_handler
    except ImportError as e:
        print(f"❌ ChromaDB handler not available: {e}")
        print("   Install ChromaDB with: pip install chromadb")
        return False
    
    print("\n💾 Initializing ChromaDB...")
    
    # Create ChromaDB handler
    chroma_handler = create_chroma_handler()
    if not chroma_handler:
        print("❌ Failed to create ChromaDB handler")
        return False
    
    # Get current stats
    initial_stats = chroma_handler.get_collection_stats()
    initial_count = initial_stats.get('total_photos', 0)
    
    print(f"📊 ChromaDB currently contains {initial_count} photos")
    print(f"🔄 Migrating {len(records)} records...")
    
    # Batch store records
    try:
        photo_ids = chroma_handler.batch_store_photos(records)
        
        # Get final stats
        final_stats = chroma_handler.get_collection_stats()
        final_count = final_stats.get('total_photos', 0)
        new_photos = len(photo_ids)
        
        print(f"✅ Migration completed successfully!")
        print(f"📊 Final ChromaDB stats:")
        print(f"   Total photos in database: {final_count}")
        print(f"   New photos added: {new_photos}")
        print(f"   Photos already existed: {len(records) - new_photos}")
        print(f"   Collection: {final_stats.get('collection_name', 'unknown')}")
        print(f"   Storage: {final_stats.get('persist_directory', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main migration function."""
    print("🔄 TestPhotos - CSV to ChromaDB Migration")
    print("=" * 60)
    
    # Check ChromaDB availability
    try:
        from main_app.modules.chroma_handler import is_chromadb_available
        if not is_chromadb_available():
            print("❌ ChromaDB is not available")
            print("   Install with: pip install chromadb")
            return False
    except ImportError:
        print("❌ ChromaDB handler not found")
        print("   Make sure the ChromaDB integration is properly set up")
        return False
    
    # Step 1: Load existing CSV data
    print("\n📖 Step 1: Loading existing CSV data...")
    records = load_existing_csv_data()
    
    if not records:
        print("❌ No data to migrate")
        return False
    
    # Step 2: Migrate to ChromaDB
    print(f"\n💾 Step 2: Migrating {len(records)} records to ChromaDB...")
    success = migrate_to_chromadb(records)
    
    if success:
        print("\n🎉 Migration completed successfully!")
        print("\nNext steps:")
        print("   • Run 'python run.py chroma_search' to test ChromaDB search")
        print("   • Use 'python run.py dashboard' and try ChromaDB search options")
        print("   • Future photo processing will automatically store to ChromaDB")
        return True
    else:
        print("\n❌ Migration failed")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)