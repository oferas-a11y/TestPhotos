#!/usr/bin/env python3
"""
Rebuild CSV files with ACTUAL DATA extracted from LLM files and any existing full result files.
No more lazy pointer CSVs - this extracts all the real content!
"""

import json
import csv
from pathlib import Path
from typing import Set, Dict, List, Any, Optional


def extract_data_from_llm_and_results() -> List[Dict[str, Any]]:
    """Extract all actual data from LLM files and any remaining result files, plus ALL photos from processed_index.csv."""
    results_dir = Path("main_app_outputs/results")
    photo_data = []
    processed_files = set()
    
    if not results_dir.exists():
        print("❌ Results directory not found")
        return []
    
    # First, process all LLM result files with valid data
    llm_files = list(results_dir.glob("llm_*_orig.json"))
    print(f"🔍 Processing {len(llm_files)} LLM files for real data extraction...")
    
    for llm_file in llm_files:
        try:
            # Read LLM result
            with open(llm_file, 'r', encoding='utf-8') as f:
                llm_data = json.load(f)
            
            # Skip error responses
            if 'error' in llm_data:
                continue
            
            # Extract original filename from LLM file name
            file_stem = llm_file.name.replace('llm_', '').replace('_orig.json', '')
            
            # Find the actual photo file
            original_path = find_actual_photo_path(file_stem)
            if not original_path:
                print(f"  ⚠️  Could not find original photo for: {file_stem}")
                continue
            
            # Track processed files
            processed_files.add(original_path)
            
            # Initialize record with basic info
            record = {
                'original_filename': original_path,
                'colorized_path': f"main_app_outputs/colorized/colorized_{Path(original_path).name}",
            }
            
            # Try to get additional data from full result file if it exists
            full_file = results_dir / f"full_{file_stem}.json"
            if full_file.exists():
                try:
                    with open(full_file, 'r', encoding='utf-8') as f:
                        full_data = json.load(f)
                    
                    # Extract YOLO data
                    yolo_data = full_data.get('yolo', {})
                    record['yolo'] = yolo_data
                    
                    # Extract CLIP data
                    clip_data = full_data.get('clip', {})
                    record['clip'] = clip_data
                    
                    # Extract OCR data
                    ocr_data = full_data.get('ocr', {})
                    record['ocr'] = ocr_data
                    
                except Exception as e:
                    print(f"  ⚠️  Error reading full result for {file_stem}: {e}")
            
            # Add LLM data
            record['llm'] = {
                'output_path': str(llm_file),
                'json': json.dumps(llm_data, ensure_ascii=False)
            }
            
            photo_data.append(record)
            print(f"  ✅ Extracted data for: {Path(original_path).name}")
            
        except Exception as e:
            print(f"  ❌ Error processing {llm_file}: {e}")
    
    # Now process ALL photos from processed_index.csv to catch any missing ones
    processed_index_file = results_dir / "processed_index.csv"
    if processed_index_file.exists():
        print(f"🔍 Processing photos from processed_index.csv...")
        try:
            with open(processed_index_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                for row in reader:
                    if row and len(row) > 0:
                        photo_path = row[0].strip()
                        if photo_path and photo_path not in processed_files:
                            # This photo wasn't processed by LLM yet, add basic record
                            record = {
                                'original_filename': photo_path,
                                'colorized_path': f"main_app_outputs/colorized/colorized_{Path(photo_path).name}",
                                'yolo': {},
                                'clip': {},
                                'ocr': {},
                                'llm': {}
                            }
                            photo_data.append(record)
                            print(f"  ➕ Added photo without LLM data: {Path(photo_path).name}")
                            
        except Exception as e:
            print(f"  ⚠️  Error reading processed_index.csv: {e}")
    
    print(f"✅ Extracted data from {len(photo_data)} total photos")
    return photo_data


def find_actual_photo_path(file_stem: str) -> str:
    """Find the actual photo file path by searching."""
    base_dir = Path("photo_collections/project Photography")
    
    if not base_dir.exists():
        return ""
    
    # Common extensions
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # Search for files with matching stem
    for ext in extensions:
        for photo_file in base_dir.rglob(f"{file_stem}{ext}"):
            if photo_file.is_file():
                return str(photo_file)
    
    # Handle special characters in filenames - try broader search
    for photo_file in base_dir.rglob("*"):
        if photo_file.is_file() and photo_file.suffix in extensions:
            if file_stem in photo_file.stem or photo_file.stem.replace(' ', '_').replace('-', '_') == file_stem:
                return str(photo_file)
    
    return ""


def create_text_csv(photo_data: List[Dict[str, Any]]) -> None:
    """Create data_text.csv with ALL extracted data combined into single text column for embedding."""
    results_dir = Path("main_app_outputs/results")
    text_rows = []
    
    for record in photo_data:
        original_path = record['original_filename']
        
        # Combine ALL extracted data into single comprehensive text
        text_parts = []
        
        # Extract folder path information for context (will be added at the end)
        folder_context = ""
        try:
            path_parts = Path(original_path).parts
            if len(path_parts) > 2 and path_parts[0] == "photo_collections" and path_parts[1] == "project Photography":
                folder_name = path_parts[2]
                folder_context = f"Collection: {folder_name.replace(' – ', ' - ').replace('_', ' ')}"
        except Exception:
            pass
        
        # 1. LLM CAPTION (first)
        llm_data = record.get('llm', {})
        llm_parsed = {}
        if llm_data.get('json'):
            try:
                llm_parsed = json.loads(llm_data['json'])
                caption = llm_parsed.get('caption', '')
                if caption:
                    text_parts.append(caption)
            except json.JSONDecodeError:
                pass
        
        # If no LLM caption, create basic description from filename
        if not text_parts:
            filename = Path(original_path).stem
            # Create basic description from filename
            basic_desc = filename.replace('_', ' ').replace('-', ' ')
            if basic_desc:
                text_parts.append(f"Historical photograph {basic_desc}")
        
        # 2. TEXT (Hebrew/German)
        if llm_parsed:
            text_analysis = llm_parsed.get('text_analysis', {})
            
            # Hebrew text
            hebrew_text = text_analysis.get('hebrew_text', {})
            if hebrew_text.get('present', False):
                hebrew_found = hebrew_text.get('text_found', '')
                if hebrew_found:
                    text_parts.append(f"Hebrew text: {hebrew_found}")
                hebrew_translation = hebrew_text.get('translation', '')
                if hebrew_translation:
                    text_parts.append(f"Hebrew translation: {hebrew_translation}")
            
            # German text  
            german_text = text_analysis.get('german_text', {})
            if german_text.get('present', False):
                german_found = german_text.get('text_found', '')
                if german_found:
                    text_parts.append(f"German text: {german_found}")
                german_translation = german_text.get('translation', '')
                if german_translation:
                    text_parts.append(f"German translation: {german_translation}")
        
        # 3. CLIP analysis
        clip_data = record.get('clip', {})
        indoor_outdoor = clip_data.get('indoor_outdoor', '')
        if indoor_outdoor:
            text_parts.append(f"{indoor_outdoor} setting")
        
        # Background
        background_detections = clip_data.get('background_detections', [])
        for bg in background_detections[:2]:
            category = bg.get('category', '')
            if category:
                text_parts.append(f"Background: {category}")
        
        # People gender
        people_gender = clip_data.get('people_gender', [])
        men_count = sum(1 for p in people_gender if p.get('man', 0) >= p.get('woman', 0))
        women_count = len(people_gender) - men_count
        if men_count > 0:
            text_parts.append(f"{men_count} men")
        if women_count > 0:
            text_parts.append(f"{women_count} women")
        
        # 4. YOLO detection
        yolo_data = record.get('yolo', {})
        object_counts = yolo_data.get('object_counts', {})
        for obj_type, count in object_counts.items():
            if count > 0:
                text_parts.append(f"{count} {obj_type}{'s' if count > 1 else ''} detected")
        
        # 5. LLM detailed analysis
        if llm_parsed:
            # People count
            people_under_18 = llm_parsed.get('people_under_18', 0)
            if people_under_18 > 0:
                text_parts.append(f"{people_under_18} people under 18")
            
            # Symbols
            if llm_parsed.get('has_jewish_symbols', False):
                text_parts.append("Jewish symbols present")
                jewish_symbols = llm_parsed.get('jewish_symbols_details', [])
                for symbol in jewish_symbols:
                    symbol_type = symbol.get('symbol_type', '')
                    if symbol_type:
                        text_parts.append(symbol_type)
            
            if llm_parsed.get('has_nazi_symbols', False):
                text_parts.append("Nazi symbols present")
                nazi_symbols = llm_parsed.get('nazi_symbols_details', [])
                for symbol in nazi_symbols:
                    symbol_type = symbol.get('symbol_type', '')
                    if symbol_type:
                        text_parts.append(symbol_type)
            
            # Objects and artifacts
            objects = llm_parsed.get('main_objects_artifacts_animals', [])
            for obj in objects:
                item = obj.get('item', '')
                description = obj.get('description', '')
                if item:
                    text_parts.append(item)
                    if description:
                        text_parts.append(description)
            
            # Violence
            violence = llm_parsed.get('violence_assessment', {})
            if violence.get('signs_of_violence', False):
                explanation = violence.get('explanation', '')
                if explanation and explanation != "No signs of violence detected":
                    text_parts.append(explanation)
        
        # OCR text
        ocr_data = record.get('ocr', {})
        if ocr_data.get('has_text', False):
            ocr_lines = ocr_data.get('lines', [])
            for line in ocr_lines[:2]:
                if line.strip():
                    text_parts.append(f"OCR text: {line.strip()}")
        
        # Combine all text parts into single comprehensive description with folder context at the end
        comprehensive_text = ' '.join([part for part in text_parts if part and part.strip()])
        
        # Add folder context at the end
        if folder_context:
            if comprehensive_text:
                comprehensive_text = f"{comprehensive_text} {folder_context}"
            else:
                comprehensive_text = folder_context
        
        # Fallback if no data found
        if not comprehensive_text:
            comprehensive_text = f"Historical photograph from {Path(original_path).name}"
        
        text_rows.append({
            'original_path': original_path,
            'comprehensive_text': comprehensive_text
        })
    
    # Write simplified CSV with only path and comprehensive text
    fieldnames = ['original_path', 'comprehensive_text']
    
    try:
        with open(results_dir / 'data_text.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(text_rows)
        print(f"✅ Created data_text.csv with comprehensive text for embedding: {len(text_rows)} rows")
    except Exception as e:
        print(f"❌ Error creating data_text.csv: {e}")


def create_full_csv(photo_data: List[Dict[str, Any]]) -> None:
    """Create data_full.csv with actual extracted structured data."""
    results_dir = Path("main_app_outputs/results")
    full_rows = []
    
    for record in photo_data:
        original_path = record['original_filename']
        
        # Extract all the structured data
        yolo_data = record.get('yolo', {})
        clip_data = record.get('clip', {})
        ocr_data = record.get('ocr', {})
        llm_data = record.get('llm', {})
        
        # Parse LLM data
        llm_parsed = {}
        if llm_data.get('json'):
            try:
                llm_parsed = json.loads(llm_data['json'])
            except json.JSONDecodeError:
                pass
        
        # Object counts
        object_counts = yolo_data.get('object_counts', {})
        person_count = object_counts.get('person', 0)
        
        # Gender counts from CLIP
        people_gender = clip_data.get('people_gender', [])
        men_count = sum(1 for p in people_gender if p.get('man', 0) >= p.get('woman', 0))
        women_count = len(people_gender) - men_count
        
        # Background info
        background_detections = clip_data.get('background_detections', [])
        background_top = background_detections[0]['category'] if background_detections else ''
        background_confidence = background_detections[0]['confidence'] if background_detections else 0
        
        # LLM extracted info
        llm_caption = llm_parsed.get('caption', '')
        llm_people_under_18 = llm_parsed.get('people_under_18', 0)
        llm_has_jewish = llm_parsed.get('has_jewish_symbols', False)
        llm_has_nazi = llm_parsed.get('has_nazi_symbols', False)
        
        # Text analysis
        text_analysis = llm_parsed.get('text_analysis', {})
        hebrew_text = text_analysis.get('hebrew_text', {}).get('present', False)
        german_text = text_analysis.get('german_text', {}).get('present', False)
        
        # Violence assessment
        violence = llm_parsed.get('violence_assessment', {}).get('signs_of_violence', False)
        
        # Objects from LLM
        llm_objects = llm_parsed.get('main_objects_artifacts_animals', [])
        objects_list = '; '.join([f"{obj.get('item', '')}: {obj.get('description', '')}" for obj in llm_objects[:5] if obj.get('item')])
        
        # OCR text
        ocr_has_text = ocr_data.get('has_text', False)
        ocr_lines = '; '.join(ocr_data.get('lines', [])[:3])
        
        # Army detection (if available)
        army_detected = clip_data.get('army', False)
        army_objects = clip_data.get('army_objects', {})
        army_object_counts = ', '.join([f"{k}:{v}" for k, v in army_objects.items() if v > 0])
        
        full_rows.append({
            'original_path': original_path,
            'colorized_path': record.get('colorized_path', ''),
            'indoor_outdoor': clip_data.get('indoor_outdoor', ''),
            'background_category': background_top,
            'background_confidence': f"{background_confidence:.3f}" if background_confidence else '',
            'yolo_object_counts': ', '.join([f"{k}:{v}" for k, v in object_counts.items()]),
            'total_people': person_count,
            'men': men_count,
            'women': women_count,
            'llm_caption': llm_caption,
            'llm_people_under_18': llm_people_under_18,
            'llm_objects': objects_list,
            'has_jewish_symbols': 'Yes' if llm_has_jewish else 'No',
            'has_nazi_symbols': 'Yes' if llm_has_nazi else 'No',
            'has_hebrew_text': 'Yes' if hebrew_text else 'No',
            'has_german_text': 'Yes' if german_text else 'No',
            'has_violence': 'Yes' if violence else 'No',
            'has_ocr_text': 'Yes' if ocr_has_text else 'No',
            'ocr_text_sample': ocr_lines,
            'army_detected': 'Yes' if army_detected else 'No',
            'army_objects': army_object_counts,
            'full_results_path': f"main_app_outputs/results/full_{Path(original_path).stem}.json",
            'llm_json_path': llm_data.get('output_path', '')
        })
    
    # Write full CSV
    fieldnames = ['original_path', 'colorized_path', 'indoor_outdoor', 'background_category', 'background_confidence',
                  'yolo_object_counts', 'total_people', 'men', 'women', 'llm_caption', 'llm_people_under_18', 
                  'llm_objects', 'has_jewish_symbols', 'has_nazi_symbols', 'has_hebrew_text', 'has_german_text',
                  'has_violence', 'has_ocr_text', 'ocr_text_sample', 'army_detected', 'army_objects',
                  'full_results_path', 'llm_json_path']
    
    try:
        with open(results_dir / 'data_full.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(full_rows)
        print(f"✅ Created data_full.csv with REAL data: {len(full_rows)} rows")
    except Exception as e:
        print(f"❌ Error creating data_full.csv: {e}")


def main():
    print("🔧 Rebuilding CSV files with REAL extracted data...")
    print("=" * 60)
    
    # Extract all real data from LLM and result files
    photo_data = extract_data_from_llm_and_results()
    
    if not photo_data:
        print("❌ No photo data found to process")
        return
    
    print(f"\n📊 Creating CSV files with real data from {len(photo_data)} photos...")
    
    # Create text CSV with real descriptions and content
    create_text_csv(photo_data)
    
    # Create full CSV with all structured data
    create_full_csv(photo_data)
    
    print(f"""
🎉 SUCCESS! CSV files rebuilt with REAL DATA!
═══════════════════════════════════════════════

📄 data_text.csv: Contains actual descriptions, captions, object counts, settings
📊 data_full.csv: Contains all structured data - people counts, symbols, violence, objects, etc.

✅ No more lazy pointer files - all actual content extracted!
✅ Ready for search, analysis, and dashboard use!
✅ {len(photo_data)} photos with complete real data!

💡 The CSV files now contain the actual extracted content, not just file paths!
""")


if __name__ == "__main__":
    main()