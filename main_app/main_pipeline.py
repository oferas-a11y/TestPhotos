"""Main image processing pipeline (colorization + AI analysis)."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

from main_app.modules.colorizer import Colorizer  # type: ignore[import]
from main_app.modules.yolo_runner import YOLOWrapper  # type: ignore[import]
from main_app.modules.clip_runner import CLIPManager  # type: ignore[import]
from main_app.modules.ocr import OCRRunner  # type: ignore[import]
from main_app.modules.yolo_army import YOLOArmyDetector  # type: ignore[import]
from main_app.modules.llm import LLMInterpreter  # type: ignore[import]
from main_app.utils.summary_writer import SummaryWriter  # type: ignore[import]

# Import ChromaDB handler (optional)
try:
    from main_app.modules.chroma_handler import create_chroma_handler, is_chromadb_available  # type: ignore[import]
    CHROMADB_AVAILABLE = True
except ImportError:
    print("Note: ChromaDB not available. Install with: pip install chromadb")
    CHROMADB_AVAILABLE = False
    
    def create_chroma_handler(*args, **kwargs):
        return None
    
    def is_chromadb_available():
        return False


class ImageProcessor:
    """Handles single image processing through all AI models."""

    def __init__(
        self,
        yolo: YOLOWrapper,
        clip_mgr: CLIPManager,
        ocr: OCRRunner,
        yolo_army: YOLOArmyDetector,
        llm: LLMInterpreter
    ):
        self.yolo = yolo
        self.clip_mgr = clip_mgr
        self.ocr = ocr
        self.yolo_army = yolo_army
        self.llm = llm

    def process_image(
        self,
        orig_name: str,
        color_path: str,
        input_path: Path,
        results_dir: Path,
        multi_photo: bool,
        last_llm_call_ts: float
    ) -> Tuple[Dict[str, Any], float]:
        """Process a single image through all analysis steps."""
        # YOLO detection
        yolo_result = self.yolo.analyze_image(color_path)
        person_boxes = yolo_result.get("person_boxes", [])

        # CLIP analysis
        clip_bg = self.clip_mgr.background_analysis(color_path)
        clip_people = self.clip_mgr.gender_on_person_crops(color_path, person_boxes)

        # Add gender labels to YOLO detections
        yolo_dets = self._add_gender_to_detections(
            yolo_result.get("detections", []),
            clip_people
        )

        # OCR analysis
        ocr_result = self.ocr.run_ocr(color_path)

        # Build record
        record = self._build_record(
            orig_name, color_path, yolo_result, yolo_dets,
            clip_bg, clip_people, ocr_result
        )

        # Save full record
        full_rec_path = results_dir / f"full_{Path(orig_name).stem}.json"
        self._save_json(record, full_rec_path)

        # LLM analysis
        new_llm_ts = self._process_llm(
            record, orig_name, input_path, results_dir,
            multi_photo, last_llm_call_ts
        )

        # Army detection if conditions met
        self._process_army_detection(record, clip_bg, yolo_result, color_path)

        return record, new_llm_ts

    def _add_gender_to_detections(
        self,
        detections: List[Dict[str, Any]],
        clip_people: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add gender information to person detections."""
        person_idx = 0

        for det in detections:
            if det.get('class_name') == 'person':
                if person_idx < len(clip_people):
                    p = clip_people[person_idx]
                    man_score = float(p.get('man', 0.0))
                    woman_score = float(p.get('woman', 0.0))
                    det['person_gender'] = {
                        'label': 'woman' if woman_score >= man_score else 'man',
                        'man': man_score,
                        'woman': woman_score
                    }

                # Add location info
                bbox = det.get('bbox', {})
                det['location'] = {
                    'x1': float(bbox.get('x1', 0.0)),
                    'y1': float(bbox.get('y1', 0.0)),
                    'x2': float(bbox.get('x2', 0.0)),
                    'y2': float(bbox.get('y2', 0.0)),
                    'center_x': float(bbox.get('center_x', 0.0)),
                    'center_y': float(bbox.get('center_y', 0.0))
                }
                person_idx += 1

        return detections

    def _build_record(
        self,
        orig_name: str,
        color_path: str,
        yolo_result: Dict[str, Any],
        yolo_dets: List[Dict[str, Any]],
        clip_bg: Dict[str, Any],
        clip_people: List[Dict[str, Any]],
        ocr_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the complete analysis record for an image."""
        clip_section = {
            "indoor_outdoor": clip_bg.get("indoor_outdoor"),
            "background_top": clip_bg.get("top_categories", []),
            "background_detections": clip_bg.get("detections", []),
            "people_gender": clip_people,
            "notes": "CLIP background categories (environment) and indoor/outdoor; gender = person crops."
        }

        return {
            "original_filename": orig_name,
            "colorized_path": str(color_path),
            "yolo": {
                "object_counts": yolo_result.get("object_counts", {}),
                "top_objects": yolo_result.get("top_objects", []),
                "detections": yolo_dets,
                "notes": "Detections from YOLO (bounding boxes and classes)."
            },
            "clip": clip_section,
            "ocr": {
                "has_text": bool(ocr_result.get("has_text")),
                "lines": ocr_result.get("lines", []),
                "items": ocr_result.get("items", [])
            }
        }

    def _save_json(self, data: Dict[str, Any], path: Path) -> None:
        """Save data to JSON file."""
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _process_llm(
        self,
        record: Dict[str, Any],
        orig_name: str,
        input_path: Path,
        results_dir: Path,
        multi_photo: bool,
        last_llm_call_ts: float
    ) -> float:
        """Process LLM analysis with rate limiting."""
        try:
            if getattr(self.llm, 'client', None) is None:
                return last_llm_call_ts

            # Rate limiting for multiple photos
            if multi_photo and last_llm_call_ts > 0:
                since = time.time() - last_llm_call_ts
                wait_s = max(0.0, 45.0 - since)
                if wait_s > 0:
                    time.sleep(wait_s)

            print("\n\n--- LLM INTERPRETATION (Groq) ---\n")
            # orig_name comes from colorizer which uses full paths from selected files
            # selected files come from input_path.rglob() which gives paths relative to input_path
            # So orig_name should already be the correct relative path from input_path
            orig_image_path = orig_name  # Just use orig_name as-is since it's already the full path
            llm_output_path = str(results_dir / f"llm_{Path(orig_name).stem}_orig.json")
            print(f"🔍 [PIPELINE DEBUG] About to call LLM analyze with:")
            print(f"🔍 [PIPELINE DEBUG]   orig_name: {orig_name}")
            print(f"🔍 [PIPELINE DEBUG]   orig_image_path: {orig_image_path}")
            print(f"🔍 [PIPELINE DEBUG]   llm_output_path: {llm_output_path}")
            print(f"🔍 [PIPELINE DEBUG]   Path exists: {Path(orig_image_path).exists()}")
            print(f"🔍 [PIPELINE DEBUG]   LLM client status: {getattr(self.llm, 'client', None) is not None}")
            llm_response = self.llm.analyze(orig_image_path, save_path=llm_output_path)
            print(f"🔍 [PIPELINE DEBUG] LLM analyze returned: {type(llm_response)} ({len(str(llm_response)) if llm_response else 0} chars)")

            # Attach to record
            try:
                existing_llm_obj = record.get("llm")
                llm_section_obj: Dict[str, Any] = (
                    existing_llm_obj if isinstance(existing_llm_obj, dict) else {}
                )
                llm_section_obj["output_path"] = llm_output_path
                if isinstance(llm_response, str) and llm_response.strip():
                    llm_section_obj["json"] = llm_response
                record["llm"] = llm_section_obj
                print(f"✅ [PIPELINE DEBUG] LLM results attached to record successfully")
            except Exception as e:
                print(f"❌ [PIPELINE DEBUG] Failed to attach LLM results to record: {str(e)}")
                print(f"❌ [PIPELINE DEBUG] LLM response type: {type(llm_response)}")
                print(f"❌ [PIPELINE DEBUG] LLM response content: {str(llm_response)[:200]}...")

            print("\n--- END LLM INTERPRETATION ---\n")
            return time.time()

        except Exception as e:
            print(f"❌ [LLM DEBUG] Exception in _process_llm: {str(e)}")
            print(f"❌ [LLM DEBUG] Exception type: {type(e).__name__}")
            import traceback
            print(f"❌ [LLM DEBUG] Full traceback:")
            traceback.print_exc()
            return last_llm_call_ts

    def _process_army_detection(
        self,
        record: Dict[str, Any],
        clip_bg: Dict[str, Any],
        yolo_result: Dict[str, Any],
        color_path: str
    ) -> None:
        """Process army object detection if conditions are met."""
        env_top = [c.lower() for c in clip_bg.get("top_categories", [])]
        is_field_or_forest = any(k in env_top for k in ["field", "forest"])
        num_persons = yolo_result.get("object_counts", {}).get("person", 0)

        if num_persons > 1 and is_field_or_forest:
            army_det = self.yolo_army.analyze_image(color_path)
            clip_section = record["clip"]
            clip_section["army"] = any(v > 0 for v in army_det.get("object_counts", {}).values())
            clip_section["army_objects"] = army_det.get("object_counts", {})


class PhotoSelector:
    """Handles photo selection logic."""

    @staticmethod
    def quick_count_estimate(input_path: Path) -> int:
        """Quick estimate of photos without full scan."""
        if not input_path.exists():
            return 0
        
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".JPG", ".JPEG", ".PNG"}
        count = 0
        
        # Sample first few directories for quick estimate
        try:
            for root_item in input_path.iterdir():
                if root_item.is_file() and root_item.suffix in exts:
                    count += 1
                elif root_item.is_dir():
                    # Quick sample of first subdirectory
                    try:
                        subcount = sum(1 for p in root_item.iterdir() 
                                     if p.is_file() and p.suffix in exts)
                        count += subcount
                        if count > 50:  # Stop estimating after reasonable sample
                            return count
                    except:
                        continue
        except:
            pass
        return count

    @staticmethod
    def select_photos(input_path: Path) -> tuple[List[str], int]:
        """Interactive photo selection with processed photo tracking."""
        # Load processed photos from index
        processed_index_path = Path("main_app_outputs/results/processed_index.csv")
        already_processed = set()
        
        if processed_index_path.exists():
            try:
                import csv
                with open(processed_index_path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('file_path'):
                            already_processed.add(row['file_path'])
            except Exception:
                pass
        
        # Scan for all photos
        exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".JPG", ".JPEG", ".PNG"]
        all_files = [
            str(p) for p in sorted(input_path.rglob("*"))
            if p.suffix in exts and p.is_file()
        ]
        
        # Filter out already processed photos
        unprocessed_files = [f for f in all_files if f not in already_processed]
        
        # Show status
        total_photos = len(all_files)
        processed_count = len(already_processed)
        unprocessed_count = len(unprocessed_files)
        
        print(f"📊 Photo Status:")
        print(f"   Total photos found: {total_photos}")
        print(f"   Already processed: {processed_count}")
        print(f"   Available to process: {unprocessed_count}")
        
        if unprocessed_count == 0:
            print("🎉 All photos have been processed!")
            return [], 1
        
        print(f"\nChoose how many photos to process:")
        print("1) One random photo")
        print("2) Five random photos") 
        print("3) Custom number")
        print("4) All remaining photos")
        choice = input("Enter 1, 2, 3, or 4: ").strip()

        import random
        if choice == '1':
            selected = random.sample(unprocessed_files, 1) if unprocessed_files else []
            return selected, 1  # processing_mode: 1 = small batch
        elif choice == '2':
            k = min(5, len(unprocessed_files))
            selected = random.sample(unprocessed_files, k) if unprocessed_files else []
            return selected, 1  # processing_mode: 1 = small batch
        elif choice == '3':
            # Custom number
            while True:
                try:
                    custom_count = int(input(f"Enter number of photos to process (1-{unprocessed_count}): ").strip())
                    if 1 <= custom_count <= unprocessed_count:
                        break
                    else:
                        print(f"Please enter a number between 1 and {unprocessed_count}")
                except ValueError:
                    print("Please enter a valid number")
            
            selected = random.sample(unprocessed_files, custom_count)
            # Use small batch mode for ≤10, large batch for >10
            processing_mode = 1 if custom_count <= 10 else 2
            return selected, processing_mode
        else:
            return unprocessed_files, 2  # processing_mode: 2 = large batch (32 at a time)


class CSVExporter:
    """Handles CSV export functionality."""

    @staticmethod
    def create_csv_rows(
        per_image: List[Dict[str, Any]],
        input_path: Path,
        results_dir: Path
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Create rows for text and full CSV files."""
        rows_for_text: List[Dict[str, str]] = []
        rows_for_full: List[Dict[str, str]] = []

        for record in per_image:
            orig_name = record.get("original_filename", "")

            # Text CSV row
            text_row = CSVExporter._build_text_row(record, input_path, orig_name, results_dir)
            rows_for_text.append(text_row)

            # Full CSV row
            full_row = CSVExporter._build_full_row(record, input_path, orig_name, results_dir)
            rows_for_full.append(full_row)

        return rows_for_text, rows_for_full

    @staticmethod
    def _build_text_row(record: Dict[str, Any], input_path: Path, orig_name: str, results_dir: Path) -> Dict[str, str]:
        """Build text CSV row with comprehensive description for embedding."""
        comprehensive_text = CSVExporter._build_comprehensive_description(record, input_path)
        original_for_csv = record.get('processed_path') or str((orig_name if Path(orig_name).is_absolute() else (input_path / orig_name)))

        return {
            'original_path': original_for_csv,
            'comprehensive_text': comprehensive_text
        }

    @staticmethod
    def _build_full_row(record: Dict[str, Any], input_path: Path, orig_name: str, results_dir: Path) -> Dict[str, str]:
        """Build full CSV row with all fields."""
        # Extract data
        clip_obj = record.get('clip', {})
        yolo_result = record.get('yolo', {})

        # Gender counts
        genders = clip_obj.get('people_gender', [])
        women = sum(1 for g in genders if isinstance(g, dict) and CSVExporter._is_woman(g))
        men = len(genders) - women

        # Object counts
        obj_counts = yolo_result.get('object_counts', {})
        obj_counts_str = '; '.join(f"{k}:{v}" for k, v in sorted(obj_counts.items()))

        # Background
        bg_list = clip_obj.get('background_top', [])
        bg_first = bg_list[0] if bg_list else ''

        # LLM fields
        llm_fields = CSVExporter._extract_llm_fields(record)

        full_rec_path = results_dir / f"full_{Path(orig_name).stem}.json"

        # Extract path words (folders + filename without extension)
        def _extract_path_words(path_str: str) -> str:
            import re as _re
            p = Path(path_str)
            parts = [seg for seg in p.parts if seg not in ['/', '']]
            words: list[str] = []
            for seg in parts:
                base = seg.rsplit('.', 1)[0]
                tokens = _re.findall(r"[\w]+", base, flags=_re.UNICODE)
                for t in tokens:
                    tt = t.strip()
                    if tt:
                        words.append(tt)
            return ' '.join(words)

        source_path = record.get('source_path') or str((orig_name if Path(orig_name).is_absolute() else (input_path / orig_name)))
        processed_path = record.get('processed_path') or source_path
        path_words = _extract_path_words(processed_path)
        source_path_words = _extract_path_words(source_path)

        return {
            'original_path': processed_path,
            'source_path': source_path,
            'colorized_path': '',
            'indoor_outdoor': clip_obj.get('indoor_outdoor', ''),
            'background': bg_first,
            'yolo_object_counts': obj_counts_str,
            'men': str(men),
            'women': str(women),
            **llm_fields,
            'path_words': path_words,
            'source_path_words': source_path_words,
            'full_results_path': str(full_rec_path),
            'llm_json_path': llm_fields.get('llm_json_path', '')
        }

    @staticmethod
    def _is_woman(g: Dict[str, Any]) -> bool:
        """Check if gender detection indicates woman."""
        m = float(g.get('man', 0.0))
        w = float(g.get('woman', 0.0))
        return w >= m

    @staticmethod
    def _build_comprehensive_description(record: Dict[str, Any], input_path: Path) -> str:
        """Build comprehensive description string combining all extracted data for embedding."""
        try:
            text_parts = []
            
            # 1. LLM CAPTION (first priority)
            llm_obj = record.get("llm")
            llm_parsed = {}
            if isinstance(llm_obj, dict):
                llm_json_str = llm_obj.get("json")
                if isinstance(llm_json_str, str) and llm_json_str.strip():
                    try:
                        llm_parsed = json.loads(llm_json_str)
                        caption = llm_parsed.get('caption', '')
                        if caption:
                            text_parts.append(caption)
                    except Exception:
                        pass
            
            # If no LLM caption, create basic description from filename
            if not text_parts:
                orig_name = record.get("original_filename", "")
                if orig_name:
                    filename = Path(orig_name).stem
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
                if isinstance(bg, dict):
                    category = bg.get('category', '')
                    if category:
                        text_parts.append(f"Background: {category}")
            
            # People gender from CLIP
            people_gender = clip_data.get('people_gender', [])
            men_count = sum(1 for p in people_gender if isinstance(p, dict) and p.get('man', 0) >= p.get('woman', 0))
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
                        if isinstance(symbol, dict):
                            symbol_type = symbol.get('symbol_type', '')
                            if symbol_type:
                                text_parts.append(symbol_type)
                
                if llm_parsed.get('has_nazi_symbols', False):
                    text_parts.append("Nazi symbols present")
                    nazi_symbols = llm_parsed.get('nazi_symbols_details', [])
                    for symbol in nazi_symbols:
                        if isinstance(symbol, dict):
                            symbol_type = symbol.get('symbol_type', '')
                            if symbol_type:
                                text_parts.append(symbol_type)
                
                # Objects and artifacts
                objects = llm_parsed.get('main_objects_artifacts_animals', [])
                for obj in objects:
                    if isinstance(obj, dict):
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
            
            # 6. OCR text
            ocr_data = record.get('ocr', {})
            if ocr_data.get('has_text', False):
                ocr_lines = ocr_data.get('lines', [])
                for line in ocr_lines[:2]:
                    if line and line.strip():
                        text_parts.append(f"OCR text: {line.strip()}")
            
            # 7. Collection context from folder path
            orig_name = record.get("original_filename", "")
            if orig_name:
                try:
                    path_parts = Path(orig_name).parts
                    if len(path_parts) > 2 and path_parts[0] == "photo_collections" and path_parts[1] == "project Photography":
                        folder_name = path_parts[2]
                        folder_context = f"Collection: {folder_name.replace(' – ', ' - ').replace('_', ' ')}"
                        text_parts.append(folder_context)
                except Exception:
                    pass
            
            # Combine all text parts into single comprehensive description
            comprehensive_text = ' '.join([part for part in text_parts if part and part.strip()])
            
            # Fallback if no data found
            if not comprehensive_text:
                comprehensive_text = f"Historical photograph from {Path(orig_name).name}" if orig_name else "Historical photograph"
            
            return comprehensive_text
            
        except Exception:
            return "Historical photograph"

    @staticmethod
    def _build_description(record: Dict[str, Any], input_path: Path) -> str:
        """Build description string from record data."""
        try:
            parts = []

            # LLM caption and details
            llm_obj = record.get("llm")
            if isinstance(llm_obj, dict):
                llm_json_str = llm_obj.get("json")
                if isinstance(llm_json_str, str) and llm_json_str.strip():
                    try:
                        lj = json.loads(llm_json_str)
                        caption = lj.get('caption')
                        if isinstance(caption, str):
                            parts.append(caption.strip())

                        # Add symbol details
                        CSVExporter._add_symbol_details(parts, lj)
                        CSVExporter._add_text_details(parts, lj)
                        CSVExporter._add_object_details(parts, lj)

                    except Exception:
                        pass

            # Gender counts - only include non-zero counts
            clip_obj = record.get('clip', {})
            genders = clip_obj.get('people_gender', [])
            women = sum(1 for g in genders if isinstance(g, dict) and CSVExporter._is_woman(g))
            men = len(genders) - women
            gender_parts = []
            if men > 0:
                gender_parts.append(f"{men} men")
            if women > 0:
                gender_parts.append(f"{women} women")
            if gender_parts:
                parts.append(" ".join(gender_parts))

            # Location info
            io = clip_obj.get('indoor_outdoor', 'unknown')
            if io:
                parts.append(io)

            bg = clip_obj.get('background_top', [])
            if bg and isinstance(bg[0], str):
                parts.append(bg[0])

            # Append path words from original path
            orig_name = record.get("original_filename", "")
            source_path = record.get('source_path') or str((orig_name if Path(orig_name).is_absolute() else (input_path / orig_name)))
            def _extract_path_words(path_str: str) -> str:
                import re as _re
                p = Path(path_str)
                parts2 = [seg for seg in p.parts if seg not in ['/', '']]
                words2: list[str] = []
                for seg in parts2:
                    base = seg.rsplit('.', 1)[0]
                    tokens = _re.findall(r"[\w]+", base, flags=_re.UNICODE)
                    for t in tokens:
                        tt = t.strip()
                        if tt:
                            words2.append(tt)
                return ' '.join(words2)
            path_words = _extract_path_words(source_path)

            body = '. '.join([p for p in parts if isinstance(p, str) and p])
            return body + (". " + path_words if path_words else "")

        except Exception:
            return ''

    @staticmethod
    def _add_symbol_details(parts: List[str], lj: Dict[str, Any]) -> None:
        """Add Jewish and Nazi symbol details to description."""
        for symbol_type in ['jewish_symbols_details', 'nazi_symbols_details']:
            if bool(lj.get(f'has_{symbol_type.split("_")[0]}_symbols', False)):
                details = lj.get(symbol_type, [])
                if isinstance(details, list) and details:
                    descs = []
                    for s in details[:5]:
                        if isinstance(s, dict):
                            st = s.get('symbol_type')
                            sd = s.get('description')
                            seg = [x.strip() for x in [st, sd] if isinstance(x, str) and x.strip()]
                            if seg:
                                descs.append(' - '.join(seg))
                    if descs:
                        parts.append('; '.join(descs))

    @staticmethod
    def _add_text_details(parts: List[str], lj: Dict[str, Any]) -> None:
        """Add Hebrew and German text details to description."""
        ta = lj.get('text_analysis', {})
        if not isinstance(ta, dict):
            return

        for lang in ['hebrew_text', 'german_text']:
            lang_data = ta.get(lang, {})
            if isinstance(lang_data, dict) and bool(lang_data.get('present', False)):
                txt = lang_data.get('text_found')
                tr = lang_data.get('translation')
                sub = [x.strip() for x in [txt, tr] if isinstance(x, str) and x.strip()]
                if sub:
                    parts.append(' - '.join(sub))

    @staticmethod
    def _add_object_details(parts: List[str], lj: Dict[str, Any]) -> None:
        """Add object details to description."""
        objs = lj.get('main_objects_artifacts_animals', [])
        if isinstance(objs, list) and objs:
            od = []
            for o in objs[:5]:
                if isinstance(o, dict):
                    it = o.get('item')
                    ds = o.get('description')
                    if isinstance(it, str) and isinstance(ds, str) and it.strip() and ds.strip():
                        od.append(f"{it.strip()}: {ds.strip()}")
            if od:
                parts.append('; '.join(od))

    @staticmethod
    def _extract_llm_fields(record: Dict[str, Any]) -> Dict[str, str]:
        """Extract LLM fields for full CSV."""
        fields = {
            'llm_caption': '',
            'llm_people_under_18': '',
            'llm_jewish_symbols': '',
            'llm_nazi_symbols': '',
            'hebrew_present': '',
            'hebrew_text': '',
            'hebrew_translation': '',
            'german_present': '',
            'german_text': '',
            'german_translation': '',
            'violence': '',
            'violence_explanation': '',
            'llm_objects': '',
            'llm_json_path': ''
        }

        llm_obj = record.get("llm")
        if not isinstance(llm_obj, dict):
            return fields

        fields['llm_json_path'] = llm_obj.get('output_path', '')
        llm_json_str = llm_obj.get('json')

        if not isinstance(llm_json_str, str) or not llm_json_str.strip():
            return fields

        try:
            lj = json.loads(llm_json_str)

            # Basic fields
            if isinstance(lj.get('caption'), str):
                fields['llm_caption'] = lj.get('caption')
            if isinstance(lj.get('people_under_18'), (int, float)):
                fields['llm_people_under_18'] = str(int(lj.get('people_under_18')))

            # Symbol details
            CSVExporter._extract_symbol_fields(fields, lj)

            # Text analysis
            CSVExporter._extract_text_fields(fields, lj)

            # Violence assessment
            CSVExporter._extract_violence_fields(fields, lj)

            # Objects
            CSVExporter._extract_object_fields(fields, lj)

        except Exception:
            pass

        return fields

    @staticmethod
    def _extract_symbol_fields(fields: Dict[str, str], lj: Dict[str, Any]) -> None:
        """Extract symbol-related fields."""
        for prefix in ['jewish', 'nazi']:
            if bool(lj.get(f'has_{prefix}_symbols', False)):
                det = lj.get(f'{prefix}_symbols_details', [])
                if isinstance(det, list):
                    parts = []
                    for s in det[:5]:
                        if isinstance(s, dict):
                            seg = []
                            for key in ['symbol_type', 'description', 'location_in_image']:
                                val = s.get(key)
                                if isinstance(val, str) and val.strip():
                                    seg.append(val.strip())
                            if seg:
                                parts.append(' - '.join(seg))
                    fields[f'llm_{prefix}_symbols'] = '; '.join(parts)

    @staticmethod
    def _extract_text_fields(fields: Dict[str, str], lj: Dict[str, Any]) -> None:
        """Extract text analysis fields."""
        ta = lj.get('text_analysis', {})
        if not isinstance(ta, dict):
            return

        for lang in ['hebrew', 'german']:
            lang_data = ta.get(f'{lang}_text', {})
            if isinstance(lang_data, dict):
                fields[f'{lang}_present'] = 'true' if bool(lang_data.get('present', False)) else 'false'

                for field in ['text_found', 'translation']:
                    val = lang_data.get(field)
                    if isinstance(val, str):
                        key = f'{lang}_text' if field == 'text_found' else f'{lang}_translation'
                        fields[key] = val

    @staticmethod
    def _extract_violence_fields(fields: Dict[str, str], lj: Dict[str, Any]) -> None:
        """Extract violence assessment fields."""
        va = lj.get('violence_assessment', {})
        if isinstance(va, dict):
            fields['violence'] = 'true' if bool(va.get('signs_of_violence', False)) else 'false'
            exp = va.get('explanation')
            if isinstance(exp, str):
                fields['violence_explanation'] = exp

    @staticmethod
    def _extract_object_fields(fields: Dict[str, str], lj: Dict[str, Any]) -> None:
        """Extract object fields."""
        objs = lj.get('main_objects_artifacts_animals', [])
        if isinstance(objs, list):
            parts = []
            for o in objs[:5]:
                if isinstance(o, dict):
                    item = o.get('item')
                    desc = o.get('description')
                    if isinstance(item, str) and isinstance(desc, str) and item.strip() and desc.strip():
                        parts.append(f"{item.strip()}: {desc.strip()}")
            fields['llm_objects'] = '; '.join(parts)


def process_small_batch(
    selected: List[str],
    colorizer: 'Colorizer',
    processor: ImageProcessor,
    input_path: Path,
    colorized_dir: Path,
    results_dir: Path,
    ab_boost: float,
    multi_photo: bool
) -> List[Dict[str, Any]]:
    """Process small batches (1-5 photos) efficiently: colorize all → process all → cleanup."""
    print(f"🎨 Step 1/3: Colorizing {len(selected)} photos...")
    
    # Colorize all photos first
    colorized_map: Dict[str, str] = colorizer.colorize_files(
        files=selected,
        output_directory=str(colorized_dir),
        ab_boost=ab_boost
    )
    
    print(f"🤖 Step 2/3: Running AI analysis on {len(colorized_map)} photos...")
    
    # Process all photos
    per_image: List[Dict[str, Any]] = []
    last_llm_call_ts: float = 0.0
    
    for i, (orig_name, color_path) in enumerate(sorted(colorized_map.items()), 1):
        print(f"[{i}/{len(colorized_map)}] Processing {Path(orig_name).name}...")
        
        # Just track the original path - no copying needed  
        # orig_name comes from colorized_map keys which are the full file paths from selected
        src_path = orig_name  # orig_name is already the full path
        processed_path = src_path  # Keep same path, just track as processed
        
        # Process image (LLM will use the source path while it still exists)
        record, last_llm_call_ts = processor.process_image(
            orig_name, color_path, input_path, results_dir,
            multi_photo, last_llm_call_ts
        )
        
        # Update record with file paths (no file operations needed)
        record['processed_path'] = processed_path
        record['source_path'] = src_path
        print(f"  ✅ Processed {Path(src_path).name} (original kept in place)")
        
        per_image.append(record)
    
    print(f"🧹 Step 3/3: Cleaning up {len(colorized_map)} colorized files...")
    
    # Clean up all colorized files
    for color_path in colorized_map.values():
        try:
            import os
            if os.path.isfile(color_path):
                os.remove(color_path)
        except Exception:
            pass
    
    print(f"✅ Small batch processing complete: {len(per_image)} photos processed")
    return per_image


def run_main_pipeline(
    input_dir: str,
    output_dir: str,
    models_dir: str,
    ab_boost: float,
    yolo_model_size: str,
    clip_model_name: str,
    confidence_yolo: float,
    confidence_clip: float
) -> Dict[str, Any]:
    """Run the main image processing pipeline."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    colorized_dir = output_path / "colorized"
    results_dir = output_path / "results"
    colorized_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Select photos with processing mode
    selected, processing_mode = PhotoSelector.select_photos(input_path)
    if not selected:
        return {"error": "No photos selected"}

    # Initialize components
    colorizer = Colorizer(models_dir=models_dir)
    yolo = YOLOWrapper(model_size=yolo_model_size, base_confidence=confidence_yolo)
    clip_mgr = CLIPManager(model_name=clip_model_name)
    ocr = OCRRunner()
    yolo_army = YOLOArmyDetector(weights_path=os.path.join("yolo_detection", "yolo_army.pt"))
    llm = LLMInterpreter()

    processor = ImageProcessor(yolo, clip_mgr, ocr, yolo_army, llm)

    # Process based on mode
    per_image: List[Dict[str, Any]] = []
    total_to_process = len(selected)
    multi_photo = total_to_process > 1
    last_llm_call_ts: float = 0.0
    processed_in_run = 0
    
    if processing_mode == 1:
        # Small batch mode (1 or 5 photos): colorize all → process all → cleanup
        print(f"🎯 Small batch mode: Processing {total_to_process} photos efficiently")
        per_image = process_small_batch(
            selected, colorizer, processor, input_path, colorized_dir, 
            results_dir, ab_boost, multi_photo
        )
    else:
        # Large batch mode: process in chunks of 32
        print(f"🔄 Large batch mode: Processing {total_to_process} photos in batches of 32")
        try:
            # Global total in project (recursive)
            exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".JPG", ".JPEG", ".PNG"]
            global_total = len([1 for p in input_path.rglob("*") if p.suffix in exts and p.is_file()])
        except Exception:
            global_total = total_to_process

        BATCH_SIZE = 32
        for b_start in range(0, len(selected), BATCH_SIZE):
            batch = selected[b_start:b_start + BATCH_SIZE]
            # Colorize batch (reuse existing colorized files when available)
            colorized_map: Dict[str, str] = {}
            newly_created: Dict[str, str] = {}
            try:
                # Pre-fill from existing colorized files
                for fp in batch:
                    try:
                        p = Path(fp)
                        out_path = colorized_dir / f"colorized_{p.stem}{p.suffix}"
                        if out_path.exists() and out_path.is_file():
                            colorized_map[str(p)] = str(out_path)
                    except Exception:
                        continue

                # Determine which need colorization
                to_colorize = [fp for fp in batch if str(fp) not in colorized_map]
                if to_colorize:
                    created_map = colorizer.colorize_files(
                        files=to_colorize,
                        output_directory=str(colorized_dir),
                        ab_boost=ab_boost
                    )
                    newly_created.update(created_map)
                    colorized_map.update(created_map)
            except Exception:
                pass

            # Process each image in batch
            for orig_name, color_path in sorted(colorized_map.items()):
                # Progress: start of photo
                current_idx = processed_in_run + 1
                try:
                    start_name = Path(orig_name).name
                except Exception:
                    start_name = str(orig_name)
                print(f"[START] {current_idx}/{total_to_process} {start_name}")

                record, last_llm_call_ts = processor.process_image(
                    orig_name, color_path, input_path, results_dir,
                    multi_photo, last_llm_call_ts
                )
                # Just track the original path - no file operations needed
                # orig_name is already the full path from colorized_map keys
                src_path = orig_name
                record['processed_path'] = src_path
                record['source_path'] = src_path
                # Remove colorized artifact immediately only if we just created it in this run
                try:
                    import os as _os
                    if str(orig_name) in newly_created:
                        if _os.path.isfile(color_path):
                            _os.remove(color_path)
                except Exception:
                    pass
                per_image.append(record)
                # Progress line with 200 OK and details
                processed_in_run += 1
                try:
                    filename = Path(record.get('original_filename', start_name)).name
                except Exception:
                    filename = record.get('original_filename', start_name)
                try:
                    clip_obj = record.get('clip', {})
                    io = clip_obj.get('indoor_outdoor', 'unknown')
                    bg_top = clip_obj.get('background_top', [])
                    bg_first = bg_top[0] if isinstance(bg_top, list) and bg_top else ''
                    yolo_obj = record.get('yolo', {})
                    detections = yolo_obj.get('detections', []) if isinstance(yolo_obj, dict) else []
                    objs = len(detections)
                    genders = clip_obj.get('people_gender', []) if isinstance(clip_obj, dict) else []
                    women = sum(1 for g in genders if isinstance(g, dict) and float(g.get('woman', 0.0)) >= float(g.get('man', 0.0)))
                    men = len(genders) - women
                    ocr_obj = record.get('ocr', {})
                    has_text = bool(ocr_obj.get('has_text')) if isinstance(ocr_obj, dict) else False
                    processed_path = record.get('processed_path', '')
                    print(f"[200 OK] {processed_in_run}/{total_to_process} {filename} | IO:{io} BG:{bg_first} objs:{objs} people:{men+women} (m:{men}/w:{women}) OCR:{str(has_text).lower()} | pooled:{processed_path}")
                except Exception:
                    print(f"[200 OK] Processed {processed_in_run}/{total_to_process} | Project total: {global_total}")

    # Write summaries
    summary_writer = SummaryWriter()
    json_out = results_dir / "summary.json"
    txt_out = results_dir / "summary.txt"
    summary_writer.write_json(per_image, str(json_out))
    summary_writer.write_text(per_image, str(txt_out))

    # Export CSV files
    export_csv_files(per_image, input_path, results_dir)

    # Store in ChromaDB (optional)
    if CHROMADB_AVAILABLE and per_image:
        print("\n💾 Storing analysis data in ChromaDB...")
        try:
            chroma_handler = create_chroma_handler()
            if chroma_handler:
                photo_ids = chroma_handler.batch_store_photos(per_image)
                print(f"✅ Stored {len(photo_ids)} photos in ChromaDB")
                
                # Print collection stats
                stats = chroma_handler.get_collection_stats()
                print(f"📊 ChromaDB Collection: {stats['total_photos']} total photos")
            else:
                print("❌ Failed to create ChromaDB handler")
        except Exception as e:
            print(f"⚠️  ChromaDB storage failed: {e}")
            print("   (Processing continues with CSV storage only)")

    # Write simple processed index
    try:
        import csv
        idx_path = results_dir / 'processed_index.csv'
        
        # Load existing processed files
        existing = set()
        if idx_path.exists():
            with open(idx_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('file_path'):
                        existing.add(row['file_path'])
        
        # Add newly processed files
        for rec in per_image:
            file_path = rec.get('source_path', '')
            if file_path:
                existing.add(file_path)
        
        # Write updated index
        with open(idx_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file_path'])
            writer.writeheader()
            for path in sorted(existing):
                writer.writerow({'file_path': path})
                
    except Exception:
        pass

    return {
        "output_dir": str(output_path),
        "results": {
            "json": str(json_out),
            "text": str(txt_out)
        },
        "num_images": len(per_image)
    }


def export_csv_files(per_image: List[Dict[str, Any]], input_path: Path, results_dir: Path) -> None:
    """Export CSV files with proper error handling."""
    try:
        import csv

        rows_for_text, rows_for_full = CSVExporter.create_csv_rows(per_image, input_path, results_dir)

        # Export text CSV
        csv_path_text = results_dir / 'data_text.csv'
        text_fields = ['original_path', 'comprehensive_text']
        write_csv_with_merge(csv_path_text, rows_for_text, text_fields)

        # Export full CSV
        csv_path_full = results_dir / 'data_full.csv'
        if rows_for_full:
            full_fields = list(rows_for_full[0].keys())
            write_csv_with_merge(csv_path_full, rows_for_full, full_fields)

    except Exception:
        pass


def write_csv_with_merge(
    csv_path: Path,
    new_rows: List[Dict[str, str]],
    field_names: List[str]
) -> None:
    """Write CSV file, merging with existing data."""
    import csv

    existing: Dict[str, Dict[str, str]] = {}

    # Load existing data
    if csv_path.exists():
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

    # Merge new rows
    for r in new_rows:
        key = (r.get('original_path') or '').strip()
        if key:
            existing[key] = {**existing.get(key, {}), **r}

    # Write merged data
    if field_names and existing:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for key in sorted(existing.keys()):
                row = existing[key]
                writer.writerow({fn: row.get(fn, '') for fn in field_names})


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Main app: colorize photos, run YOLO + CLIP, detect symbols")
    parser.add_argument("--input_dir", default=os.path.join("sample_photos"), help="Directory of input photos")
    parser.add_argument("--output_dir", default=os.path.join("main_app_outputs"), help="Directory to write outputs")
    parser.add_argument("--models_dir", default=os.path.join("opencv_analysis", "models"), help="OpenCV colorization models dir")
    parser.add_argument("--ab_boost", type=float, default=1.0, help="Saturation boost for colorization ab channels")
    parser.add_argument("--yolo_model_size", default="s", help="Ultralytics YOLOv8 size: n/s/m/l/x")
    parser.add_argument("--clip_model_name", default="ViT-L/14@336px", help="CLIP model name")
    parser.add_argument("--confidence_yolo", type=float, default=0.4, help="YOLO base confidence")
    parser.add_argument("--confidence_clip", type=float, default=0.3, help="CLIP confidence for scene selections")

    args = parser.parse_args()

    result = run_main_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        ab_boost=args.ab_boost,
        yolo_model_size=args.yolo_model_size,
        clip_model_name=args.clip_model_name,
        confidence_yolo=args.confidence_yolo,
        confidence_clip=args.confidence_clip,
    )

    print(json.dumps({"status": "ok", **result}, indent=2))


if __name__ == "__main__":
    main()
