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
            orig_image_path = str(input_path / orig_name)
            llm_output_path = str(results_dir / f"llm_{Path(orig_image_path).stem}_orig.json")
            llm_response = self.llm.analyze(orig_image_path, save_path=llm_output_path)

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
            except Exception:
                pass

            print("\n--- END LLM INTERPRETATION ---\n")
            return time.time()

        except Exception:
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
    def select_photos(input_path: Path) -> List[str]:
        """Interactive photo selection."""
        print("Choose how many photos to process:")
        print("1) One random photo")
        print("2) Five random photos")
        print("3) All photos")
        choice = input("Enter 1, 2, or 3: ").strip()

        exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".JPG", ".JPEG", ".PNG"]
        all_files = [
            str(p) for p in sorted(input_path.iterdir())
            if p.suffix in exts and p.is_file()
        ]

        import random
        if choice == '1':
            return random.sample(all_files, 1) if all_files else []
        elif choice == '2':
            k = min(5, len(all_files))
            return random.sample(all_files, k) if all_files else []
        else:
            return all_files


class CSVExporter:
    """Handles CSV export functionality."""

    @staticmethod
    def create_csv_rows(
        per_image: List[Dict[str, Any]],
        input_path: Path
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Create rows for text and full CSV files."""
        rows_for_text: List[Dict[str, str]] = []
        rows_for_full: List[Dict[str, str]] = []

        for record in per_image:
            orig_name = record.get("original_filename", "")

            # Text CSV row
            text_row = CSVExporter._build_text_row(record, input_path, orig_name)
            rows_for_text.append(text_row)

            # Full CSV row
            full_row = CSVExporter._build_full_row(record, input_path, orig_name)
            rows_for_full.append(full_row)

        return rows_for_text, rows_for_full

    @staticmethod
    def _build_text_row(record: Dict[str, Any], input_path: Path, orig_name: str) -> Dict[str, str]:
        """Build text CSV row with description."""
        description = CSVExporter._build_description(record)
        results_dir = input_path.parent / "main_app_outputs" / "results"
        full_rec_path = results_dir / f"full_{Path(orig_name).stem}.json"

        llm_obj = record.get("llm") if isinstance(record.get("llm"), dict) else None

        return {
            'original_path': str(input_path / orig_name),
            'colorized_path': record.get("colorized_path", ""),
            'full_results_path': str(full_rec_path),
            'llm_json_path': llm_obj.get('output_path') if isinstance(llm_obj, dict) else '',
            'description': description
        }

    @staticmethod
    def _build_full_row(record: Dict[str, Any], input_path: Path, orig_name: str) -> Dict[str, str]:
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

        results_dir = input_path.parent / "main_app_outputs" / "results"
        full_rec_path = results_dir / f"full_{Path(orig_name).stem}.json"

        return {
            'original_path': str(input_path / orig_name),
            'colorized_path': record.get("colorized_path", ""),
            'indoor_outdoor': clip_obj.get('indoor_outdoor', ''),
            'background': bg_first,
            'yolo_object_counts': obj_counts_str,
            'men': str(men),
            'women': str(women),
            **llm_fields,
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
    def _build_description(record: Dict[str, Any]) -> str:
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

            return '. '.join([p for p in parts if isinstance(p, str) and p])

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

    # Select photos
    selected = PhotoSelector.select_photos(input_path)
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

    # Colorize selected images
    colorized_map: Dict[str, str] = colorizer.colorize_files(
        files=selected,
        output_directory=str(colorized_dir),
        ab_boost=ab_boost
    )

    # Process each image
    per_image: List[Dict[str, Any]] = []
    multi_photo = len(colorized_map) > 1
    last_llm_call_ts: float = 0.0

    for orig_name, color_path in sorted(colorized_map.items()):
        record, last_llm_call_ts = processor.process_image(
            orig_name, color_path, input_path, results_dir,
            multi_photo, last_llm_call_ts
        )
        per_image.append(record)

    # Write summaries
    summary_writer = SummaryWriter()
    json_out = results_dir / "summary.json"
    txt_out = results_dir / "summary.txt"
    summary_writer.write_json(per_image, str(json_out))
    summary_writer.write_text(per_image, str(txt_out))

    # Export CSV files
    export_csv_files(per_image, input_path, results_dir)

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

        rows_for_text, rows_for_full = CSVExporter.create_csv_rows(per_image, input_path)

        # Export text CSV
        csv_path_text = results_dir / 'data_text.csv'
        text_fields = ['original_path', 'colorized_path', 'full_results_path', 'llm_json_path', 'description']
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
