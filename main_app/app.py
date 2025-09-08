import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List

from main_app.modules.colorizer import Colorizer  # type: ignore[import]
from main_app.modules.yolo_runner import YOLOWrapper  # type: ignore[import]
from main_app.modules.clip_runner import CLIPManager  # type: ignore[import]
from main_app.modules.ocr import OCRRunner  # type: ignore[import]
from main_app.modules.yolo_army import YOLOArmyDetector  # type: ignore[import]
from main_app.modules.llm import LLMInterpreter  # type: ignore[import]
"""Main pipeline app (colorize + YOLO + CLIP)."""
from main_app.utils.summary_writer import SummaryWriter  # type: ignore[import]


def run_pipeline(
    input_dir: str,
    output_dir: str,
    models_dir: str,
    ab_boost: float,
    yolo_model_size: str,
    clip_model_name: str,
    confidence_yolo: float,
    confidence_clip: float
) -> Dict[str, Any]:

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    colorized_dir = output_path / "colorized"
    results_dir = output_path / "results"
    processed_dir = output_path / "processed_photos"
    colorized_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # File deletion proposal list (do not delete automatically)
    files_to_delete_path = output_path / "files_to_delete.txt"

    # 1) Colorize
    colorizer = Colorizer(models_dir=models_dir)
    # Interactive selection of how many photos
    print("Choose how many photos to process:")
    print("1) One random photo")
    print("2) Five random photos")
    print("3) All photos")
    choice = input("Enter 1, 2, or 3: ").strip()

    # Collect candidates (recursive)
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".JPG", ".JPEG", ".PNG"]
    all_files = [str(p) for p in sorted(input_path.rglob("*")) if p.suffix in exts and p.is_file()]

    import random
    selected: List[str]
    if choice == '1':
        selected = random.sample(all_files, 1) if all_files else []
    elif choice == '2':
        k = min(5, len(all_files))
        selected = random.sample(all_files, k) if all_files else []
    else:
        selected = all_files

    # 1) Colorize selected
    colorized_map: Dict[str, str] = colorizer.colorize_files(
        files=selected,
        output_directory=str(colorized_dir),
        ab_boost=ab_boost
    )

    # 2) Initialize detectors
    yolo = YOLOWrapper(model_size=yolo_model_size, base_confidence=confidence_yolo)
    clip_mgr = CLIPManager(model_name=clip_model_name)
    ocr = OCRRunner()
    yolo_army = YOLOArmyDetector(weights_path=os.path.join("yolo_detection", "yolo_army.pt"))
    llm = LLMInterpreter()

    # 3) For each colorized image: run YOLO, CLIP scene, CLIP gender on person crops, and symbols
    per_image: List[Dict[str, Any]] = []
    multi_photo = len(colorized_map) > 1
    last_llm_call_ts: float = 0.0
    rows_for_text: List[Dict[str, Any]] = []
    rows_for_full: List[Dict[str, Any]] = []
    for orig_name, color_path in sorted(colorized_map.items()):
        # YOLO on colorized
        yolo_result = yolo.analyze_image(color_path)
        person_boxes = yolo_result.get("person_boxes", [])

        # CLIP background (with indoor/outdoor)
        clip_bg = clip_mgr.background_analysis(color_path)

        # CLIP gender on person crops
        clip_people = clip_mgr.gender_on_person_crops(color_path, person_boxes)

        # Attach gender label to each YOLO person detection with bbox and location
        # Tie-break to woman; otherwise choose higher score
        def pick_label(man_score: float, woman_score: float) -> str:
            return 'woman' if woman_score >= man_score else 'man'

        yolo_dets = yolo_result.get("detections", [])
        # Map by index order; both lists are aligned by person index order as produced
        person_idx = 0
        for det in yolo_dets:
            if det.get('class_name') == 'person':
                if person_idx < len(clip_people):
                    p = clip_people[person_idx]
                    man_score = float(p.get('man', 0.0))
                    woman_score = float(p.get('woman', 0.0))
                    det['person_gender'] = {
                        'label': pick_label(man_score, woman_score),
                        'man': man_score,
                        'woman': woman_score
                    }
                # Always expose location
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

        # OCR on colorized image
        ocr_result = ocr.run_ocr(color_path)

        # Build clip section first
        clip_section = {
            "indoor_outdoor": clip_bg.get("indoor_outdoor"),
            "background_top": clip_bg.get("top_categories", []),
            "background_detections": clip_bg.get("detections", []),
            "people_gender": clip_people,
            "notes": "CLIP background categories (environment) and indoor/outdoor; gender = person crops."
        }

        # Compose record
        record = {
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
            },
        }
        # Save full per-image record JSON path for CSV reference
        full_rec_path = results_dir / f"full_{Path(orig_name).stem}.json"
        try:
            with open(full_rec_path, 'w') as f:
                json.dump(record, f, indent=2)
        except Exception:
            pass

        per_image.append(record)

        # LLM interpretation (optional; uses .env GROQ_API_KEY). Runs once at end of each image processing.
        try:
            if getattr(llm, 'client', None) is not None:
                # If more than one photo, throttle to one call per 45 seconds
                if multi_photo and last_llm_call_ts > 0:
                    since = time.time() - last_llm_call_ts
                    wait_s = max(0.0, 45.0 - since)
                    if wait_s > 0:
                        time.sleep(wait_s)
                print("\n\n--- LLM INTERPRETATION (Groq) ---\n")
                # Support absolute or relative paths
                from pathlib import Path as _P
                orig_image_path = orig_name if _P(orig_name).is_absolute() else str(input_path / orig_name)
                llm_output_path = str((results_dir / f"llm_{Path(orig_image_path).stem}_orig.json"))
                llm_response = llm.analyze(orig_image_path, save_path=llm_output_path)
                # Attach to record so summary can include it
                try:
                    existing_llm_obj = record.get("llm")
                    llm_section_obj: Dict[str, Any] = existing_llm_obj if isinstance(existing_llm_obj, dict) else {}
                    llm_section_obj["output_path"] = llm_output_path
                    if isinstance(llm_response, str) and llm_response.strip():
                        llm_section_obj["json"] = llm_response
                    record["llm"] = llm_section_obj
                except Exception:
                    pass
                print("\n--- END LLM INTERPRETATION ---\n")
                last_llm_call_ts = time.time()
        except Exception:
            pass

        # Separate activation: If more than 1 person AND environment is field or forest → YOLO army objects
        env_top = [c.lower() for c in clip_bg.get("top_categories", [])]
        is_field_or_forest = any(k in env_top for k in ["field", "forest"])
        num_persons = yolo_result.get("object_counts", {}).get("person", 0)
        if num_persons > 1 and is_field_or_forest:
            army_det = yolo_army.analyze_image(color_path)
            clip_section["army"] = any(v > 0 for v in army_det.get("object_counts", {}).values())
            clip_section["army_objects"] = army_det.get("object_counts", {})

        # Prepare original paths and persist a single B&W copy under processed_photos
        from pathlib import Path as _P
        orig_full_path = orig_name if _P(orig_name).is_absolute() else str(input_path / orig_name)
        processed_copy_path = str(processed_dir / _P(orig_full_path).name)
        try:
            import shutil as _shutil
            _shutil.copy2(orig_full_path, processed_copy_path)
            # Propose deleting source to avoid double processing
            try:
                with open(files_to_delete_path, 'a', encoding='utf-8') as _ftd:
                    _ftd.write(f"{orig_full_path}\tReason: migrated to processed_photos to avoid reprocessing\n")
            except Exception:
                pass
        except Exception:
            processed_copy_path = orig_full_path

        # Remove the colorized image to avoid persisting it
        try:
            import os as _os
            if _os.path.isfile(color_path):
                _os.remove(color_path)
        except Exception:
            pass

        # Build CSV description string
        try:
            # Caption from LLM (adjusted printed caption is in summary only; use raw here)
            caption = None
            llm_obj = record.get("llm") if isinstance(record.get("llm"), dict) else None
            if isinstance(llm_obj, dict):
                llm_json_str = llm_obj.get("json")
                if isinstance(llm_json_str, str) and llm_json_str.strip():
                    try:
                        lj = json.loads(llm_json_str)
                        caption = lj.get('caption') if isinstance(lj.get('caption'), str) else None
                        # Flags and details
                        parts = []
                        if caption:
                            parts.append(caption.strip())
                        # Jewish symbols
                        if bool(lj.get('has_jewish_symbols', False)):
                            details = lj.get('jewish_symbols_details', [])
                            if isinstance(details, list) and details:
                                descs = []
                                for s in details[:5]:
                                    if isinstance(s, dict):
                                        st = s.get('symbol_type')
                                        sd = s.get('description')
                                        seg = []
                                        if isinstance(st, str) and st.strip():
                                            seg.append(st.strip())
                                        if isinstance(sd, str) and sd.strip():
                                            seg.append(sd.strip())
                                        if seg:
                                            descs.append(' - '.join(seg))
                                if descs:
                                    parts.append('; '.join(descs))
                        # Nazi symbols
                        if bool(lj.get('has_nazi_symbols', False)):
                            details = lj.get('nazi_symbols_details', [])
                            if isinstance(details, list) and details:
                                descs = []
                                for s in details[:5]:
                                    if isinstance(s, dict):
                                        st = s.get('symbol_type')
                                        sd = s.get('description')
                                        seg = []
                                        if isinstance(st, str) and st.strip():
                                            seg.append(st.strip())
                                        if isinstance(sd, str) and sd.strip():
                                            seg.append(sd.strip())
                                        if seg:
                                            descs.append(' - '.join(seg))
                                if descs:
                                    parts.append('; '.join(descs))
                        # Texts
                        ta = lj.get('text_analysis', {}) if isinstance(lj.get('text_analysis'), dict) else {}
                        if isinstance(ta, dict):
                            heb = ta.get('hebrew_text', {}) if isinstance(ta.get('hebrew_text'), dict) else {}
                            ger = ta.get('german_text', {}) if isinstance(ta.get('german_text'), dict) else {}
                            if isinstance(heb, dict) and bool(heb.get('present', False)):
                                txt = heb.get('text_found')
                                tr = heb.get('translation')
                                sub = []
                                if isinstance(txt, str) and txt.strip():
                                    sub.append(txt.strip())
                                if isinstance(tr, str) and tr.strip():
                                    sub.append(tr.strip())
                                if sub:
                                    parts.append(' - '.join(sub))
                            if isinstance(ger, dict) and bool(ger.get('present', False)):
                                txt = ger.get('text_found')
                                tr = ger.get('translation')
                                sub = []
                                if isinstance(txt, str) and txt.strip():
                                    sub.append(txt.strip())
                                if isinstance(tr, str) and tr.strip():
                                    sub.append(tr.strip())
                                if sub:
                                    parts.append(' - '.join(sub))
                        # Objects items + descriptions only
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
                    except Exception:
                        parts = [caption.strip()] if isinstance(caption, str) else []
                else:
                    parts = []
            else:
                parts = []

            # Gender counts
            clip_obj = record.get('clip') if isinstance(record.get('clip'), dict) else None
            genders = clip_obj.get('people_gender', []) if isinstance(clip_obj, dict) else []
            def _is_woman(g: Dict[str, Any]) -> bool:
                m = float(g.get('man', 0.0))
                w = float(g.get('woman', 0.0))
                return w >= m
            women = sum(1 for g in genders if isinstance(g, dict) and _is_woman(g))
            men = len(genders) - women
            parts.append(f"{men} men {women} women")

            # IO and background
            io = clip_obj.get('indoor_outdoor', 'unknown') if isinstance(clip_obj, dict) else 'unknown'
            if isinstance(io, str) and io:
                parts.append(io)
            bg = clip_obj.get('background_top', []) if isinstance(clip_obj, dict) else []
            bg1 = bg[0] if isinstance(bg, list) and bg else None
            if isinstance(bg1, str) and bg1:
                parts.append(bg1)

            description = '. '.join([p for p in parts if isinstance(p, str) and p])
        except Exception:
            description = ''

        # Row for TEXT csv (embedding-like)
        # Extract path words (folders + filename without extension)
        def _extract_path_words(path_str: str) -> str:
            import re as _re
            p = _P(path_str)
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

        path_words = _extract_path_words(processed_copy_path)
        desc_with_paths = description + (". " + path_words if path_words else "")

        rows_for_text.append({
            'original_path': processed_copy_path,
            'colorized_path': '',
            'full_results_path': str(full_rec_path),
            'llm_json_path': llm_obj.get('output_path') if isinstance(llm_obj, dict) else '',
            'description': desc_with_paths
        })

        # Row for FULL csv (no confidences, first background, compact lists)
        try:
            # YOLO object counts compact
            obj_counts = yolo_result.get('object_counts', {}) if isinstance(yolo_result, dict) else {}
            obj_counts_str = '; '.join(f"{k}:{v}" for k, v in sorted(obj_counts.items()))

            # Background first
            bg_first = None
            if isinstance(clip_obj, dict):
                bg_list = clip_obj.get('background_top', [])
                if isinstance(bg_list, list) and bg_list:
                    bg_first = bg_list[0]

            # LLM parsed fields
            llm_caption = ''
            ppl_under_18 = ''
            jewish_details = ''
            nazi_details = ''
            heb_present = ''
            heb_text = ''
            heb_tr = ''
            ger_present = ''
            ger_text = ''
            ger_tr = ''
            viol_present = ''
            viol_expl = ''
            llm_objs = ''
            if isinstance(llm_obj, dict):
                llm_json_str2 = llm_obj.get('json')
                if isinstance(llm_json_str2, str) and llm_json_str2.strip():
                    try:
                        lj2 = json.loads(llm_json_str2)
                        if isinstance(lj2.get('caption'), str):
                            llm_caption = lj2.get('caption')
                        if isinstance(lj2.get('people_under_18'), (int, float)):
                            ppl_under_18 = str(int(lj2.get('people_under_18')))
                        if bool(lj2.get('has_jewish_symbols', False)):
                            det = lj2.get('jewish_symbols_details', [])
                            if isinstance(det, list):
                                parts_j = []
                                for s in det[:5]:
                                    if isinstance(s, dict):
                                        st = s.get('symbol_type')
                                        sd = s.get('description')
                                        loc = s.get('location_in_image')
                                        seg = []
                                        if isinstance(st, str) and st.strip():
                                            seg.append(st.strip())
                                        if isinstance(sd, str) and sd.strip():
                                            seg.append(sd.strip())
                                        if isinstance(loc, str) and loc.strip():
                                            seg.append(loc.strip())
                                        if seg:
                                            parts_j.append(' - '.join(seg))
                                jewish_details = '; '.join(parts_j)
                        if bool(lj2.get('has_nazi_symbols', False)):
                            det = lj2.get('nazi_symbols_details', [])
                            if isinstance(det, list):
                                parts_n = []
                                for s in det[:5]:
                                    if isinstance(s, dict):
                                        st = s.get('symbol_type')
                                        sd = s.get('description')
                                        loc = s.get('location_in_image')
                                        seg = []
                                        if isinstance(st, str) and st.strip():
                                            seg.append(st.strip())
                                        if isinstance(sd, str) and sd.strip():
                                            seg.append(sd.strip())
                                        if isinstance(loc, str) and loc.strip():
                                            seg.append(loc.strip())
                                        if seg:
                                            parts_n.append(' - '.join(seg))
                                nazi_details = '; '.join(parts_n)
                        ta_val = lj2.get('text_analysis')
                        ta2 = ta_val if isinstance(ta_val, dict) else {}
                        if isinstance(ta2, dict):
                            ht = ta2.get('hebrew_text')
                            gt = ta2.get('german_text')
                            heb2 = ht if isinstance(ht, dict) else {}
                            ger2 = gt if isinstance(gt, dict) else {}
                            if isinstance(heb2, dict):
                                heb_present = 'true' if bool(heb2.get('present', False)) else 'false'
                                tf = heb2.get('text_found')
                                trn = heb2.get('translation')
                                heb_text = tf if isinstance(tf, str) else ''
                                heb_tr = trn if isinstance(trn, str) else ''
                            if isinstance(ger2, dict):
                                ger_present = 'true' if bool(ger2.get('present', False)) else 'false'
                                tf2 = ger2.get('text_found')
                                trn2 = ger2.get('translation')
                                ger_text = tf2 if isinstance(tf2, str) else ''
                                ger_tr = trn2 if isinstance(trn2, str) else ''
                        va_val = lj2.get('violence_assessment')
                        va2 = va_val if isinstance(va_val, dict) else {}
                        if isinstance(va2, dict):
                            viol_present = 'true' if bool(va2.get('signs_of_violence', False)) else 'false'
                            exp = va2.get('explanation')
                            viol_expl = exp if isinstance(exp, str) else ''
                        mo = lj2.get('main_objects_artifacts_animals', [])
                        if isinstance(mo, list):
                            parts_o = []
                            for o in mo[:5]:
                                if isinstance(o, dict):
                                    it2 = o.get('item')
                                    ds2 = o.get('description')
                                    if isinstance(it2, str) and isinstance(ds2, str) and it2.strip() and ds2.strip():
                                        parts_o.append(f"{it2.strip()}: {ds2.strip()}")
                            llm_objs = '; '.join(parts_o)
                    except Exception:
                        pass

            rows_for_full.append({
                'original_path': processed_copy_path,
                'colorized_path': '',
                'indoor_outdoor': io,
                'background': bg_first or '',
                'yolo_object_counts': obj_counts_str,
                'men': str(men),
                'women': str(women),
                'llm_caption': llm_caption,
                'llm_people_under_18': ppl_under_18,
                'llm_jewish_symbols': jewish_details,
                'llm_nazi_symbols': nazi_details,
                'hebrew_present': heb_present,
                'hebrew_text': heb_text,
                'hebrew_translation': heb_tr,
                'german_present': ger_present,
                'german_text': ger_text,
                'german_translation': ger_tr,
                'violence': viol_present,
                'violence_explanation': viol_expl,
                'llm_objects': llm_objs,
                'path_words': path_words,
                'full_results_path': str(full_rec_path),
                'llm_json_path': llm_obj.get('output_path') if isinstance(llm_obj, dict) else ''
            })
        except Exception:
            pass

    # 4) Write summaries (JSON + TXT)
    summary_writer = SummaryWriter()
    json_out = results_dir / "summary.json"
    txt_out = results_dir / "summary.txt"
    summary_writer.write_json(per_image, str(json_out))
    summary_writer.write_text(per_image, str(txt_out))

    # 5) Write CSV datasets (merge by original_path; overwrite existing rows)
    try:
        import csv
        # TEXT CSV
        csv_path_text = results_dir / 'data_text.csv'
        text_fields = ['original_path','colorized_path','full_results_path','llm_json_path','description']
        existing_text: Dict[str, Dict[str, Any]] = {}
        if csv_path_text.exists():
            with open(csv_path_text, 'r', newline='', encoding='utf-8') as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    k = (r.get('original_path') or '').strip()
                    if k:
                        existing_text[k] = {kk: (vv or '') for kk, vv in r.items()}
        for r in rows_for_text:
            k = (r.get('original_path') or '').strip()
            if not k:
                continue
            existing_text[k] = {**existing_text.get(k, {}), **r}
        with open(csv_path_text, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=text_fields)
            w.writeheader()
            for k in sorted(existing_text.keys()):
                row = existing_text[k]
                w.writerow({fn: row.get(fn, '') for fn in text_fields})

        # FULL CSV
        csv_path_full = results_dir / 'data_full.csv'
        full_fields = list(rows_for_full[0].keys()) if rows_for_full else []
        existing_full: Dict[str, Dict[str, Any]] = {}
        if csv_path_full.exists():
            with open(csv_path_full, 'r', newline='', encoding='utf-8') as f:
                rdr = csv.DictReader(f)
                # update fieldnames to superset if needed
                for r in rdr:
                    k = (r.get('original_path') or '').strip()
                    if k:
                        existing_full[k] = {kk: (vv or '') for kk, vv in r.items()}
                        for kk in r.keys():
                            if kk not in full_fields:
                                full_fields.append(kk)
        for r in rows_for_full:
            k = (r.get('original_path') or '').strip()
            if not k:
                continue
            existing_full[k] = {**existing_full.get(k, {}), **r}
        if full_fields:
            with open(csv_path_full, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=full_fields)
                w.writeheader()
                for k in sorted(existing_full.keys()):
                    row = existing_full[k]
                    w.writerow({fn: row.get(fn, '') for fn in full_fields})
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


def main() -> None:
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

    result = run_pipeline(
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


