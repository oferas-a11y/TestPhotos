import json
from typing import List, Dict, Any


class SummaryWriter:
    def write_json(self, per_image: List[Dict[str, Any]], output_path: str) -> None:
        with open(output_path, 'w') as f:
            json.dump({"photos": per_image}, f, indent=2)

    def write_text(self, per_image: List[Dict[str, Any]], output_path: str) -> None:
        lines: List[str] = []
        for rec in per_image:
            lines.append(f"Image: {rec['original_filename']}")
            lines.append("Objects (YOLO):")
            oc = rec.get('yolo', {}).get('object_counts', {})
            for k, v in sorted(oc.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"  - {k}: {v}")
            lines.append("People (CLIP gender on YOLO person crops):")
            genders = rec.get('clip', {}).get('people_gender', [])
            # Tie-break to woman; otherwise choose higher score
            def is_woman(g):
                m = float(g.get('man', 0.0))
                w = float(g.get('woman', 0.0))
                return w >= m
            women = sum(1 for g in genders if is_woman(g))
            men = len(genders) - women
            lines.append(f"  man {men}, woman {women}")
            for g in genders:
                m = float(g.get('man', 0.0))
                w = float(g.get('woman', 0.0))
                label = 'woman' if w >= m else 'man'
                lines.append(f"  - person {g.get('person_index')}: {label} (man {m:.2f}, woman {w:.2f})")
            io = rec.get('clip', {}).get('indoor_outdoor', 'unknown')
            lines.append(f"Indoor/Outdoor: {io}")
            lines.append("Background (CLIP top-5):")
            top = rec.get('clip', {}).get('background_detections', [])[:5]
            for d in top:
                lines.append(f"  - {d.get('category')}: {float(d.get('confidence', 0.0)):.2f}")
            # Army (separate activation)
            if 'army' in rec.get('clip', {}):
                a = rec['clip']
                lines.append(f"Army (conditional): {'true' if a.get('army') else 'false'}")
                objs = a.get('army_objects', {})
                if objs:
                    lines.append("  Army objects:")
                    for k, v in objs.items():
                        lines.append(f"    - {k}: {v}")
            # OCR
            ocr = rec.get('ocr', {})
            has_text = bool(ocr.get('has_text'))
            lines.append(f"Text in photo: {'yes' if has_text else 'no'}")
            if has_text:
                for t in ocr.get('lines', [])[:5]:
                    lines.append(f"  " + t)
            # LLM output path (if any)
            llm = rec.get('llm', {})
            if llm.get('output_path'):
                lines.append(f"LLM JSON: {llm.get('output_path')}")
            # LLM parsed highlights: caption, true categories, objects (item + description)
            llm_json = llm.get('json') if isinstance(llm, dict) else None
            if isinstance(llm_json, str) and llm_json.strip():
                try:
                    import json as _json
                    lj = _json.loads(llm_json)
                    caption = lj.get('caption')
                    if isinstance(caption, str) and caption.strip():
                        # Interpret wartime qualifiers only if we have evidence
                        clip_sec = rec.get('clip', {}) if isinstance(rec.get('clip', {}), dict) else {}
                        army_true = bool(clip_sec.get('army', False))
                        army_counts = clip_sec.get('army_objects', {}) if isinstance(clip_sec.get('army_objects', {}), dict) else {}
                        army_any = bool(army_true or any(int(v) > 0 for v in army_counts.values()))
                        violence_true = False
                        va = lj.get('violence_assessment', {}) if isinstance(lj.get('violence_assessment'), dict) else {}
                        if isinstance(va, dict) and bool(va.get('signs_of_violence', False)):
                            violence_true = True
                        nazi_true = bool(lj.get('has_nazi_symbols', False))
                        wartime_evidence = bool(army_any or violence_true or nazi_true)
                        interpreted = caption.strip()
                        if not wartime_evidence:
                            # Remove wartime qualifiers conservatively
                            for pat in [
                                ' during wartime evacuation',
                                ' wartime evacuation',
                                ' during wartime',
                                ' wartime'
                            ]:
                                interpreted = interpreted.replace(pat, '')
                            # Normalize spaces
                            interpreted = ' '.join(interpreted.split())
                        lines.append(f"Caption: {interpreted}")
                    # Categories if true / >0
                    try:
                        ppl_u18 = int(lj.get('people_under_18', 0))
                        if ppl_u18 > 0:
                            lines.append(f"People under 18: {ppl_u18}")
                    except Exception:
                        pass
                    if bool(lj.get('has_jewish_symbols', False)):
                        lines.append("Jewish symbols: true")
                    if bool(lj.get('has_nazi_symbols', False)):
                        lines.append("Nazi symbols: true")
                    ta = lj.get('text_analysis', {}) if isinstance(lj.get('text_analysis'), dict) else {}
                    heb = ta.get('hebrew_text', {}) if isinstance(ta, dict) else {}
                    ger = ta.get('german_text', {}) if isinstance(ta, dict) else {}
                    if isinstance(heb, dict) and bool(heb.get('present', False)):
                        lines.append("Hebrew text: true")
                        txt = heb.get('text_found')
                        if isinstance(txt, str) and txt.strip():
                            lines.append(f"  Hebrew text found: {txt.strip()}")
                        tr = heb.get('translation')
                        if isinstance(tr, str) and tr.strip():
                            lines.append(f"  Hebrew translation: {tr.strip()}")
                    if isinstance(ger, dict) and bool(ger.get('present', False)):
                        lines.append("German text: true")
                        txt = ger.get('text_found')
                        if isinstance(txt, str) and txt.strip():
                            lines.append(f"  German text found: {txt.strip()}")
                        tr = ger.get('translation')
                        if isinstance(tr, str) and tr.strip():
                            lines.append(f"  German translation: {tr.strip()}")
                    va = lj.get('violence_assessment', {}) if isinstance(lj.get('violence_assessment'), dict) else {}
                    if isinstance(va, dict) and bool(va.get('signs_of_violence', False)):
                        lines.append("Violence: true")
                    # Symbols (list details)
                    if bool(lj.get('has_jewish_symbols', False)):
                        sym = lj.get('jewish_symbols_details', [])
                        if isinstance(sym, list) and sym:
                            lines.append("Jewish symbols details:")
                            for s in sym[:5]:
                                if not isinstance(s, dict):
                                    continue
                                st = s.get('symbol_type')
                                sd = s.get('description')
                                loc = s.get('location_in_image')
                                seg = []
                                if isinstance(st, str) and st.strip():
                                    seg.append(st.strip())
                                if isinstance(sd, str) and sd.strip():
                                    seg.append(sd.strip())
                                if isinstance(loc, str) and loc.strip():
                                    seg.append(f"({loc.strip()})")
                                if seg:
                                    lines.append("  - " + ": ".join(seg[:2]) + (" " + seg[2] if len(seg) > 2 else ""))
                    if bool(lj.get('has_nazi_symbols', False)):
                        sym = lj.get('nazi_symbols_details', [])
                        if isinstance(sym, list) and sym:
                            lines.append("Nazi symbols details:")
                            for s in sym[:5]:
                                if not isinstance(s, dict):
                                    continue
                                st = s.get('symbol_type')
                                sd = s.get('description')
                                loc = s.get('location_in_image')
                                seg = []
                                if isinstance(st, str) and st.strip():
                                    seg.append(st.strip())
                                if isinstance(sd, str) and sd.strip():
                                    seg.append(sd.strip())
                                if isinstance(loc, str) and loc.strip():
                                    seg.append(f"({loc.strip()})")
                                if seg:
                                    lines.append("  - " + ": ".join(seg[:2]) + (" " + seg[2] if len(seg) > 2 else ""))

                    # Objects list (item + description only)
                    objs = lj.get('main_objects_artifacts_animals', [])
                    if isinstance(objs, list) and objs:
                        lines.append("LLM objects:")
                        for o in objs:
                            if not isinstance(o, dict):
                                continue
                            item = o.get('item')
                            desc = o.get('description')
                            if isinstance(item, str) and isinstance(desc, str):
                                lines.append(f"  - {item}: {desc}")
                except Exception:
                    pass
            lines.append("")
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))


