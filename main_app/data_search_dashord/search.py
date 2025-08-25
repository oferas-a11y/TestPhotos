import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List


RESULTS_DIR = Path("main_app_outputs") / "results"
DATA_FULL = RESULTS_DIR / "data_full.csv"
DATA_TEXT = RESULTS_DIR / "data_text.csv"
OUT_DIR = Path("main_app_outputs") / "data_search_dashord"


def load_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def build_index() -> Dict[str, Dict[str, str]]:
    full_rows = load_csv(DATA_FULL)
    text_rows = load_csv(DATA_TEXT)
    # Index by original_path
    idx: Dict[str, Dict[str, str]] = {}
    for r in full_rows:
        idx[r.get("original_path", "")] = r
    for r in text_rows:
        k = r.get("original_path", "")
        if k in idx:
            idx[k]["text_description"] = r.get("description", "")
        else:
            idx[k] = {"text_description": r.get("description", "")}
    return idx


def compute_category_counts(idx: Dict[str, Dict[str, str]]) -> Dict[str, int]:
    counts = {
        "nazi_symbols": 0,
        "jewish_symbols": 0,
        "hebrew_text": 0,
        "german_text": 0,
        "violence": 0,
        "indoor": 0,
        "outdoor": 0,
    }
    for _, row in idx.items():
        if (row.get("llm_nazi_symbols") or "").strip():
            counts["nazi_symbols"] += 1
        if (row.get("llm_jewish_symbols") or "").strip():
            counts["jewish_symbols"] += 1
        if row.get("hebrew_present", "").lower() == "true":
            counts["hebrew_text"] += 1
        if row.get("german_present", "").lower() == "true":
            counts["german_text"] += 1
        if row.get("violence", "").lower() == "true":
            counts["violence"] += 1
        io = (row.get("indoor_outdoor") or "").lower()
        if io == "indoor":
            counts["indoor"] += 1
        elif io == "outdoor":
            counts["outdoor"] += 1
    return counts


def select_category() -> str:
    print("\nSearch by categories (type a number):")
    print("1) Nazi symbols present")
    print("2) Jewish symbols present")
    print("3) Hebrew text present")
    print("4) German text present")
    print("5) Violence present")
    print("6) Indoor")
    print("7) Outdoor")
    choice = input("Enter 1-7: ").strip()
    mapping = {
        "1": "nazi_symbols",
        "2": "jewish_symbols",
        "3": "hebrew_text",
        "4": "german_text",
        "5": "violence",
        "6": "indoor",
        "7": "outdoor",
    }
    return mapping.get(choice, "")


def filter_rows(idx: Dict[str, Dict[str, str]], cat: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _, r in idx.items():
        if cat == "nazi_symbols" and (r.get("llm_nazi_symbols") or "").strip():
            rows.append(r)
        elif cat == "jewish_symbols" and (r.get("llm_jewish_symbols") or "").strip():
            rows.append(r)
        elif cat == "hebrew_text" and r.get("hebrew_present", "").lower() == "true":
            rows.append(r)
        elif cat == "german_text" and r.get("german_present", "").lower() == "true":
            rows.append(r)
        elif cat == "violence" and r.get("violence", "").lower() == "true":
            rows.append(r)
        elif cat == "indoor" and (r.get("indoor_outdoor") or "").lower() == "indoor":
            rows.append(r)
        elif cat == "outdoor" and (r.get("indoor_outdoor") or "").lower() == "outdoor":
            rows.append(r)
    return rows


def write_report(cat: str, rows: List[Dict[str, str]]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"search_{cat}_{ts}.txt"
    lines: List[str] = []
    title = {
        "nazi_symbols": "Photos with Nazi symbols",
        "jewish_symbols": "Photos with Jewish symbols",
        "hebrew_text": "Photos with Hebrew text",
        "german_text": "Photos with German text",
        "violence": "Photos with signs of violence",
        "indoor": "Indoor photos",
        "outdoor": "Outdoor photos",
    }.get(cat, cat)
    lines.append(title)
    lines.append(f"Total: {len(rows)}")
    lines.append("")
    for r in rows:
        orig = r.get("original_path", "")
        desc = r.get("text_description", "")
        lines.append(f"Original: {orig}")
        if isinstance(desc, str) and desc.strip():
            lines.append(f"Description: {desc}")
        # Add compact details from full row
        io = r.get("indoor_outdoor", "")
        bg = r.get("background", "") or r.get("background_top", "")
        if io or bg:
            lines.append(f"Location: {io} {(' - ' + bg) if bg else ''}".strip())
        oc = r.get("yolo_object_counts", "")
        if oc:
            lines.append(f"Objects: {oc}")
        cap = r.get("llm_caption", "")
        if cap:
            lines.append(f"Caption: {cap}")
        if r.get("llm_jewish_symbols", ""):
            lines.append(f"Jewish: {r.get('llm_jewish_symbols')}")
        if r.get("llm_nazi_symbols", ""):
            lines.append(f"Nazi: {r.get('llm_nazi_symbols')}")
        # Texts
        if r.get("hebrew_present", "").lower() == "true":
            ht = r.get("hebrew_text", "")
            tr = r.get("hebrew_translation", "")
            show = " - ".join([s for s in [ht, tr] if s])
            if show:
                lines.append(f"Hebrew: {show}")
        if r.get("german_present", "").lower() == "true":
            gt = r.get("german_text", "")
            tr2 = r.get("german_translation", "")
            show2 = " - ".join([s for s in [gt, tr2] if s])
            if show2:
                lines.append(f"German: {show2}")
        lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return out_path


def main() -> None:
    if not DATA_FULL.exists():
        print(f"Missing {DATA_FULL}. Run the main app first.")
        return
    idx = build_index()
    counts = compute_category_counts(idx)
    print("\nAvailable in data:")
    print(f"- Nazi symbols: {counts['nazi_symbols']}")
    print(f"- Jewish symbols: {counts['jewish_symbols']}")
    print(f"- Hebrew text: {counts['hebrew_text']}")
    print(f"- German text: {counts['german_text']}")
    print(f"- Violence: {counts['violence']}")
    print(f"- Indoor: {counts['indoor']}")
    print(f"- Outdoor: {counts['outdoor']}")

    cat = select_category()
    if not cat:
        print("Invalid choice")
        return
    rows = filter_rows(idx, cat)
    out = write_report(cat, rows)
    print(f"\nWrote report: {out}")


if __name__ == "__main__":
    main()


