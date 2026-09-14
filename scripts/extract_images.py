"""
Textbook Image and Diagram Extraction Pipeline for PrepPilot.

Extracts both:
1. Embedded raster images (JPEG, PNG) using doc.extract_image(xref), filtering spacer/icon noise.
2. Rendered vector diagram figures (trees, cards, flowcharts, graphs) via PyMuPDF bounding-box crops.

Outputs images to `extracted_images/<book_id>/` and registers metadata in both:
- `extracted_images/images_metadata.jsonl`
- `preppilot.db` (extracted_images table)
"""

from pathlib import Path
import argparse
import json
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = BASE_DIR / "extracted_images"
DB_PATH = BASE_DIR / "preppilot.db"

# Mapping of book_id to possible filenames in either books/ or scripts/
BOOK_FILES = {
    "DS": ["DS.pdf", "books/DS.pdf"],
    "C": ["Let us c - yashwantkanetkar.pdf", "books/Let us c - yashwantkanetkar.pdf"],
    "CPP": ["C++ The Complete Reference.pdf", "books/C++ The Complete Reference.pdf"],
    "CLRS": ["APS.pdf", "books/APS.pdf"],
}

FIG_CAPTION_RE = re.compile(
    r"(?:FIGURE|Figure|FIG)\s+(\d+(?:\.\d+)*)\s*[:.\-—]?\s*([^\n\r]*)",
    re.I
)


def find_pdf_path(book_id: str) -> Optional[Path]:
    """Find the PDF location in either books/ or scripts/."""
    candidates = BOOK_FILES.get(book_id, [])
    for cand in candidates:
        p1 = BASE_DIR / cand
        if p1.exists():
            return p1
        p2 = SCRIPT_DIR / cand
        if p2.exists():
            return p2
        p3 = BASE_DIR / "scripts" / cand
        if p3.exists():
            return p3
    return None


def extract_raster_images(
    doc: fitz.Document,
    page: fitz.Page,
    page_num: int,
    book_id: str,
    book_dir: Path,
    min_dim: int = 64
) -> List[Dict[str, Any]]:
    """Extract embedded raster image XObjects, filtering out tiny icons and spacers."""
    extracted = []
    seen_xrefs = set()

    for img_info in page.get_images():
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        try:
            base_img = doc.extract_image(xref)
        except Exception:
            continue

        w, h = base_img.get("width", 0), base_img.get("height", 0)
        ext = base_img.get("ext", "png")

        # Skip tiny icons, 1x1 spacer artifacts, or thin separator rules
        if w < min_dim or h < min_dim:
            continue
        if (w / max(h, 1) > 15) or (h / max(w, 1) > 15):
            continue

        img_id = f"{book_id}_p{page_num}_img{xref}"
        filename = f"{img_id}.{ext}"
        filepath = book_dir / filename

        with open(filepath, "wb") as f:
            f.write(base_img["image"])

        rel_path = f"extracted_images/{book_id}/{filename}"
        extracted.append({
            "image_id": img_id,
            "book_id": book_id,
            "page_number": page_num,
            "caption": f"Embedded image on page {page_num}",
            "image_type": "raster_xobject",
            "file_path": rel_path,
            "width": w,
            "height": h,
        })

    return extracted


def extract_vector_figures(
    page: fitz.Page,
    page_num: int,
    book_id: str,
    book_dir: Path,
    dpi: int = 150
) -> List[Dict[str, Any]]:
    """Detect figure captions on pages with vector drawings and crop the diagram."""
    extracted = []
    page_text = page.get_text()
    matches = list(FIG_CAPTION_RE.finditer(page_text))
    if not matches:
        return extracted

    drawings = page.get_drawings()
    if not drawings:
        return extracted

    # Group drawing rects
    drawing_rects = [d["rect"] for d in drawings]
    if not drawing_rects:
        return extracted

    for match in matches:
        fig_num = match.group(1)
        full_caption_text = match.group(0).strip().replace("\n", " ")
        # Sanitize caption for display
        clean_caption = full_caption_text[:140]

        # Search for caption text position on page
        search_term = f"Figure {fig_num}"
        cap_hits = page.search_for(search_term) or page.search_for(f"FIGURE {fig_num}") or page.search_for(f"FIG {fig_num}")
        if not cap_hits:
            continue

        # In case term appears multiple times in prose, find the one closest to a cluster of drawings
        best_cap_rect = None
        best_drawings_cluster = []
        min_distance = float("inf")

        for cap_rect in cap_hits:
            # Drawings typically sit immediately above the caption (within 350 points)
            nearby_drawings = [
                r for r in drawing_rects
                if (cap_rect.y0 - 380 <= r.y1 <= cap_rect.y1 + 10)
                and (abs(r.x0 - cap_rect.x0) < 300)
            ]
            if nearby_drawings:
                dist = abs(cap_rect.y0 - max(r.y1 for r in nearby_drawings))
                if dist < min_distance:
                    min_distance = dist
                    best_cap_rect = cap_rect
                    best_drawings_cluster = nearby_drawings

        if not best_drawings_cluster or not best_cap_rect:
            continue

        # Compute union bounding box
        union_box = fitz.Rect(best_drawings_cluster[0])
        for r in best_drawings_cluster[1:]:
            union_box |= r

        # Include the caption rectangle in the crop
        crop_box = union_box | best_cap_rect

        # Add comfortable margin (6 points)
        crop_box.x0 = max(0, crop_box.x0 - 8)
        crop_box.y0 = max(0, crop_box.y0 - 8)
        crop_box.x1 = min(page.rect.width, crop_box.x1 + 8)
        crop_box.y1 = min(page.rect.height, crop_box.y1 + 8)

        # Skip invalid or tiny boxes
        if crop_box.width < 50 or crop_box.height < 40:
            continue

        fig_id = f"{book_id}_p{page_num}_fig_{fig_num.replace('.', '_')}"
        filename = f"{fig_id}.png"
        filepath = book_dir / filename

        # Render clip
        pix = page.get_pixmap(clip=crop_box, dpi=dpi)
        pix.save(str(filepath))

        rel_path = f"extracted_images/{book_id}/{filename}"
        extracted.append({
            "image_id": fig_id,
            "book_id": book_id,
            "page_number": page_num,
            "caption": clean_caption,
            "image_type": "vector_figure",
            "file_path": rel_path,
            "width": pix.width,
            "height": pix.height,
        })

    return extracted


def save_metadata_to_db(metadata_records: List[Dict[str, Any]]) -> None:
    """Save metadata to preppilot.db if available."""
    if not DB_PATH.exists():
        return

    try:
        conn = sqlite3.connect(str(DB_PATH))
        with conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS extracted_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT UNIQUE NOT NULL,
                book_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                caption TEXT,
                image_type TEXT,
                file_path TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            for rec in metadata_records:
                conn.execute("""
                    INSERT OR REPLACE INTO extracted_images (
                        image_id, book_id, page_number, caption, image_type, file_path, width, height
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rec["image_id"], rec["book_id"], rec["page_number"],
                    rec["caption"], rec["image_type"], rec["file_path"],
                    rec["width"], rec["height"]
                ))
        conn.close()
    except Exception as e:
        print(f"Warning: Could not save image metadata to DB: {e}")


def process_book(
    book_id: str,
    max_pages: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Extract raster and vector images for a single textbook."""
    pdf_path = find_pdf_path(book_id)
    if not pdf_path:
        print(f"SKIP {book_id}: PDF file not found.")
        return []

    book_dir = OUTPUT_DIR / book_id
    book_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {book_id} from {pdf_path.name}...")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    limit = min(total_pages, max_pages) if max_pages else total_pages

    all_extracted = []
    raster_count = 0
    vector_count = 0

    for page_num in range(1, limit + 1):
        page = doc[page_num - 1]

        # 1. Raster images
        rasters = extract_raster_images(doc, page, page_num, book_id, book_dir)
        if rasters:
            all_extracted.extend(rasters)
            raster_count += len(rasters)

        # 2. Vector diagram figures
        vectors = extract_vector_figures(page, page_num, book_id, book_dir)
        if vectors:
            all_extracted.extend(vectors)
            vector_count += len(vectors)

    print(f"  {book_id} Complete: {raster_count} raster images, {vector_count} vector diagrams extracted (pages 1-{limit}).")
    return all_extracted


def main():
    parser = argparse.ArgumentParser(description="Extract textbook figures and diagrams.")
    parser.add_argument("--book", choices=["DS", "C", "CPP", "CLRS", "ALL"], default="ALL",
                        help="Specific book to extract, or ALL")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Maximum pages to process per book (useful for testing)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    books = ["CLRS", "DS", "CPP", "C"] if args.book == "ALL" else [args.book]

    all_records = []
    for b in books:
        records = process_book(b, max_pages=args.max_pages)
        all_records.extend(records)

    # Save metadata to JSONL
    meta_path = OUTPUT_DIR / "images_metadata.jsonl"
    with open(meta_path, "a", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save to SQLite database
    save_metadata_to_db(all_records)

    print(f"\n=======================================================")
    print(f"Extraction Summary:")
    print(f"Total images saved: {len(all_records)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metadata file: {meta_path}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
