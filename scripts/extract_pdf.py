"""
Extract text from each book's PDF, preserving page markers.

IMPORTANT: the "===== PAGE N =====" markers are load-bearing — 
semantic_chunker.py parses them to track page numbers and detect
chapter/section boundaries. Do not run clean_text.py on this output
before feeding it to semantic_chunker.py; clean_text.py strips these
markers, which will silently break page tracking and heading detection.

Expects PDFs in ../books/ (relative to this script, i.e. major/books/),
named per PDF_PATHS below. Writes output to ../extracted/<ID>_raw.txt,
matching what semantic_chunker.py reads.
"""
from pathlib import Path

import fitz  # PyMuPDF; the "fitz" import name is deprecated upstream but
              # still works — `import pymupdf` is the future-proof alternative

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = BASE_DIR / "extracted"

# book_id -> source PDF filename. These match your project's books/ folder
# (major/books/*.pdf) as laid out in your workspace.
PDF_PATHS = {
    "DS": BASE_DIR / "books" / "DS.pdf",
    "C": BASE_DIR / "books" / "Let us c - yashwantkanetkar.pdf",
    "CPP": BASE_DIR / "books" / "C++ The Complete Reference.pdf",
    "CLRS": BASE_DIR / "books" / "APS.pdf",   # note: written out as APA_raw.txt below, to match semantic_chunker.py
}

# Output filename per book_id — must match the "input" paths in
# semantic_chunker.py's BOOKS dict exactly.
OUTPUT_NAMES = {
    "DS": "DS_raw.txt",
    "C": "C_raw.txt",
    "CPP": "CPP_raw.txt",
    "CLRS": "APA_raw.txt",
}


def extract_one(book_id: str, pdf_path: Path, output_path: Path) -> None:
    if not pdf_path.exists():
        print(f"SKIP {book_id}: PDF not found at {pdf_path}")
        return

    doc = fitz.open(pdf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text()
            f.write(f"\n\n===== PAGE {page_number} =====\n\n")
            f.write(text)

    print(f"{book_id}: extracted {len(doc)} pages -> {output_path}")


def main():
    for book_id, pdf_path in PDF_PATHS.items():
        output_path = OUTPUT_DIR / OUTPUT_NAMES[book_id]
        extract_one(book_id, pdf_path, output_path)

    print("\nExtraction complete!")


if __name__ == "__main__":
    main()