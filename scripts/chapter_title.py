"""
Read each book's PDF outline (bookmarks) and write chapter titles to chapter_titles.json.
fix_chunks.py uses that file to fill the chapter names in the breadcrumbs
("Chapter 33: Data Structures for Sets" instead of just "Chapter 33").

Run once from C:\\major:   python make_chapter_titles.py
Then LOOK at the printed table - if a title looks wrong, put the right one in
CHAPTER_TITLE_OVERRIDES in fix_chunks.py (overrides win over this file).

If a PDF has no outline, the script says so and writes nothing for that book.
"""
import json
import re
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz

HERE = Path(__file__).resolve().parent
BASE = HERE if (HERE / "books").exists() else HERE.parent

PDF_NAMES = {
    "DS": "DS.pdf",
    "C": "Let us c - yashwantkanetkar.pdf",
    "CPP": "C++ The Complete Reference.pdf",
    "CLRS": "APS.pdf",
}


def _find_pdf(name):
    """PDFs may sit in books/, scripts/ or the project root."""
    for folder in (BASE / "books", BASE / "scripts", BASE, HERE):
        if (folder / name).exists():
            return folder / name
    return BASE / "books" / name          # not found; main() reports it as missing


PDFS = {book: _find_pdf(name) for book, name in PDF_NAMES.items()}
OUT = BASE / "chapter_titles.json"

# "12 Title", "12. Title", "Chapter 12: Title", "Chapter 12 Title"
CHAPTER = re.compile(r"^\s*(?:chapter\s+)?(\d{1,3})\s*[.:\-\u2013\u2014]?\s+(\S.*?)\s*$", re.I)


def chapters_from_outline(pdf_path):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc(simple=True)                    # [[level, title, page], ...]
    if not toc:
        return None, 0
    found = {}                                        # chapter number -> (level, title, page)
    for level, title, page in toc:
        m = CHAPTER.match(title)
        if not m:
            continue
        num, name = m.group(1), re.sub(r"\s+", " ", m.group(2)).strip(" .")
        if len(name) < 3 or re.fullmatch(r"[\d.\s-]+", name):
            continue
        old = found.get(num)
        if old is None or level < old[0]:             # keep the shallowest entry for that number
            found[num] = (level, name, page)
    return found, len(toc)


def main():
    result = {}
    if OUT.exists():
        try:
            result = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:                             # noqa: BLE001
            result = {}
    for book, pdf in PDFS.items():
        print(f"\n{book}: {pdf.name}")
        if not pdf.exists():
            print("  PDF not found - skipped")
            continue
        found, n_entries = chapters_from_outline(pdf)
        if not found:
            print(f"  no usable outline ({n_entries} bookmark entries) - add titles by hand in fix_chunks.py")
            continue
        nums = sorted(found, key=int)
        result[book] = {n: found[n][1] for n in nums}
        print(f"  {len(nums)} chapters from {n_entries} bookmarks (chapter numbers {nums[0]}..{nums[-1]})")
        for n in nums[:6] + (["..."] if len(nums) > 12 else []) + nums[-3:]:
            if n == "...":
                print("    ...")
            else:
                lvl, name, page = found[n]
                print(f"    {n:>3}  {name}   (PDF page {page})")
        missing = [str(i) for i in range(int(nums[0]), int(nums[-1]) + 1) if str(i) not in found]
        if missing:
            print(f"  numbers missing between first and last: {', '.join(missing[:20])}")
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()