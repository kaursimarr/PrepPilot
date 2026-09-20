"""
PrepPilot retriever: question -> top-k relevant chunks from the existing preppilot.db.

Reuses what is already in the project and changes nothing in it:
  * scripts/db.py          get_connection(), search_chunks()   (same calls main.py and the eval use)
  * all-MiniLM-L6-v2       same model, same normalisation as the embedding stage

Usage (from the project root):
    python retriever.py "What is a binary search tree?"
    python retriever.py "How does malloc work?" --book C -k 5
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
if not (BASE_DIR / "scripts").exists() and (BASE_DIR.parent / "scripts").exists():
    BASE_DIR = BASE_DIR.parent
for _p in (BASE_DIR, BASE_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DB_PATH = BASE_DIR / "preppilot.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

BOOK_TITLES = {
    "DS": "Handbook of Data Structures and Applications",
    "C": "Let Us C",
    "CPP": "C++: The Complete Reference",
    "CLRS": "Introduction to Algorithms (CLRS)",
}

# breadcrumbs look like: "Book > Chapter 12: Binary Search Trees > Section 12.1: What is a binary search tree?"
_CHAPTER_RE = re.compile(r"Chapter\s+(\d+)\s*(?::\s*([^>]*))?")
_SECTION_RE = re.compile(r"Section\s+([\d.]+)\s*(?::\s*([^>]*))?")


def _clean(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _pages(start: Any, end: Any) -> str:
    if start in (None, "", "?"):
        return ""
    if end in (None, "", start):
        return f"p. {start}"
    return f"pp. {start}-{end}"


def _normalise(rank: int, r: Dict[str, Any]) -> Dict[str, Any]:
    """Turn one search_chunks row into a stable dict. Every field is read with .get(), so a missing key never crashes."""
    breadcrumb = _clean(r.get("breadcrumb"))
    chapter, chapter_title = _clean(r.get("chapter")), _clean(r.get("chapter_title"))
    section, section_title = _clean(r.get("section")), _clean(r.get("section_title"))
    m = _CHAPTER_RE.search(breadcrumb)
    if m:
        chapter = chapter or m.group(1)
        chapter_title = chapter_title or _clean(m.group(2))
    m = _SECTION_RE.search(breadcrumb)
    if m:
        section = section or m.group(1)
        section_title = section_title or _clean(m.group(2))

    sim = r.get("similarity")
    if sim is None:
        sim = 1.0 - float(r.get("distance", 1.0))
    book_id = _clean(r.get("book_id") or r.get("book"))
    return {
        "rank": rank,
        "chunk_id": _clean(r.get("chunk_id") or r.get("id")),
        "text": _clean(r.get("text")),
        "book_id": book_id,
        "book": BOOK_TITLES.get(book_id, book_id),
        "chapter": chapter,
        "chapter_title": chapter_title,
        "section": section,
        "section_title": section_title,
        "page_start": r.get("page_start"),
        "page_end": r.get("page_end"),
        "pages": _pages(r.get("page_start"), r.get("page_end")),
        "similarity": float(sim),
        "breadcrumb": breadcrumb,
        "image_paths": r.get("image_paths"),
    }


def location_label(c: Dict[str, Any]) -> str:
    """'Introduction to Algorithms (CLRS) > Chapter 12: Binary Search Trees > Section 12.1: ... > p. 308'"""
    parts = [c["book"]]
    if c["chapter"]:
        parts.append(f"Chapter {c['chapter']}" + (f": {c['chapter_title']}" if c["chapter_title"] else ""))
    if c["section"] or c["section_title"]:
        parts.append(f"Section {c['section']}".strip() + (f": {c['section_title']}" if c["section_title"] else ""))
    if c["pages"]:
        parts.append(c["pages"])
    return " > ".join(parts)


class Retriever:
    """Loads the embedding model and the database connection once, then answers many questions."""

    def __init__(self, db_path: Path = DB_PATH, model_name: str = MODEL_NAME):
        self.db_path = Path(db_path)
        self.model_name = model_name
        self._model = None
        self._conn = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _get_conn(self):
        if self._conn is None:
            if not self.db_path.exists():
                raise FileNotFoundError(f"{self.db_path} not found. Run `python main.py` (Stage 4) first.")
            from scripts.db import get_connection
            self._conn = get_connection(self.db_path)
        return self._conn

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K, book_id: Optional[str] = None) -> List[Dict[str, Any]]:
        question = question.strip()
        if not question:
            return []
        from scripts.db import search_chunks
        vec = self._get_model().encode(question, normalize_embeddings=True)
        rows = search_chunks(self._get_conn(), query_vector=vec, top_k=top_k, book_id=book_id)
        return [_normalise(i, r) for i, r in enumerate(rows, 1)]

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None


def print_chunks(chunks: List[Dict[str, Any]], full: bool = False, width: int = 350) -> None:
    for c in chunks:
        print(f"[{c['rank']}] similarity {c['similarity']:.3f} | {location_label(c)}")
        text = c["text"].replace("\n", " ")
        print("    " + (text if full or len(text) <= width else text[:width].rstrip() + " ..."))
        print()


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description="Retrieve the top chunks for a question from preppilot.db")
    ap.add_argument("question")
    ap.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--book", choices=sorted(BOOK_TITLES), default=None, help="limit to one book")
    ap.add_argument("--full", action="store_true", help="print the full chunk text")
    args = ap.parse_args()
    r = Retriever()
    try:
        chunks = r.retrieve(args.question, top_k=args.top_k, book_id=args.book)
    finally:
        r.close()
    print_chunks(chunks, full=args.full)


if __name__ == "__main__":
    main()