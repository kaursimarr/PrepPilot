"""
Pipeline script to populate the unified SQLite + sqlite-vec database (preppilot.db).

Workflow:
1. Initializes relational tables and vec_chunks virtual table via db.init_db().
2. Indexes extracted images by (book_id, page_number) to link diagrams to chunks.
3. Loads chunks from `chunks_semantic/*.jsonl` (or generates them via semantic_chunker).
4. Generates 384-dimensional dense embeddings using sentence-transformers/all-MiniLM-L6-v2.
5. Batch-inserts chunks + vectors into preppilot.db.
6. Runs a verification similarity search across all indexed textbooks.
"""

from pathlib import Path
import argparse
import json
import sqlite3
import sys
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
CHUNKS_DIR = BASE_DIR / "chunks_semantic"
DB_PATH = BASE_DIR / "preppilot.db"

# Ensure workspace is on sys.path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from scripts.db import (
    get_connection,
    init_db,
    insert_chunks_batch,
    search_chunks,
    EMBEDDING_DIM
)

CHUNK_FILES = {
    "DS": CHUNKS_DIR / "DS_chunks.jsonl",
    "C": CHUNKS_DIR / "C_chunks.jsonl",
    "CPP": CHUNKS_DIR / "CPP_chunks.jsonl",
    "CLRS": CHUNKS_DIR / "CLRS_chunks.jsonl",
}

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_image_lookup(conn: sqlite3.Connection) -> Dict[Tuple[str, int], List[str]]:
    """Build a lookup mapping (book_id, page_number) -> list of image file paths."""
    lookup: Dict[Tuple[str, int], List[str]] = {}
    try:
        rows = conn.execute("""
            SELECT book_id, page_number, file_path
            FROM extracted_images
            ORDER BY page_number ASC
        """).fetchall()
        for r in rows:
            key = (r["book_id"].upper(), int(r["page_number"]))
            lookup.setdefault(key, []).append(r["file_path"])
    except sqlite3.OperationalError:
        pass
    return lookup


def load_chunks_for_book(book_id: str, chunk_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL chunks from disk."""
    if not chunk_path.exists():
        print(f"Warning: {chunk_path.name} not found.")
        return []

    records = []
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def attach_images_to_chunks(
    chunks: List[Dict[str, Any]],
    image_lookup: Dict[Tuple[str, int], List[str]]
) -> None:
    """Link extracted diagrams to each chunk based on its page range."""
    for c in chunks:
        book_id = c.get("book_id", "").upper()
        p_start = c.get("page_start", 0)
        p_end = c.get("page_end", p_start)

        matched_images = []
        for p in range(p_start, p_end + 1):
            imgs = image_lookup.get((book_id, p), [])
            for img in imgs:
                if img not in matched_images:
                    matched_images.append(img)

        c["image_paths"] = matched_images
        if matched_images:
            c["contains_figure"] = True


def migrate(book_filter: Optional[str] = None, max_chunks_per_book: Optional[int] = None) -> None:
    print("=" * 76)
    print("PREPPILOT - UNIFIED SQLITE + SQLITE-VEC MIGRATION PIPELINE")
    print("=" * 76)

    # 1. Initialize DB
    conn = get_connection(DB_PATH)
    init_db(conn)

    # 2. Image Lookup
    image_lookup = load_image_lookup(conn)
    print(f"Loaded image mappings for {len(image_lookup)} pages from extracted_images table.")

    # 3. Load Sentence Transformers
    print(f"\nLoading embedding model: {MODEL_NAME} ...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
    except ImportError:
        print("ERROR: sentence-transformers is still installing. Please retry once installation completes.")
        return

    # 4. Process Books
    books_to_process = [book_filter] if book_filter else ["CLRS", "DS", "CPP", "C"]
    total_indexed = 0

    for book_id in books_to_process:
        chunk_file = CHUNK_FILES.get(book_id)
        if not chunk_file or not chunk_file.exists():
            print(f"\nSkipping {book_id}: {chunk_file} does not exist. (Run semantic_chunker.py to create chunks)")
            continue

        print(f"\nProcessing {book_id} chunks from {chunk_file.name}...")
        raw_chunks = load_chunks_for_book(book_id, chunk_file)
        if max_chunks_per_book:
            raw_chunks = raw_chunks[:max_chunks_per_book]

        if not raw_chunks:
            continue

        attach_images_to_chunks(raw_chunks, image_lookup)

        # Prepare contextual text with breadcrumbs for high-quality embedding
        texts_to_embed = []
        for c in raw_chunks:
            if not c.get("breadcrumb"):
                bc_parts = [c.get("book_title") or c.get("book_id", "")]
                if c.get("chapter"):
                    bc_parts.append(f"Chapter {c['chapter']}: {c.get('chapter_title','')}".strip(": "))
                if c.get("section"):
                    bc_parts.append(f"Section {c['section']}: {c.get('section_title','')}".strip(": "))
                c["breadcrumb"] = " > ".join(bc_parts)

            search_text = c.get("search_text") or f"[{c['breadcrumb']}]\n\n{c['text']}"
            texts_to_embed.append(search_text)

        print(f"  Encoding {len(texts_to_embed)} contextualized chunks into {EMBEDDING_DIM}-dim vectors (batch_size={BATCH_SIZE})...")

        embeddings = model.encode(
            texts_to_embed,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        chunks_with_embeddings = list(zip(raw_chunks, embeddings))
        print(f"  Inserting {len(chunks_with_embeddings)} chunks into SQLite (chunks + vec_chunks)...")
        inserted = insert_chunks_batch(conn, chunks_with_embeddings)
        total_indexed += inserted
        print(f"  Successfully indexed {inserted} chunks for {book_id}.")

    print("\n" + "=" * 76)
    print(f"MIGRATION COMPLETE: {total_indexed} total chunks indexed into {DB_PATH.name}.")
    print("=" * 76)

    # 5. Quick Verification Search
    if total_indexed > 0:
        test_query = "How does insertion sort compare elements and arrange cards?"
        print(f"\nRunning Verification Search: {test_query!r}")
        q_vec = model.encode([test_query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
        results = search_chunks(conn, q_vec, top_k=2)

        print("\nTop Retrieved Chunks:")
        for idx, r in enumerate(results, 1):
            print(f"  #{idx} [{r['chunk_id']}] {r['book_id']} - Page {r['page_start']}-{r['page_end']}")
            print(f"      Section: {r.get('section', '')} {r.get('section_title', '')}")
            print(f"      Similarity: {r['similarity']:.4f} (Distance: {r['distance']:.4f})")
            if r.get('image_paths'):
                print(f"      Linked Diagrams: {r['image_paths']}")
            print(f"      Preview: {r['text'][:200].replace(chr(10), ' ')}...\n")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Populate unified SQLite + sqlite-vec database.")
    parser.add_argument("--book", choices=["DS", "C", "CPP", "CLRS"], default=None,
                        help="Index a specific book only")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of chunks to index per book (for testing)")
    args = parser.parse_args()

    migrate(book_filter=args.book, max_chunks_per_book=args.limit)


if __name__ == "__main__":
    main()
