#!/usr/bin/env python3
"""
=============================================================================
PrepPilot - Master Pipeline Orchestrator
=============================================================================
A unified, single-file entry point that compiles, coordinates, and executes
all stages of the PrepPilot AI-learning & RAG pipeline:

  Stage 1: PDF Text Extraction       (raw text + page markers)
  Stage 2: Diagram & Figure Cropping (raster XObjects + 150 DPI vector diagrams)
  Stage 3: Contextual Chunking       (breadcrumbs, de-hyphenation, atomic blocks)
  Stage 4: Unified Database Embed    (MiniLM-L6-v2, SQLite + sqlite-vec)
  Stage 5: AWS RDS Cloud Sync        (PostgreSQL + pgvector, optional --rds)
  Stage 6: Verification / Semantic Search (--query "...")

Usage:
  python main.py                    # Run complete pipeline end-to-end
  python main.py --status           # Display current artifact and database status
  python main.py --query "query"    # Test semantic vector search on indexed chunks
  python main.py --extract-text     # Run Stage 1 only
  python main.py --extract-images   # Run Stage 2 only
  python main.py --chunk            # Run Stage 3 only
  python main.py --embed            # Run Stage 4 only
  python main.py --rds              # Sync to AWS RDS PostgreSQL
  python main.py --compile-check    # Compile all repo Python files into bytecode
=============================================================================
"""

from pathlib import Path
import argparse
import compileall
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root and scripts are in sys.path
BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Standard directories
EXTRACTED_DIR = BASE_DIR / "extracted"
CHUNKS_DIR = BASE_DIR / "chunks_semantic"
IMAGES_DIR = BASE_DIR / "extracted_images"
DB_PATH = BASE_DIR / "preppilot.db"

# Textbook configuration & filename resolution
BOOKS_CONFIG = {
    "DS": {
        "title": "Handbook of Data Structures and Applications",
        "filenames": ["DS.pdf", "books/DS.pdf", "scripts/DS.pdf"],
        "raw_name": "DS_raw.txt",
        "chunk_name": "DS_chunks.jsonl",
    },
    "C": {
        "title": "Let Us C",
        "filenames": [
            "Let us c - yashwantkanetkar.pdf",
            "books/Let us c - yashwantkanetkar.pdf",
            "scripts/Let us c - yashwantkanetkar.pdf",
        ],
        "raw_name": "C_raw.txt",
        "chunk_name": "C_chunks.jsonl",
    },
    "CPP": {
        "title": "C++: The Complete Reference",
        "filenames": [
            "C++ The Complete Reference.pdf",
            "books/C++ The Complete Reference.pdf",
            "scripts/C++ The Complete Reference.pdf",
        ],
        "raw_name": "CPP_raw.txt",
        "chunk_name": "CPP_chunks.jsonl",
    },
    "CLRS": {
        "title": "Introduction to Algorithms (CLRS)",
        "filenames": ["APS.pdf", "books/APS.pdf", "scripts/APS.pdf"],
        "raw_name": "APA_raw.txt",
        "chunk_name": "CLRS_chunks.jsonl",
    },
}


def find_pdf(book_id: str) -> Optional[Path]:
    """Locate the PDF file for a book across possible directories."""
    cfg = BOOKS_CONFIG.get(book_id)
    if not cfg:
        return None
    for candidate in cfg["filenames"]:
        p = BASE_DIR / candidate
        if p.exists():
            return p
    return None


def print_banner(title: str):
    """Print an eye-catching section banner."""
    width = 70
    print("\n" + "=" * width)
    print(f" {title.center(width - 2)} ")
    print("=" * width)


# =============================================================================
# STAGE 1: PDF TEXT EXTRACTION
# =============================================================================
def run_stage_1_pdf_extraction(book_filter: Optional[str] = None) -> bool:
    """Extract raw text with page boundary markers from PDFs."""
    print_banner("STAGE 1: PDF TEXT EXTRACTION")
    try:
        import fitz
    except ImportError:
        print("ERROR: PyMuPDF (fitz) is not installed. Run: pip install PyMuPDF")
        return False

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    books = [book_filter] if book_filter else list(BOOKS_CONFIG.keys())

    extracted_count = 0
    for book_id in books:
        if book_id not in BOOKS_CONFIG:
            print(f"Warning: Unknown book ID: {book_id}")
            continue

        pdf_path = find_pdf(book_id)
        if not pdf_path:
            print(f"  [!] {book_id}: PDF not found. Searched candidate paths.")
            continue

        out_file = EXTRACTED_DIR / BOOKS_CONFIG[book_id]["raw_name"]
        print(f"  [>] Processing {book_id}: {pdf_path.name} ...")
        t0 = time.time()

        doc = fitz.open(pdf_path)
        page_count = len(doc)
        with out_file.open("w", encoding="utf-8") as f:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                f.write(f"\n\n===== PAGE {page_num} =====\n\n")
                f.write(text)

        elapsed = time.time() - t0
        file_size_mb = out_file.stat().st_size / (1024 * 1024)
        print(f"      OK -> {out_file.name} ({page_count} pages, {file_size_mb:.2f} MB in {elapsed:.1f}s)")
        extracted_count += 1

    print(f"\nStage 1 Complete: {extracted_count}/{len(books)} books extracted.")
    return extracted_count > 0


# =============================================================================
# STAGE 2: DIAGRAM & FIGURE EXTRACTION
# =============================================================================
def run_stage_2_image_extraction(book_filter: Optional[str] = None, max_pages: Optional[int] = None) -> bool:
    """Extract embedded raster images and crop vector diagram figures."""
    print_banner("STAGE 2: MULTIMODAL DIAGRAM & FIGURE EXTRACTION")
    try:
        from scripts import extract_images
    except ImportError:
        print("ERROR: Could not import scripts.extract_images.")
        return False

    books = [book_filter] if book_filter else list(BOOKS_CONFIG.keys())
    total_extracted = 0

    for book_id in books:
        print(f"  [>] Scanning {book_id} for diagrams and captioned figures...")
        t0 = time.time()
        extracted = extract_images.process_book(
            book_id=book_id,
            max_pages=max_pages,
            db_path=DB_PATH,
            extract_rasters=True,
            crop_figures=True,
        )
        elapsed = time.time() - t0
        total_extracted += len(extracted)
        print(f"      Extracted {len(extracted)} visual assets in {elapsed:.1f}s")

    print(f"\nStage 2 Complete: Total {total_extracted} diagrams/figures registered in DB.")
    return True


# =============================================================================
# STAGE 3: CONTEXTUAL SEMANTIC CHUNKING
# =============================================================================
def run_stage_3_chunking(book_filter: Optional[str] = None) -> bool:
    """Chunk extracted text into contextual semantic chunks with breadcrumbs."""
    print_banner("STAGE 3: CONTEXTUAL SEMANTIC CHUNKING")
    try:
        from scripts import semantic_chunker
    except ImportError:
        print("ERROR: Could not import scripts.semantic_chunker.")
        return False

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    books = [book_filter] if book_filter else ["DS", "C", "CPP", "CLRS"]

    chunked_count = 0
    for book_id in books:
        if book_id not in semantic_chunker.BOOKS:
            continue

        cfg = semantic_chunker.BOOKS[book_id]
        if not cfg["input"].exists():
            print(f"  [!] Missing raw text for {book_id} at {cfg['input']}. Run Stage 1 first.")
            continue

        print(f"  [>] Chunking {book_id} ('{cfg['title']}') ...")
        t0 = time.time()
        try:
            semantic_chunker.process(book_id, cfg)
            elapsed = time.time() - t0
            chunked_count += 1
            print(f"      Completed in {elapsed:.1f}s")
        except Exception as e:
            print(f"      FAILED: {e}")

    print(f"\nStage 3 Complete: {chunked_count}/{len(books)} books chunked into {CHUNKS_DIR}.")
    return chunked_count > 0


# =============================================================================
# STAGE 4: UNIFIED DATABASE & VECTOR EMBEDDING (SQLite + sqlite-vec)
# =============================================================================
def run_stage_4_database_embedding(book_filter: Optional[str] = None, batch_size: int = 32) -> bool:
    """Embed chunks using MiniLM and store in unified preppilot.db."""
    print_banner("STAGE 4: UNIFIED VECTOR DATABASE INGESTION")
    try:
        from scripts import migrate_to_sqlite
    except ImportError:
        print("ERROR: Could not import scripts.migrate_to_sqlite.")
        return False

    print(f"  Target SQLite DB: {DB_PATH}")
    t0 = time.time()
    try:
        migrate_to_sqlite.run_pipeline(
            db_path=DB_PATH,
            book_filter=book_filter,
            batch_size=batch_size,
            auto_chunk=False,
        )
        elapsed = time.time() - t0
        print(f"\nStage 4 Complete: Database indexed in {elapsed:.1f}s.")
        return True
    except Exception as e:
        print(f"Stage 4 Error: {e}")
        return False


# =============================================================================
# STAGE 5: AWS RDS POSTGRESQL SYNC (Optional)
# =============================================================================
def run_stage_5_rds_sync() -> bool:
    """Migrate local SQLite data and vectors to AWS RDS PostgreSQL (pgvector)."""
    print_banner("STAGE 5: AWS RDS CLOUD SYNC (PostgreSQL + pgvector)")
    try:
        from scripts import migrate_sqlite_to_rds
    except ImportError:
        print("ERROR: Could not import scripts.migrate_sqlite_to_rds.")
        return False

    t0 = time.time()
    try:
        migrate_sqlite_to_rds.main()
        elapsed = time.time() - t0
        print(f"\nStage 5 Complete: AWS RDS updated in {elapsed:.1f}s.")
        return True
    except Exception as e:
        print(f"Stage 5 Error: {e}")
        return False


# =============================================================================
# VERIFICATION & SEMANTIC QUERY
# =============================================================================
def run_semantic_query(query: str, top_k: int = 3, book_filter: Optional[str] = None, use_rds: bool = False):
    """Execute a live similarity search using the unified database layer."""
    print_banner(f"SEMANTIC SEARCH: '{query}'")
    from scripts import db

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers is required for semantic query.")
        return

    print("Encoding query using sentence-transformers/all-MiniLM-L6-v2 ...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_vec = embedder.encode(query, normalize_embeddings=True)

    t0 = time.time()
    if use_rds:
        print("Target: AWS RDS PostgreSQL (pgvector)")
        pg_conn = db.get_rds_connection()
        if not pg_conn:
            print("ERROR: Could not connect to AWS RDS. Check .env configuration.")
            return
        results = db.search_chunks_rds(pg_conn, query_vector=q_vec, top_k=top_k, book_id=book_filter)
        pg_conn.close()
    else:
        print(f"Target: Local SQLite ({DB_PATH.name})")
        if not DB_PATH.exists():
            print(f"ERROR: Local database {DB_PATH} not found. Run Stage 4 first.")
            return
        conn = db.get_connection(DB_PATH)
        results = db.search_chunks(conn, query_vector=q_vec, top_k=top_k, book_id=book_filter)
        conn.close()

    elapsed = time.time() - t0
    print(f"Search completed in {elapsed:.3f}s. Results ({len(results)} matches):\n")

    for i, r in enumerate(results, start=1):
        score = r.get("similarity", 1.0 - r.get("distance", 0.0))
        print(f"[{i}] {r.get('book_id', r.get('book', 'Unknown'))} | {r.get('chapter', '')} | Score: {score:.4f}")
        print(f"    Breadcrumb : {r.get('breadcrumb', 'N/A')}")
        print(f"    Pages      : {r.get('page_start', '?')} - {r.get('page_end', '?')}")
        if r.get("image_paths"):
            print(f"    Diagrams   : {r.get('image_paths')}")
        snippet = (r.get("text", "")[:220] + "...").replace("\n", " ")
        print(f"    Snippet    : {snippet}\n")


# =============================================================================
# SYSTEM STATUS & BYTECODE COMPILATION
# =============================================================================
def run_compile_check() -> bool:
    """Compile all Python files in the repository to bytecode to verify syntax."""
    print_banner("BYTECODE COMPILATION CHECK")
    print(f"Compiling all Python files under: {BASE_DIR}")
    success = compileall.compile_dir(str(BASE_DIR), maxlevels=4, force=False, quiet=1)
    if success:
        print("[+] All Python files compiled successfully with zero syntax errors!")
    else:
        print("[-] Syntax errors detected in some files.")
    return bool(success)


def show_system_status():
    """Display the health and statistics of all project artifacts and tables."""
    print_banner("PREPPILOT SYSTEM STATUS")

    print("\n1. Textbooks (PDFs):")
    for book_id, cfg in BOOKS_CONFIG.items():
        pdf = find_pdf(book_id)
        if pdf:
            mb = pdf.stat().st_size / (1024 * 1024)
            print(f"   [+] {book_id:4s} : FOUND ({pdf.name}, {mb:.1f} MB)")
        else:
            print(f"   [-] {book_id:4s} : MISSING")

    print("\n2. Extracted Raw Texts:")
    for book_id, cfg in BOOKS_CONFIG.items():
        p = EXTRACTED_DIR / cfg["raw_name"]
        if p.exists():
            mb = p.stat().st_size / (1024 * 1024)
            print(f"   [+] {cfg['raw_name']:12s} : {mb:.2f} MB")
        else:
            print(f"   [-] {cfg['raw_name']:12s} : NOT EXTRACTED")

    print("\n3. Semantic Chunks (JSONL):")
    for book_id, cfg in BOOKS_CONFIG.items():
        p = CHUNKS_DIR / cfg["chunk_name"]
        if p.exists():
            lines = sum(1 for _ in p.open("r", encoding="utf-8"))
            print(f"   [+] {cfg['chunk_name']:18s} : {lines:,} chunks")
        else:
            print(f"   [-] {cfg['chunk_name']:18s} : NOT CHUNKED")

    print(f"\n4. Local Database ({DB_PATH.name}):")
    if DB_PATH.exists():
        import sqlite3
        mb = DB_PATH.stat().st_size / (1024 * 1024)
        print(f"   Size: {mb:.2f} MB")
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cur = conn.cursor()
            for tbl in ["chunks", "extracted_images", "students", "topic_mastery", "quiz_attempts", "pyq_questions"]:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                    cnt = cur.fetchone()[0]
                    print(f"   • {tbl:18s} : {cnt:,} rows")
                except sqlite3.OperationalError:
                    print(f"   • {tbl:18s} : Table not initialized")
            conn.close()
        except Exception as e:
            print(f"   Database read error: {e}")
    else:
        print("   [-] Database not created yet. Run Stage 4 or 'python main.py'")

    print("\n" + "=" * 70)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="PrepPilot Unified Master Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                         # Run full pipeline end-to-end
  python main.py --status                # Show status of all datasets & database
  python main.py --query "binary search" # Test semantic vector search
  python main.py --rds                   # Sync local database to AWS RDS
  python main.py --compile-check         # Verify syntax across all Python files
        """
    )
    parser.add_argument("--status", action="store_true", help="Display artifact and database status")
    parser.add_argument("--extract-text", action="store_true", help="Run Stage 1 (PDF text extraction)")
    parser.add_argument("--extract-images", action="store_true", help="Run Stage 2 (Diagram & figure cropping)")
    parser.add_argument("--chunk", action="store_true", help="Run Stage 3 (Contextual semantic chunking)")
    parser.add_argument("--embed", action="store_true", help="Run Stage 4 (MiniLM embedding & SQLite DB ingestion)")
    parser.add_argument("--rds", action="store_true", help="Run Stage 5 (Sync SQLite database to AWS RDS PostgreSQL)")
    parser.add_argument("--query", type=str, default=None, help="Run a test semantic similarity search query")
    parser.add_argument("--top-k", type=int, default=3, help="Number of search results to return (default: 3)")
    parser.add_argument("--book", type=str, default=None, help="Filter to specific book (DS, C, CPP, CLRS)")
    parser.add_argument("--compile-check", action="store_true", help="Bytecode compile all project Python files")

    args = parser.parse_args()

    # 1. Standalone commands
    if args.status:
        show_system_status()
        return

    if args.compile_check:
        run_compile_check()
        return

    if args.query:
        run_semantic_query(args.query, top_k=args.top_k, book_filter=args.book, use_rds=args.rds)
        return

    # 2. Granular step execution
    ran_individual = False
    if args.extract_text:
        run_stage_1_pdf_extraction(book_filter=args.book)
        ran_individual = True
    if args.extract_images:
        run_stage_2_image_extraction(book_filter=args.book)
        ran_individual = True
    if args.chunk:
        run_stage_3_chunking(book_filter=args.book)
        ran_individual = True
    if args.embed:
        run_stage_4_database_embedding(book_filter=args.book)
        ran_individual = True
    if args.rds and not ran_individual:
        run_stage_5_rds_sync()
        return

    if ran_individual:
        if args.rds:
            run_stage_5_rds_sync()
        return

    # 3. Default: Full End-to-End Execution
    start_time = time.time()
    print_banner("PREPPILOT MASTER PIPELINE: FULL EXECUTION")
    print(f"Root: {BASE_DIR}")

    # Stage 1: Text
    ok1 = run_stage_1_pdf_extraction(book_filter=args.book)
    # Stage 2: Images/Diagrams
    ok2 = run_stage_2_image_extraction(book_filter=args.book)
    # Stage 3: Chunks
    ok3 = run_stage_3_chunking(book_filter=args.book)
    # Stage 4: Embed & Ingest to SQLite
    ok4 = run_stage_4_database_embedding(book_filter=args.book)

    # Optional RDS sync
    if args.rds:
        run_stage_5_rds_sync()

    # Default quick verification search
    if ok4 and DB_PATH.exists():
        run_semantic_query("What is a binary search tree?", top_k=2)

    total_time = time.time() - start_time
    print_banner(f"PIPELINE EXECUTION FINISHED IN {total_time:.1f} SECONDS")
    show_system_status()


if __name__ == "__main__":
    main()
