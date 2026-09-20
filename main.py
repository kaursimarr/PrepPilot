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
  Stage 3b: Chapter Titles           (PDF bookmarks -> chapter_titles.json)
  Stage 3c: Chunk Cleanup            (fix_chunks.py -> chunks_semantic_clean/)
  Stage 4: Unified Database Embed    (MiniLM-L6-v2, SQLite + sqlite-vec)
  Stage 5: AWS RDS Cloud Sync        (PostgreSQL + pgvector, optional --rds)
  Stage 6: Retrieval Eval            (25 test questions, --eval)
  Also:    Semantic Search           (--query "...")

Usage:
  python main.py                    # Run complete pipeline end-to-end
  python main.py --status           # Display current artifact and database status
  python main.py --query "query"    # Test semantic vector search on indexed chunks
  python main.py --extract-text     # Run Stage 1 only
  python main.py --extract-images   # Run Stage 2 only
  python main.py --chunk            # Run Stage 3 only
  python main.py --titles           # Run Stage 3b only
  python main.py --fix              # Run Stage 3c only
  python main.py --embed            # Run Stage 4 only
  python main.py --schema           # Print preppilot.db tables, columns and a sample row
  python main.py --find "term"      # Is a topic indexed or excluded? (add --book C)
  python main.py --tidy             # Move dropped/excluded audit files into chunk_audit/
  python main.py --eval             # Run Stage 6 only (add --verbose for top hits)
  python main.py --fresh-db         # Full run, but back up + delete preppilot.db first
  python main.py --rds              # Sync to AWS RDS PostgreSQL
  python main.py --compile-check    # Compile all repo Python files into bytecode
=============================================================================
"""
from __future__ import annotations
from pathlib import Path
import argparse
import compileall
import importlib.util
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
AUDIT_DIR = BASE_DIR / "chunk_audit"                  # dropped/excluded pieces are moved here
CLEAN_DIR = BASE_DIR / "chunks_semantic_clean"      # output of fix_chunks.py (this is what gets embedded)
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
        "raw_name": "APS_raw.txt",
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
        )
        if extracted:
            extract_images.save_metadata_to_db(extracted)
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
        expected_raw = BOOKS_CONFIG[book_id]["raw_name"]
        if Path(cfg["input"]).name != expected_raw:
            print(f"  [!] {book_id}: Stage 1 writes {expected_raw} but semantic_chunker.py reads {Path(cfg['input']).name}. "
                  f"Make the two names match (edit semantic_chunker.py) or Stage 3 will chunk the wrong or a missing file.")
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

    archive_audit_files()
    print(f"\nStage 3 Complete: {chunked_count}/{len(books)} books chunked into {CHUNKS_DIR}.")
    return chunked_count > 0


def _load_script(label: str, candidates: List[str]):
    """Import a helper script by file path, looking in the project root and scripts/.
    Loaded lazily so import-time work (e.g. fix_chunks reading chapter_titles.json)
    happens after the earlier stage has produced its files."""
    for folder in (BASE_DIR, SCRIPTS_DIR):
        for name in candidates:
            path = folder / name
            if path.exists():
                spec = importlib.util.spec_from_file_location(f"_pp_{path.stem}", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
    print(f"ERROR: {label} not found. Looked for {', '.join(candidates)} in {BASE_DIR} and {SCRIPTS_DIR}.")
    return None


# =============================================================================
# STAGE 3b: CHAPTER TITLES (PDF bookmarks -> chapter_titles.json)
# =============================================================================
def run_stage_3b_chapter_titles() -> bool:
    """Read each PDF's bookmarks and write chapter_titles.json for fix_chunks.py."""
    print_banner("STAGE 3b: CHAPTER TITLES FROM PDF BOOKMARKS")
    mod = _load_script("chapter titles script", ["make_chapter_titles.py", "chapter_title.py"])
    if mod is None:
        return False
    try:
        mod.main()
        print("\n  Check the printed titles above. Fix any wrong ones in CHAPTER_TITLE_OVERRIDES inside fix_chunks.py.")
        return True
    except Exception as e:
        print(f"Stage 3b Error: {e}")
        return False


# =============================================================================
# STAGE 3c: CHUNK CLEANUP (fix_chunks.py)
# =============================================================================
def run_stage_3c_fix_chunks(book_filter: Optional[str] = None) -> bool:
    """Clean and re-split chunks_semantic/ into chunks_semantic_clean/ (what Stage 4 embeds)."""
    print_banner("STAGE 3c: CHUNK CLEANUP (fix_chunks)")
    mod = _load_script("fix_chunks.py", ["fix_chunks.py"])
    if mod is None:
        return False
    books = [book_filter] if book_filter else list(mod.BOOKS)
    t0 = time.time()
    try:
        count = mod.make_token_counter()
        report: List[str] = []
        mod.DST.mkdir(parents=True, exist_ok=True)
        for b in books:
            mod.process_book(b, count, report)
        (mod.DST / "fix_report.txt").write_text("\n".join(report), encoding="utf-8")
        print("\n".join(report))
    except Exception as e:
        print(f"Stage 3c Error: {e}")
        return False

    missing = [b for b in books if not (mod.DST / f"{b}_chunks.jsonl").exists()]
    if missing:
        print(f"  [!] No cleaned output for: {', '.join(missing)}")
        return False
    archive_audit_files()
    print(f"\nStage 3c Complete: cleaned chunks in {mod.DST} ({time.time() - t0:.1f}s). Report: fix_report.txt")
    return True


def archive_audit_files() -> int:
    """Move <BOOK>_dropped.jsonl (Stage 3) and <BOOK>_excluded.jsonl (Stage 3c) into chunk_audit/,
    so chunks_semantic/ and chunks_semantic_clean/ hold only the chunk files that are actually used.
    Files with the same name are overwritten, so chunk_audit/ always reflects the latest run."""
    moved = 0
    for folder, pattern in ((CHUNKS_DIR, "*_dropped.jsonl"), (CLEAN_DIR, "*_excluded.jsonl")):
        if not folder.exists():
            continue
        for f in sorted(folder.glob(pattern)):
            AUDIT_DIR.mkdir(parents=True, exist_ok=True)
            os.replace(f, AUDIT_DIR / f.name)
            moved += 1
    if moved:
        print(f"  Moved {moved} audit file(s) (dropped/excluded pieces) into {AUDIT_DIR.name}/")
    return moved


def run_find(terms: List[str], book_filter: Optional[str] = None) -> None:
    """Show where each term ended up: in the indexed chunks or in the excluded pieces."""
    import json
    print_banner("WHERE DID THIS TOPIC END UP?")
    books = [book_filter] if book_filter else list(BOOKS_CONFIG.keys())
    for book_id in books:
        if book_id not in BOOKS_CONFIG:
            print(f"Unknown book: {book_id}")
            continue
        name = BOOKS_CONFIG[book_id]["chunk_name"]
        excl_name = name.replace("_chunks", "_excluded")
        sources = [("indexed", CLEAN_DIR / name),
                   ("EXCLUDED", AUDIT_DIR / excl_name if (AUDIT_DIR / excl_name).exists() else CLEAN_DIR / excl_name)]
        data = {}
        for label, path in sources:
            if path.exists():
                data[label] = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
            else:
                data[label] = None
                print(f"  [!] {book_id}: file not found -> {path}")
        print(f"\n{book_id}: {len(data['indexed'] or []):,} indexed pieces, {len(data['EXCLUDED'] or []):,} excluded pieces")
        for term in terms:
            t = term.lower()
            print(f"  '{term}'")
            for label in ("indexed", "EXCLUDED"):
                rows = data[label]
                if rows is None:
                    continue
                hits = [r for r in rows if t in (r.get("text", "") + " " + str(r.get("section_title", ""))).lower()]
                print(f"    {label:9s}: {len(hits)} pieces")
                for r in hits[:2]:
                    i = r.get("text", "").lower().find(t)
                    snip = r["text"][max(0, i - 60): i + 90].replace("\n", " ") if i >= 0 else ""
                    why = f" [{r['exclude_reason']}]" if label == "EXCLUDED" and r.get("exclude_reason") else ""
                    print(f"        p{r.get('page_start')} ch{r.get('chapter')}{why} ...{snip}...")


def show_db_schema() -> None:
    """Print every table in preppilot.db: its SQL, columns, row count and one sample row (read-only)."""
    import sqlite3
    print_banner("DATABASE SCHEMA (read-only)")
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found.")
        return
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    objs = conn.execute("SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name").fetchall()
    for typ, name, sql in objs:
        print(f"\n[{typ}] {name}")
        if sql:
            print("  " + sql.replace("\n", "\n  "))
        if typ != "table":
            continue
        try:
            cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{name}")')]
            n = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            print(f"  columns: {', '.join(cols)}")
            print(f"  rows: {n:,}")
            row = conn.execute(f'SELECT * FROM "{name}" LIMIT 1').fetchone()
            if row:
                shown = []
                for c, v in zip(cols, row):
                    if isinstance(v, (bytes, bytearray)):
                        v = f"<blob {len(v)} bytes>"
                    elif isinstance(v, str) and len(v) > 90:
                        v = v[:90] + "..."
                    shown.append(f"{c}={v!r}")
                print("  sample: " + " | ".join(shown))
        except Exception as e:  # virtual tables (sqlite-vec) need the extension loaded
            print(f"  (could not read: {e})")
    conn.close()


def audit_chunk_counts() -> bool:
    """Show raw parent chunks -> cleaned pieces -> rows in preppilot.db, and flag any mismatch.
    The cleaned pieces (not the raw parents) are the real RAG units."""
    print("\n  Chunk audit (raw parent chunks -> cleaned pieces -> rows in DB):")
    db_counts: Dict[str, int] = {}
    if DB_PATH.exists():
        import sqlite3
        try:
            conn = sqlite3.connect(str(DB_PATH))
            db_counts = dict(conn.execute("SELECT book_id, COUNT(*) FROM chunks GROUP BY book_id").fetchall())
            conn.close()
        except Exception as e:  # noqa: BLE001
            print(f"  (could not read chunk counts from the DB: {e})")
    all_ok = True
    for book_id, cfg in BOOKS_CONFIG.items():
        raw_p = CHUNKS_DIR / cfg["chunk_name"]
        clean_p = CLEAN_DIR / cfg["chunk_name"]
        n_raw = sum(1 for l in raw_p.open(encoding="utf-8") if l.strip()) if raw_p.exists() else 0
        n_clean, max_words = 0, 0
        if clean_p.exists():
            import json
            for l in clean_p.open(encoding="utf-8"):
                if l.strip():
                    n_clean += 1
                    max_words = max(max_words, json.loads(l).get("word_count", 0))
        n_db = db_counts.get(book_id)
        flag = ""
        if n_db is not None and n_db != n_clean:
            flag = "  <-- DB does not match the cleaned file"
            all_ok = False
        db_txt = f"{n_db:,}" if n_db is not None else "n/a"
        print(f"   {book_id:4s} raw {n_raw:>6,}  clean {n_clean:>6,} (longest {max_words} words)  db {db_txt:>6}{flag}")
    return all_ok


def _warn_if_clean_chunks_stale(book_filter: Optional[str] = None) -> bool:
    """Return False if chunks_semantic_clean/ is missing or older than chunks_semantic/."""
    books = [book_filter] if book_filter else list(BOOKS_CONFIG.keys())
    ok = True
    for b in books:
        raw = CHUNKS_DIR / BOOKS_CONFIG[b]["chunk_name"]
        clean = CLEAN_DIR / BOOKS_CONFIG[b]["chunk_name"]
        if not clean.exists():
            print(f"  [!] {b}: no cleaned chunks at {clean}. Run --fix first.")
            ok = False
        elif raw.exists() and clean.stat().st_mtime < raw.stat().st_mtime:
            print(f"  [!] {b}: cleaned chunks are older than the raw chunks. Run --fix again.")
            ok = False
    return ok


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
    if not _warn_if_clean_chunks_stale(book_filter):
        print("  [!] Continuing anyway. If migrate_to_sqlite.py reads chunks_semantic_clean/, you are embedding stale or missing files.")
    t0 = time.time()
    try:
        migrate_to_sqlite.migrate(
            book_filter=book_filter,
            max_chunks_per_book=None,
        )
        elapsed = time.time() - t0
        print(f"\nStage 4 Complete: Database indexed in {elapsed:.1f}s.")
        if not audit_chunk_counts():
            print("  [!] The database does not hold the cleaned chunks. Check which folder migrate_to_sqlite.py reads.")
            return False
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
# STAGE 6: RETRIEVAL EVAL (25 test questions across the four books)
# =============================================================================
def run_stage_6_eval(verbose: bool = False, k: int = 5) -> bool:
    """Run the retrieval smoke test against preppilot.db."""
    print_banner("STAGE 6: RETRIEVAL EVAL")
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found. Run Stage 4 first.")
        return False
    mod = _load_script("eval script", ["eval_retrieval.py", "eval.py"])
    if mod is None:
        return False
    # The eval script has its own argparse; give it a clean argv instead of main.py's.
    saved_argv = sys.argv
    sys.argv = ["eval_retrieval.py", "--k", str(k)] + (["--verbose"] if verbose else [])
    try:
        mod.main()
        return True
    except Exception as e:
        print(f"Stage 6 Error: {e}")
        return False
    finally:
        sys.argv = saved_argv


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

    print("\n3b. Cleaned Chunks (chunks_semantic_clean, what gets embedded):")
    for book_id, cfg in BOOKS_CONFIG.items():
        p = CLEAN_DIR / cfg["chunk_name"]
        if p.exists():
            lines = sum(1 for _ in p.open("r", encoding="utf-8"))
            print(f"   [+] {cfg['chunk_name']:18s} : {lines:,} chunks")
        else:
            print(f"   [-] {cfg['chunk_name']:18s} : NOT CLEANED (run --fix)")

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

    if DB_PATH.exists():
        audit_chunk_counts()

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
    parser.add_argument("--titles", action="store_true", help="Run Stage 3b (chapter titles from PDF bookmarks)")
    parser.add_argument("--fix", action="store_true", help="Run Stage 3c (fix_chunks cleanup into chunks_semantic_clean/)")
    parser.add_argument("--schema", action="store_true", help="Print the tables, columns, row counts and a sample row of preppilot.db")
    parser.add_argument("--find", nargs="+", metavar="TERM",
                        help='Show whether a topic is indexed or excluded, e.g. --find "call by value" --book C')
    parser.add_argument("--tidy", action="store_true", help="Move *_dropped.jsonl / *_excluded.jsonl into chunk_audit/")
    parser.add_argument("--embed", action="store_true", help="Run Stage 4 (MiniLM embedding & SQLite DB ingestion)")
    parser.add_argument("--eval", action="store_true", help="Run Stage 6 (retrieval eval, 25 questions)")
    parser.add_argument("--verbose", action="store_true", help="With --eval: show top results for every question")
    parser.add_argument("--fresh-db", action="store_true",
                        help="Full run only: back up preppilot.db to preppilot.db.bak-<time>, then delete it before rebuilding")
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

    if args.schema:
        show_db_schema()
        return

    if args.find:
        run_find(args.find, book_filter=args.book)
        return

    if args.tidy:
        archive_audit_files()
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
    if args.titles:
        run_stage_3b_chapter_titles()
        ran_individual = True
    if args.fix:
        run_stage_3c_fix_chunks(book_filter=args.book)
        ran_individual = True
    if args.embed:
        run_stage_4_database_embedding(book_filter=args.book)
        ran_individual = True
    if args.eval:
        run_stage_6_eval(verbose=args.verbose)
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

    if args.fresh_db and DB_PATH.exists():
        backup = DB_PATH.with_name(f"{DB_PATH.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(DB_PATH, backup)
        DB_PATH.unlink()
        print(f"Backed up old database to {backup.name} and deleted {DB_PATH.name}.")
    elif DB_PATH.exists():
        print(f"NOTE: {DB_PATH.name} already exists. Pass --fresh-db if you need to drop stale chunks from it.")

    # Stage 1: Text
    ok1 = run_stage_1_pdf_extraction(book_filter=args.book)
    # Stage 2: Images/Diagrams
    ok2 = run_stage_2_image_extraction(book_filter=args.book)
    # Stage 3: Chunks
    ok3 = run_stage_3_chunking(book_filter=args.book)
    # Stage 3b/3c: chapter titles, then chunk cleanup (must finish before embedding)
    ok3b = run_stage_3b_chapter_titles()
    ok3c = run_stage_3c_fix_chunks(book_filter=args.book)
    if not ok3c:
        print("\nStopping before Stage 4: chunk cleanup failed, so embedding now would index stale chunks.")
        print_banner("PIPELINE STOPPED AT STAGE 3c")
        show_system_status()
        return
    # Stage 4: Embed & Ingest to SQLite
    ok4 = run_stage_4_database_embedding(book_filter=args.book)

    # Optional RDS sync
    if args.rds:
        run_stage_5_rds_sync()

    # Default quick verification search
    if ok4 and DB_PATH.exists():
        run_semantic_query("What is a binary search tree?", top_k=2)
        run_stage_6_eval()

    total_time = time.time() - start_time
    print_banner(f"PIPELINE EXECUTION FINISHED IN {total_time:.1f} SECONDS")
    show_system_status()


if __name__ == "__main__":
    main()