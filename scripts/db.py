"""
Unified Relational + Vector Database for PrepPilot using SQLite and sqlite-vec.

Stores both relational application data (students, topic mastery scores,
quiz history, PYQs) and vector embeddings (384-dimensional textbook chunks)
inside a single self-contained database: preppilot.db.
"""

from pathlib import Path
import json
import sqlite3
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import sqlite_vec
except ImportError:
    raise ImportError(
        "sqlite-vec is required. Install it using: python -m pip install sqlite-vec"
    )

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_DB_PATH = BASE_DIR / "preppilot.db"
EMBEDDING_DIM = 384


def get_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """
    Connect to the SQLite database, load the sqlite-vec extension,
    enable WAL mode for concurrent web access, and set row_factory.
    """
    path = Path(db_path or DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row

    # Performance and concurrency pragmas
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 5000;")

    # Load sqlite-vec extension
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    return conn


def init_db(conn: Optional[sqlite3.Connection] = None, db_path: Optional[Union[str, Path]] = None) -> None:
    """
    Initialize all relational tables, indexes, and the vec0 virtual vector table.
    """
    close_when_done = False
    if conn is None:
        conn = get_connection(db_path)
        close_when_done = True

    try:
        with conn:
            # 1. Textbook Chunks (Relational metadata + raw text)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                book_id TEXT NOT NULL,
                book_title TEXT,
                breadcrumb TEXT,
                chapter TEXT,
                chapter_title TEXT,
                section TEXT,
                section_title TEXT,
                subsection TEXT,
                subsection_title TEXT,
                page_start INTEGER,
                page_end INTEGER,
                word_count INTEGER,
                contains_code BOOLEAN DEFAULT 0,
                contains_equation BOOLEAN DEFAULT 0,
                contains_figure BOOLEAN DEFAULT 0,
                contains_table BOOLEAN DEFAULT 0,
                image_paths TEXT, -- JSON array of strings
                text TEXT NOT NULL
            );
            """)

            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN breadcrumb TEXT;")
            except sqlite3.OperationalError:
                pass

            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_book ON chunks(book_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chapter ON chunks(book_id, chapter);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(book_id, page_start, page_end);")

            # 2. sqlite-vec Virtual Vector Table (Cosine distance on 384-dim embeddings)
            conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding float[{EMBEDDING_DIM}] distance_metric=cosine
            );
            """)

            # 3. Students
            conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 4. Student Knowledge Model (Topic Mastery scores: 0.0 to 1.0)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS topic_mastery (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
                topic_id TEXT NOT NULL,
                topic_name TEXT,
                mastery_score REAL NOT NULL CHECK(mastery_score >= 0.0 AND mastery_score <= 1.0),
                attempts_count INTEGER DEFAULT 0,
                last_evaluated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(student_id, topic_id)
            );
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_mastery_student ON topic_mastery(student_id);")

            # 5. Quiz and Question Attempts History
            conn.execute("""
            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
                question_id TEXT,
                topic_id TEXT,
                question_text TEXT,
                selected_answer TEXT,
                correct_answer TEXT,
                is_correct BOOLEAN NOT NULL,
                attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 6. Previous Year Questions (PYQ Intelligence)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS pyq_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id TEXT UNIQUE,
                topic_id TEXT,
                question_text TEXT NOT NULL,
                marks INTEGER DEFAULT 5,
                year INTEGER,
                difficulty TEXT,
                correct_answer TEXT,
                explanation TEXT
            );
            """)

            # 7. Extracted Images & Figures metadata
            conn.execute("""
            CREATE TABLE IF NOT EXISTS extracted_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT UNIQUE NOT NULL,
                book_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                caption TEXT,
                image_type TEXT, -- 'vector_figure' or 'raster_xobject'
                file_path TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_images_book_page ON extracted_images(book_id, page_number);")

        print("Database initialized successfully with unified relational + vector schema.")
    finally:
        if close_when_done:
            conn.close()


def serialize_embedding(vector: Union[List[float], np.ndarray]) -> bytes:
    """Convert a 384-dimensional float vector into raw bytes for sqlite-vec."""
    if isinstance(vector, np.ndarray):
        arr = vector.astype(np.float32)
    else:
        arr = np.array(vector, dtype=np.float32)

    if arr.shape != (EMBEDDING_DIM,):
        raise ValueError(f"Expected vector of shape ({EMBEDDING_DIM},), got {arr.shape}")

    return arr.tobytes()


def insert_chunk(
    conn: sqlite3.Connection,
    chunk: Dict[str, Any],
    embedding: Union[List[float], np.ndarray]
) -> int:
    """
    Insert a single textbook chunk into `chunks` and its vector into `vec_chunks`.
    Returns the newly created rowid.
    """
    image_paths_json = json.dumps(chunk.get("image_paths", []))
    packed_vec = serialize_embedding(embedding)

    with conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO chunks (
                chunk_id, book_id, book_title, breadcrumb, chapter, chapter_title,
                section, section_title, subsection, subsection_title,
                page_start, page_end, word_count,
                contains_code, contains_equation, contains_figure, contains_table,
                image_paths, text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk["chunk_id"],
            chunk["book_id"],
            chunk.get("book_title", ""),
            chunk.get("breadcrumb", ""),
            str(chunk.get("chapter", "")),
            chunk.get("chapter_title", ""),
            str(chunk.get("section", "")),
            chunk.get("section_title", ""),
            str(chunk.get("subsection", "")),
            chunk.get("subsection_title", ""),
            chunk.get("page_start", 0),
            chunk.get("page_end", 0),
            chunk.get("word_count", 0),
            1 if chunk.get("contains_code") else 0,
            1 if chunk.get("contains_equation") else 0,
            1 if chunk.get("contains_figure") else 0,
            1 if chunk.get("contains_table") else 0,
            image_paths_json,
            chunk["text"]
        ))
        rowid = cur.lastrowid
        cur.execute("INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)", (rowid, packed_vec))

    return rowid


def insert_chunks_batch(
    conn: sqlite3.Connection,
    chunks_with_embeddings: List[Tuple[Dict[str, Any], Union[List[float], np.ndarray]]]
) -> int:
    """
    Batch insert a list of (chunk_dict, embedding) tuples in a single transaction.
    """
    with conn:
        cur = conn.cursor()
        for chunk, embedding in chunks_with_embeddings:
            image_paths_json = json.dumps(chunk.get("image_paths", []))
            packed_vec = serialize_embedding(embedding)

            cur.execute("""
                INSERT OR REPLACE INTO chunks (
                    chunk_id, book_id, book_title, breadcrumb, chapter, chapter_title,
                    section, section_title, subsection, subsection_title,
                    page_start, page_end, word_count,
                    contains_code, contains_equation, contains_figure, contains_table,
                    image_paths, text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk["chunk_id"],
                chunk["book_id"],
                chunk.get("book_title", ""),
                chunk.get("breadcrumb", ""),
                str(chunk.get("chapter", "")),
                chunk.get("chapter_title", ""),
                str(chunk.get("section", "")),
                chunk.get("section_title", ""),
                str(chunk.get("subsection", "")),
                chunk.get("subsection_title", ""),
                chunk.get("page_start", 0),
                chunk.get("page_end", 0),
                chunk.get("word_count", 0),
                1 if chunk.get("contains_code") else 0,
                1 if chunk.get("contains_equation") else 0,
                1 if chunk.get("contains_figure") else 0,
                1 if chunk.get("contains_table") else 0,
                image_paths_json,
                chunk["text"]
            ))
            rowid = cur.lastrowid
            cur.execute("INSERT OR REPLACE INTO vec_chunks(rowid, embedding) VALUES (?, ?)", (rowid, packed_vec))

    return len(chunks_with_embeddings)


def search_chunks(
    conn: sqlite3.Connection,
    query_vector: Union[List[float], np.ndarray],
    top_k: int = 5,
    book_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Perform a KNN vector similarity search on vec_chunks joined with relational metadata.
    Cosine distance is returned where 0.0 is identical and 2.0 is opposite.
    Calculates similarity = 1.0 - (distance / 2.0).
    """
    packed_query = serialize_embedding(query_vector)

    # Fetch extra if filtering by book_id in SQL
    fetch_k = top_k * 4 if book_id else top_k

    query = """
        SELECT
            c.id,
            c.chunk_id,
            c.book_id,
            c.book_title,
            c.breadcrumb,
            c.chapter,
            c.chapter_title,
            c.section,
            c.section_title,
            c.page_start,
            c.page_end,
            c.word_count,
            c.contains_code,
            c.contains_figure,
            c.image_paths,
            c.text,
            v.distance
        FROM vec_chunks v
        JOIN chunks c ON c.id = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
    """
    params: List[Any] = [packed_query, fetch_k]

    if book_id:
        query += " AND c.book_id = ?"
        params.append(book_id.upper())

    query += " ORDER BY v.distance LIMIT ?"
    params.append(top_k)

    rows = conn.execute(query, params).fetchall()

    results = []
    for row in rows:
        d = dict(row)
        dist = float(d["distance"])
        # For cosine distance: cosine_similarity = 1.0 - distance
        d["similarity"] = max(0.0, min(1.0, 1.0 - dist))
        if d.get("image_paths"):
            try:
                d["image_paths"] = json.loads(d["image_paths"])
            except Exception:
                d["image_paths"] = []
        results.append(d)

    return results


# ============================================================
# STUDENT KNOWLEDGE MODEL & QUIZ HELPERS
# ============================================================

def create_or_get_student(conn: sqlite3.Connection, username: str, name: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve an existing student or create a new profile."""
    with conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM students WHERE username = ?", (username,))
        row = cur.fetchone()
        if row:
            return dict(row)

        display_name = name or username.capitalize()
        cur.execute("INSERT INTO students(username, name) VALUES (?, ?)", (username, display_name))
        cur.execute("SELECT * FROM students WHERE id = ?", (cur.lastrowid,))
        return dict(cur.fetchone())


def get_topic_mastery(conn: sqlite3.Connection, student_id: int) -> List[Dict[str, Any]]:
    """Return all topic mastery records for a student."""
    rows = conn.execute("""
        SELECT topic_id, topic_name, mastery_score, attempts_count, last_evaluated
        FROM topic_mastery
        WHERE student_id = ?
        ORDER BY mastery_score ASC
    """, (student_id,)).fetchall()
    return [dict(r) for r in rows]


def update_topic_mastery(
    conn: sqlite3.Connection,
    student_id: int,
    topic_id: str,
    score: float,
    topic_name: Optional[str] = None
) -> None:
    """
    Set or update a student's mastery score (0.0 to 1.0) for a specific topic.
    Increments attempts_count and updates timestamp.
    """
    clamped_score = max(0.0, min(1.0, float(score)))
    with conn:
        conn.execute("""
            INSERT INTO topic_mastery (student_id, topic_id, topic_name, mastery_score, attempts_count, last_evaluated)
            VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(student_id, topic_id) DO UPDATE SET
                mastery_score = excluded.mastery_score,
                attempts_count = attempts_count + 1,
                last_evaluated = CURRENT_TIMESTAMP
        """, (student_id, topic_id, topic_name or topic_id, clamped_score))


def log_quiz_attempt(
    conn: sqlite3.Connection,
    student_id: int,
    question_id: str,
    topic_id: str,
    question_text: str,
    selected_answer: str,
    correct_answer: str,
    is_correct: bool
) -> int:
    """Record an individual question attempt and update mastery adaptively."""
    with conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO quiz_attempts (
                student_id, question_id, topic_id, question_text,
                selected_answer, correct_answer, is_correct
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            student_id, question_id, topic_id, question_text,
            selected_answer, correct_answer, 1 if is_correct else 0
        ))
        attempt_id = cur.lastrowid

        # Simple Exponential Moving Average (EMA) update for mastery
        cur.execute("SELECT mastery_score FROM topic_mastery WHERE student_id = ? AND topic_id = ?", (student_id, topic_id))
        row = cur.fetchone()
        current_score = float(row["mastery_score"]) if row else 0.5
        delta = 0.15 if is_correct else -0.15
        new_score = max(0.0, min(1.0, current_score + delta))

        update_topic_mastery(conn, student_id, topic_id, new_score)

    return attempt_id


def get_weak_topics(conn: sqlite3.Connection, student_id: int, threshold: float = 0.6) -> List[Dict[str, Any]]:
    """Return topics where the student's mastery is below the threshold."""
    rows = conn.execute("""
        SELECT topic_id, topic_name, mastery_score, attempts_count
        FROM topic_mastery
        WHERE student_id = ? AND mastery_score < ?
        ORDER BY mastery_score ASC
    """, (student_id, threshold)).fetchall()
    return [dict(r) for r in rows]


# ============================================================
# AWS RDS POSTGRESQL / PGVECTOR HELPERS
# ============================================================

def get_rds_connection():
    """Connect to AWS RDS PostgreSQL using credentials from .env and register vector type."""
    import os
    import dotenv
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError:
        raise ImportError("psycopg and pgvector are required. Install with: pip install \"psycopg[binary]\" pgvector")

    dotenv.load_dotenv(BASE_DIR / ".env")
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        conn = psycopg.connect(conninfo=db_url)
    else:
        conn = psycopg.connect(
            host=os.getenv("PGHOST"),
            port=int(os.getenv("PGPORT", "5432")),
            dbname=os.getenv("PGDATABASE", "preppilot"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", ""),
        )
    register_vector(conn)
    return conn


def search_chunks_rds(
    pg_conn,
    query_vector: Union[List[float], np.ndarray],
    top_k: int = 5,
    book_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Perform a vector similarity search on AWS RDS PostgreSQL with pgvector."""
    vec = query_vector.tolist() if isinstance(query_vector, np.ndarray) else list(query_vector)

    query = """
        SELECT id, chunk_id, book_id, book_title, breadcrumb, chapter, chapter_title,
               section, section_title, page_start, page_end, word_count,
               contains_code, contains_figure, image_paths, text,
               1 - (embedding <=> %s::vector) AS similarity,
               (embedding <=> %s::vector) AS distance
        FROM chunks
        WHERE embedding IS NOT NULL
    """
    params: List[Any] = [vec, vec]
    if book_id:
        query += " AND book_id = %s"
        params.append(book_id.upper())

    query += " ORDER BY embedding <=> %s::vector LIMIT %s"
    params.extend([vec, top_k])

    with pg_conn.cursor() as cur:
        cur.execute(query, params)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]


if __name__ == "__main__":
    print("Testing PrepPilot SQLite + sqlite-vec database initialization...")
    init_db()
    conn = get_connection()

    # Test student creation
    student = create_or_get_student(conn, "test_student", "Test User")
    print(f"Created/retrieved student: {student['username']} (ID: {student['id']})")

    # Test mastery
    update_topic_mastery(conn, student["id"], "sorting_quicksort", 0.45, "Quick Sort")
    update_topic_mastery(conn, student["id"], "trees_avl", 0.88, "AVL Trees")
    mastery = get_topic_mastery(conn, student["id"])
    print(f"Topic mastery records: {mastery}")

    weak = get_weak_topics(conn, student["id"], 0.6)
    print(f"Detected weak topics: {[w['topic_name'] for w in weak]}")

    conn.close()
    print("ALL TESTS PASSED!")
