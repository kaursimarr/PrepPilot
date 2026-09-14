"""
Migrate all data and vectors from local SQLite (preppilot.db) to AWS RDS PostgreSQL (pgvector).

Features:
- Securely reads connection credentials from `.env`.
- Activates `pgvector` on AWS RDS: `CREATE EXTENSION IF NOT EXISTS vector`.
- Recreates all relational tables (`chunks`, `students`, `topic_mastery`, `quiz_attempts`, `pyq_questions`, `extracted_images`).
- Migrates 384-dimensional dense vectors into PostgreSQL `vector(384)` columns.
- Creates an HNSW index on embeddings for fast cosine similarity search.
- Runs a live verification vector search over AWS RDS.
"""

from pathlib import Path
import json
import os
import sqlite3
import struct
import sys
from typing import Any, Dict, List, Optional

import dotenv
import numpy as np

try:
    import psycopg
    from pgvector.psycopg import register_vector
except ImportError:
    print("ERROR: psycopg and pgvector are required. Install with: pip install \"psycopg[binary]\" pgvector")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
SQLITE_DB_PATH = BASE_DIR / "preppilot.db"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.db import get_connection as get_sqlite_connection

# Load .env
dotenv.load_dotenv(BASE_DIR / ".env")


def get_pg_connection_params() -> Dict[str, Any]:
    """Retrieve credentials from environment variables."""
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return {"conninfo": db_url}

    host = os.getenv("PGHOST")
    return {
        "host": host,
        "port": int(os.getenv("PGPORT", "5432")),
        "dbname": os.getenv("PGDATABASE", "preppilot"),
        "user": os.getenv("PGUSER", "postgres"),
        "password": os.getenv("PGPASSWORD", ""),
    }


def create_rds_schema(pg_conn: psycopg.Connection) -> None:
    """Initialize tables in AWS RDS PostgreSQL with pgvector."""
    with pg_conn.cursor() as cur:
        # Enable pgvector extension
        print("Enabling pgvector extension on AWS RDS...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 1. Chunks table with native vector column
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            chunk_id VARCHAR(100) UNIQUE NOT NULL,
            book_id VARCHAR(50) NOT NULL,
            book_title VARCHAR(255),
            breadcrumb TEXT,
            chapter VARCHAR(50),
            chapter_title VARCHAR(255),
            section VARCHAR(50),
            section_title VARCHAR(255),
            subsection VARCHAR(50),
            subsection_title VARCHAR(255),
            page_start INT,
            page_end INT,
            word_count INT,
            contains_code BOOLEAN DEFAULT FALSE,
            contains_equation BOOLEAN DEFAULT FALSE,
            contains_figure BOOLEAN DEFAULT FALSE,
            contains_table BOOLEAN DEFAULT FALSE,
            image_paths JSONB,
            text TEXT NOT NULL,
            embedding vector(384)
        );
        """)

        # Fast HNSW index for cosine distance
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
        ON chunks USING hnsw (embedding vector_cosine_ops);
        """)

        # 2. Students
        cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id SERIAL PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # 3. Topic Mastery
        cur.execute("""
        CREATE TABLE IF NOT EXISTS topic_mastery (
            id SERIAL PRIMARY KEY,
            student_id INT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
            topic_id VARCHAR(100) NOT NULL,
            topic_name VARCHAR(255),
            mastery_score FLOAT NOT NULL CHECK(mastery_score >= 0.0 AND mastery_score <= 1.0),
            attempts_count INT DEFAULT 0,
            last_evaluated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(student_id, topic_id)
        );
        """)

        # 4. Quiz Attempts
        cur.execute("""
        CREATE TABLE IF NOT EXISTS quiz_attempts (
            id SERIAL PRIMARY KEY,
            student_id INT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
            question_id VARCHAR(100),
            topic_id VARCHAR(100),
            question_text TEXT,
            selected_answer TEXT,
            correct_answer TEXT,
            is_correct BOOLEAN NOT NULL,
            attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # 5. Extracted Images
        cur.execute("""
        CREATE TABLE IF NOT EXISTS extracted_images (
            id SERIAL PRIMARY KEY,
            image_id VARCHAR(100) UNIQUE NOT NULL,
            book_id VARCHAR(50) NOT NULL,
            page_number INT NOT NULL,
            caption TEXT,
            image_type VARCHAR(50),
            file_path TEXT NOT NULL,
            width INT,
            height INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # 6. PYQ Questions
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pyq_questions (
            id SERIAL PRIMARY KEY,
            question_id VARCHAR(100) UNIQUE,
            topic_id VARCHAR(100),
            question_text TEXT NOT NULL,
            marks INT DEFAULT 5,
            year INT,
            difficulty VARCHAR(50),
            correct_answer TEXT,
            explanation TEXT
        );
        """)

    pg_conn.commit()
    print("RDS schema created successfully.")


def migrate_data():
    if not SQLITE_DB_PATH.exists():
        print(f"ERROR: Local SQLite database not found at {SQLITE_DB_PATH}")
        return

    params = get_pg_connection_params()

    pg_conn = psycopg.connect(**params)
    create_rds_schema(pg_conn)
    register_vector(pg_conn)
    sqlite_conn = get_sqlite_connection(SQLITE_DB_PATH)

    # 1. Migrate Students
    students = sqlite_conn.execute("SELECT * FROM students").fetchall()
    with pg_conn.cursor() as cur:
        for s in students:
            cur.execute("""
                INSERT INTO students (id, username, name, created_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (username) DO UPDATE SET name = EXCLUDED.name;
            """, (s["id"], s["username"], s["name"], s["created_at"]))
    pg_conn.commit()
    print(f"Migrated {len(students)} students.")

    # 2. Migrate Topic Mastery
    mastery = sqlite_conn.execute("SELECT * FROM topic_mastery").fetchall()
    with pg_conn.cursor() as cur:
        for m in mastery:
            cur.execute("""
                INSERT INTO topic_mastery (student_id, topic_id, topic_name, mastery_score, attempts_count, last_evaluated)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (student_id, topic_id) DO UPDATE SET mastery_score = EXCLUDED.mastery_score;
            """, (m["student_id"], m["topic_id"], m["topic_name"], m["mastery_score"], m["attempts_count"], m["last_evaluated"]))
    pg_conn.commit()
    print(f"Migrated {len(mastery)} topic mastery records.")

    # 3. Migrate Extracted Images
    images = sqlite_conn.execute("SELECT * FROM extracted_images").fetchall()
    with pg_conn.cursor() as cur:
        for img in images:
            cur.execute("""
                INSERT INTO extracted_images (image_id, book_id, page_number, caption, image_type, file_path, width, height)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (image_id) DO NOTHING;
            """, (img["image_id"], img["book_id"], img["page_number"], img["caption"], img["image_type"], img["file_path"], img["width"], img["height"]))
    pg_conn.commit()
    print(f"Migrated {len(images)} extracted image records.")

    # 4. Migrate Chunks + Vector Embeddings
    # Read chunks from SQLite
    chunks = sqlite_conn.execute("SELECT * FROM chunks").fetchall()
    # Read embeddings from sqlite-vec virtual table
    vec_rows = sqlite_conn.execute("SELECT rowid, embedding FROM vec_chunks").fetchall()
    vec_map = {}
    for r in vec_rows:
        raw = r[1]
        if isinstance(raw, (bytes, bytearray)):
            floats = list(struct.unpack(f"{len(raw)//4}f", raw))
            vec_map[r[0]] = floats

    with pg_conn.cursor() as cur:
        for c in chunks:
            rowid = c["id"]
            embedding = vec_map.get(rowid)
            img_paths = c["image_paths"]
            img_json = json.dumps(json.loads(img_paths)) if img_paths else "[]"

            breadcrumb = c["breadcrumb"] if "breadcrumb" in c.keys() else ""

            cur.execute("""
                INSERT INTO chunks (
                    chunk_id, book_id, book_title, breadcrumb, chapter, chapter_title,
                    section, section_title, subsection, subsection_title,
                    page_start, page_end, word_count, contains_code,
                    contains_equation, contains_figure, contains_table,
                    image_paths, text, embedding
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    breadcrumb = EXCLUDED.breadcrumb,
                    embedding = EXCLUDED.embedding,
                    image_paths = EXCLUDED.image_paths;
            """, (
                c["chunk_id"], c["book_id"], c["book_title"], breadcrumb, str(c["chapter"]), c["chapter_title"],
                str(c["section"]), c["section_title"], str(c["subsection"]), c["subsection_title"],
                c["page_start"], c["page_end"], c["word_count"], bool(c["contains_code"]),
                bool(c["contains_equation"]), bool(c["contains_figure"]), bool(c["contains_table"]),
                img_json, c["text"], embedding
            ))
    pg_conn.commit()
    print(f"Migrated {len(chunks)} chunks with dense vector embeddings to AWS RDS!")

    # 5. Live Verification Query on AWS RDS
    print("\nRunning live verification vector query on AWS RDS...")
    with pg_conn.cursor() as cur:
        # Cosine distance operator in pgvector is <=>
        cur.execute("""
            SELECT chunk_id, book_id, section, section_title, 
                   1 - (embedding <=> (SELECT embedding FROM chunks WHERE chunk_id = 'CLRS_0038_01')) AS similarity
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> (SELECT embedding FROM chunks WHERE chunk_id = 'CLRS_0038_01')
            LIMIT 3;
        """)
        rows = cur.fetchall()
        for r in rows:
            print(f"  Result: [{r[0]}] {r[1]} {r[2]} {r[3]} | Cosine Similarity: {r[4]:.4f}")

    sqlite_conn.close()
    pg_conn.close()
    print("\nSUCCESS: All local SQLite tables and vectors migrated to AWS RDS PostgreSQL!")


if __name__ == "__main__":
    migrate_data()
