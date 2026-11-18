import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "preppilot.db"


def conn() -> sqlite3.Connection:
    """Plain connection for the new feature tables. Open per request, close after."""
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init_schema():
    sql = (Path(__file__).parent / "schema_features.sql").read_text(encoding="utf-8")
    c = conn()
    try:
        c.executescript(sql)
        c.commit()
    finally:
        c.close()