import sqlite3

conn = sqlite3.connect('preppilot.db')
c = conn.cursor()

tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables:", tables)

for b in ['CLRS', 'DS', 'C', 'CPP']:
    row = c.execute("SELECT count(*), min(chunk_id), max(chunk_id) FROM chunks WHERE book_id = ?", (b,)).fetchone()
    print(f"Book {b}: count={row[0]}, min={row[1]}, max={row[2]}")
    sample = c.execute("SELECT chunk_id, chapter, chapter_title FROM chunks WHERE book_id = ? AND chapter IS NOT NULL LIMIT 3", (b,)).fetchall()
    print(f"  Sample: {sample}")
