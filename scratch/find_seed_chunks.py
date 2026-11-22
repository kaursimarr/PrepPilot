import sqlite3

conn = sqlite3.connect('preppilot.db')
c = conn.cursor()

def find_chunk(book_id, query):
    res = c.execute(
        "SELECT chunk_id, chapter, chapter_title, section, page_start FROM chunks WHERE book_id = ? AND text LIKE ? LIMIT 1",
        (book_id, f"%{query}%")
    ).fetchone()
    if res:
        return res
    # fallback
    return c.execute("SELECT chunk_id, chapter, chapter_title, section, page_start FROM chunks WHERE book_id = ? LIMIT 1", (book_id,)).fetchone()

searches = {
    'CLRS': ['asymptotic', 'divide-and-conquer', 'dynamic programming', 'greedy', 'Dijkstra', 'breadth-first'],
    'DS': ['stack', 'queue', 'linked list', 'binary search tree', 'hash', 'heap'],
    'C': ['pointer', 'malloc', 'structure', 'function', 'array'],
    'CPP': ['class', 'virtual', 'template', 'inheritance', 'destructor']
}

for b, qlist in searches.items():
    print(f"=== {b} ===")
    for q in qlist:
        row = find_chunk(b, q)
        print(f"  Query '{q}': {row}")
