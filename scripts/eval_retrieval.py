"""
Retrieval smoke test for preppilot.db: ask real study questions, check the right book / topic comes back.

Put this next to migrate_to_sqlite.py (in scripts/) and run from C:\\major:
    python scripts/eval_retrieval.py            # summary + the questions that failed
    python scripts/eval_retrieval.py --verbose  # top results for every question

A question PASSES when, within the top K results, a chunk from one of the expected books
contains at least one of the expected keywords (case-insensitive) in its text or section title.
The keyword lists are deliberately loose: a FAIL means "look at this one", not "definitely broken".
Several questions target chapters/appendices that earlier builds were missing (late CLRS chapters,
DS chapters 59-64, C appendices).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.db import get_connection, search_chunks   # same helpers migrate_to_sqlite.py uses

DB_PATH = BASE_DIR / "preppilot.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# (question, expected books, any-of keywords)
QUESTIONS = [
    # CLRS - early chapters
    ("How does insertion sort work?", ("CLRS",), ["insertion sort", "insertion-sort"]),
    ("What is a loop invariant and how is it used to prove correctness?", ("CLRS",), ["loop invariant"]),
    ("How does heapsort build a max-heap?", ("CLRS",), ["max-heap", "heapsort", "build-max-heap"]),
    ("How do hash tables resolve collisions with chaining?", ("CLRS", "DS"), ["chaining", "collision"]),
    ("What are the properties of a red-black tree?", ("CLRS",), ["red-black"]),
    # CLRS - chapters that used to be missing
    ("How does the Knuth-Morris-Pratt string matching algorithm work?", ("CLRS",), ["knuth-morris-pratt", "prefix function"]),
    ("What is the maximum-flow problem and the Ford-Fulkerson method?", ("CLRS",), ["ford-fulkerson", "maximum flow", "max-flow"]),
    ("What does it mean for a problem to be NP-complete?", ("CLRS",), ["np-complete"]),
    ("How does Dijkstra's algorithm find shortest paths?", ("CLRS",), ["dijkstra"]),
    ("What is amortized analysis?", ("CLRS", "DS"), ["amortized"]),
    # DS handbook
    ("What is a binary search tree?", ("DS", "CLRS"), ["binary search tree"]),
    ("What is a splay tree?", ("DS",), ["splay"]),
    ("How does a skip list work?", ("DS",), ["skip list"]),
    ("What is the difference between a stack and a queue?", ("DS",), ["stack", "queue"]),
    ("What is a clustered index in a database?", ("DS",), ["clustered index"]),
    ("How does the Apriori algorithm find frequent itemsets?", ("DS",), ["apriori", "support count", "candidate"]),
    # Let Us C
    ("What does the malloc function do?", ("C", "CPP"), ["malloc"]),   # C++ Complete Reference also covers malloc/free
    ("How does pointer arithmetic work in C?", ("C",), ["pointer"]),
    ("What is the difference between call by value and call by reference?", ("C",), ["call by value", "call by reference"]),
    ("How do you write a switch statement?", ("C",), ["switch"]),
    ("How do I convert a hexadecimal number to binary?", ("C",), ["hexadecimal", "hex", "binary"]),
    # C++ Complete Reference
    ("What is a virtual function?", ("CPP",), ["virtual"]),
    ("How does function overloading work?", ("CPP",), ["overload"]),
    ("What is a copy constructor?", ("CPP",), ["copy constructor"]),
    ("How does exception handling with try and catch work?", ("CPP",), ["try", "catch"]),
    ("How do I read from a file using ifstream?", ("CPP",), ["ifstream", "file"]),
]


def norm(t):
    """Lower-case and ignore line breaks / hyphens, so "binary search\ntree" and "INSERTION-SORT" still match."""
    return " ".join(t.lower().replace("-", " ").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--k", type=int, default=TOP_K)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    conn = get_connection(DB_PATH)

    passed, first_rank, failures = 0, [], []
    for q, books, kws in QUESTIONS:
        vec = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
        res = search_chunks(conn, vec, top_k=args.k)
        rank = None
        for i, r in enumerate(res, 1):
            hay = norm(r.get("text", "") + " " + str(r.get("section_title", "")))
            if r["book_id"] in books and any(norm(k) in hay for k in kws):
                rank = i
                break
        ok = rank is not None
        passed += ok
        if ok:
            first_rank.append(rank)
        else:
            failures.append(q)
        print(f"[{'PASS' if ok else 'FAIL'}] {q}" + (f"  (rank {rank})" if ok else ""))
        if args.verbose or not ok:
            for i, r in enumerate(res[:3], 1):
                snippet = r.get("text", "")[:110].replace("\n", " ")
                print(f"      #{i} {r['book_id']} p{r['page_start']} sec {r.get('section', '')} "
                      f"sim {r['similarity']:.3f} | {snippet}")

    n = len(QUESTIONS)
    print("\n" + "=" * 60)
    print(f"passed {passed}/{n} within top {args.k}   "
          f"(rank 1: {sum(r == 1 for r in first_rank)}, mean rank of hits: "
          f"{(sum(first_rank) / len(first_rank)) if first_rank else 0:.2f})")
    if failures:
        print("failed:")
        for q in failures:
            print("  -", q)
    conn.close()


if __name__ == "__main__":
    main()