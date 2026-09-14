"""
Eyeball a sample of chunks from one book's *_chunks.jsonl before trusting it
enough to feed into build_vector_db.py.

Usage:
    python review_chunks.py DS
    python review_chunks.py CLRS --n 15
    python review_chunks.py CLRS --flag contains_equation   # only show flagged chunks
"""
import argparse
import json
import random
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
CHUNKS_DIR = BASE_DIR / "chunks_semantic"


def load(book_id: str):
    path = CHUNKS_DIR / f"{book_id}_chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No chunk file at {path} — run semantic_chunker.py first.")
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def summarize(chunks):
    flags = ["contains_code", "contains_equation", "contains_example",
             "contains_definition", "contains_algorithm", "contains_table", "contains_figure"]
    counts = Counter()
    for c in chunks:
        for flag in flags:
            if c.get(flag):
                counts[flag] += 1

    print(f"Total chunks: {len(chunks)}")
    for flag in flags:
        pct = 100 * counts[flag] / len(chunks) if chunks else 0
        print(f"  {flag:<22}: {counts[flag]:>5}  ({pct:.1f}%)")

    chapters = Counter(c.get("chapter", "") for c in chunks)
    print(f"\nDistinct chapters seen: {len([k for k in chapters if k])}")
    empty_chapter = chapters.get("", 0)
    if empty_chapter:
        print(f"  WARNING: {empty_chapter} chunks have no chapter assigned "
              f"(structure detection may have missed a heading)")


def show_sample(chunks, n, flag_filter=None, seed=None):
    pool = chunks
    if flag_filter:
        pool = [c for c in chunks if c.get(flag_filter)]
        print(f"\n{len(pool)} chunks match --flag {flag_filter}\n")
        if not pool:
            return

    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))
    sample.sort(key=lambda c: c["chunk_id"])

    for c in sample:
        print("=" * 80)
        print(f"{c['chunk_id']}  |  pages {c['page_start']}-{c['page_end']}  |  {c['word_count']} words")
        print(f"Chapter {c.get('chapter','')}: {c.get('chapter_title','')}  |  "
              f"Section {c.get('section','')} {c.get('section_title','')}")
        flags_on = [k for k in ("contains_code", "contains_equation", "contains_example",
                                 "contains_definition", "contains_algorithm") if c.get(k)]
        if flags_on:
            print(f"Flags: {', '.join(flags_on)}")
        print("-" * 80)
        preview = c["text"][:700]
        print(preview)
        if len(c["text"]) > 700:
            print(f"... [{len(c['text']) - 700} more chars]")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("book_id", help="e.g. DS, C, CPP, CLRS")
    ap.add_argument("--n", type=int, default=10, help="number of chunks to sample")
    ap.add_argument("--flag", default=None,
                    help="only sample chunks with this flag set, e.g. contains_equation")
    ap.add_argument("--seed", type=int, default=None, help="fix the random sample for reproducibility")
    args = ap.parse_args()

    chunks = load(args.book_id.upper())
    print(f"\n### {args.book_id.upper()} ###\n")
    summarize(chunks)
    print()
    show_sample(chunks, args.n, flag_filter=args.flag, seed=args.seed)


if __name__ == "__main__":
    main()