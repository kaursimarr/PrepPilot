"""
Verify chunks_semantic/*.jsonl are RAG-ready for all-MiniLM-L6-v2.

Usage (from C:\\major):
    python verify_chunks.py
    python verify_chunks.py --book CLRS --samples 5
"""
import argparse, json, re, statistics, random
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# Works whether this file sits in C:\\major or C:\\major\\scripts
BASE_DIR = SCRIPT_DIR if (SCRIPT_DIR / "chunks_semantic").exists() or (SCRIPT_DIR / "chunks_semantic_clean").exists() else SCRIPT_DIR.parent
CHUNKS_DIR = BASE_DIR / ("chunks_semantic_clean" if (BASE_DIR / "chunks_semantic_clean").exists() else "chunks_semantic")
BOOKS = ["DS", "C", "CPP", "CLRS"]
MAX_TOKENS = 256          # all-MiniLM-L6-v2 truncates input beyond this
MIN_WORDS = 30            # below this a chunk carries little meaning
REQUIRED = ["chunk_id", "book_id", "text", "page_start"]


def get_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        return lambda s: len(tok.encode(s, add_special_tokens=True, truncation=False))
    except Exception:
        return lambda s: int(len(s.split()) * 1.4)  # rough estimate


def pct(n, total):
    return f"{n} ({100 * n / max(total, 1):.1f}%)"


def check_book(book, count_tokens, samples):
    path = CHUNKS_DIR / f"{book}_chunks.jsonl"
    print(f"\n{'=' * 70}\n{book}: {path}\n{'=' * 70}")
    if not path.exists():
        print("  MISSING FILE")
        return

    chunks, bad_json = [], 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                bad_json += 1
                print(f"  line {i}: invalid JSON")

    n = len(chunks)
    print(f"  chunks: {n}   invalid JSON lines: {bad_json}")
    if not n:
        return

    missing = Counter()
    empty_text, short, long_, no_bc, bad_pages, wrong_book = [], [], [], [], [], []
    noisy, tok_counts, ids, texts = [], [], Counter(), Counter()

    for c in chunks:
        for k in REQUIRED:
            if k not in c or c[k] in (None, ""):
                missing[k] += 1
        cid = c.get("chunk_id", "?")
        ids[cid] += 1
        text = (c.get("text") or "").strip()
        texts[text] += 1
        if not text:
            empty_text.append(cid)
            continue
        if str(c.get("book_id", "")).upper() != book:
            wrong_book.append(cid)
        words = len(text.split())
        if words < MIN_WORDS:
            short.append(cid)
        search_text = c.get("search_text") or f"[{c.get('breadcrumb', '')}]\n\n{text}"
        t = count_tokens(search_text)
        tok_counts.append(t)
        if t > MAX_TOKENS:
            long_.append(cid)
        if not (c.get("breadcrumb") or c.get("chapter") or c.get("section")):
            no_bc.append(cid)
        ps, pe = c.get("page_start"), c.get("page_end", c.get("page_start"))
        if not isinstance(ps, int) or not isinstance(pe, int) or ps < 1 or pe < ps:
            bad_pages.append(cid)
        alpha = sum(ch.isalpha() for ch in text) / max(len(text), 1)
        is_code = re.search(r"[;{}]|#include|==|->", text)
        if (alpha < 0.5 and not is_code) or re.search(r"(.)\1{9,}", text) or "\ufffd" in text:
            noisy.append(cid)

    dup_ids = [k for k, v in ids.items() if v > 1]
    dup_text = sum(v - 1 for t, v in texts.items() if v > 1 and t)

    print("\n  Structure")
    print(f"    missing required fields : {dict(missing) or 'none'}")
    print(f"    empty text              : {len(empty_text)}")
    print(f"    duplicate chunk_ids     : {len(dup_ids)}")
    print(f"    duplicate texts         : {dup_text}")
    print(f"    book_id mismatch        : {len(wrong_book)}")
    print(f"    invalid page ranges     : {len(bad_pages)}")

    print("\n  Retrieval quality")
    print(f"    too short (<{MIN_WORDS} words)  : {pct(len(short), n)}")
    print(f"    over {MAX_TOKENS} tokens (truncated by model): {pct(len(long_), n)}")
    print(f"    no breadcrumb/chapter/section: {pct(len(no_bc), n)}")
    print(f"    noisy/OCR-garbage text  : {pct(len(noisy), n)}")
    if tok_counts:
        print(f"    tokens min/median/mean/max: {min(tok_counts)}/"
              f"{int(statistics.median(tok_counts))}/{int(statistics.mean(tok_counts))}/{max(tok_counts)}")

    with_fig = sum(1 for c in chunks if c.get("contains_figure"))
    print(f"    flagged contains_figure : {with_fig}")

    # Verdict
    problems = []
    if bad_json or empty_text or dup_ids or missing or bad_pages or wrong_book:
        problems.append("structural errors")
    if len(long_) / n > 0.10:
        problems.append("many chunks exceed 256 tokens (tail text is never embedded)")
    if len(short) / n > 0.10:
        problems.append("many tiny chunks")
    if len(noisy) / n > 0.02:
        problems.append("noisy text")
    print("\n  VERDICT:", "READY" if not problems else "NEEDS WORK -> " + "; ".join(problems))

    for label, lst in [("too long", long_), ("too short", short), ("noisy", noisy)]:
        if lst:
            print(f"    examples {label}: {lst[:5]}")

    print(f"\n  Random samples:")
    for c in random.sample(chunks, min(samples, n)):
        preview = (c.get("text") or "")[:300].replace("\n", " ")
        print(f"   [{c.get('chunk_id')}] p{c.get('page_start')}-{c.get('page_end')} "
              f"| {c.get('breadcrumb', '')[:80]}\n      {preview}...\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", choices=BOOKS)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--dir", default=None,
                    help="folder name or full path (default: chunks_semantic_clean if it exists, else chunks_semantic)")
    a = ap.parse_args()
    if a.dir:
        d = Path(a.dir)
        CHUNKS_DIR = d if d.is_absolute() else BASE_DIR / d
    print(f"Checking: {CHUNKS_DIR}")
    counter = get_tokenizer()
    for b in ([a.book] if a.book else BOOKS):
        check_book(b, counter, a.samples)