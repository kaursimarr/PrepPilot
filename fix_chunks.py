"""
Clean and re-split chunks_semantic/*.jsonl so they fit all-MiniLM-L6-v2 (256 tokens).

Reads  : chunks_semantic/{DS,C,CPP,CLRS}_chunks.jsonl        (originals are NOT modified)
Writes : chunks_semantic_clean/{BOOK}_chunks.jsonl + fix_report.txt

Fixes applied:
  * ligatures (ﬁ ﬂ ﬃ ...) -> plain letters, control-character junk removed
  * running headers and standalone page numbers removed
  * split drop caps rejoined ("T o" -> "To")
  * overlap between consecutive old chunks removed, then each chunk re-split into
    ~200-token pieces (with a small overlap) on paragraph/sentence boundaries
  * breadcrumbs rebuilt: trailing page numbers stripped (C), truncated titles
    recovered from running headers (CPP), wrong CLRS chapters repaired
  * search_text, word_count, char_count, chunk_id regenerated; page ranges
    re-estimated per piece (approximate, flagged with pages_approx=True)
  * spaced-out running headers ("Ch a p t e r 1 1 : ...") removed
  * page numbers stuck to the start of a piece ("834 Here, T is...") removed
  * empty chapter titles recovered from "Chapter N: Title" lines in the text
  * DS section labels that are table captions or absurdly long are replaced by
    the previous good section of the same chapter
  * C end-of-chapter exercise chunks flagged is_exercise=True (a block ends after 2 pieces in a row with no exercise-like text)
  * chapter/section labels are re-derived for every piece from the numbered headings inside it
    (DS, CLRS), so a piece from section 3.2 is no longer labelled with the section that
    happened to open its 900-word parent chunk
  * chapter titles come from chapter_titles.json (make_chapter_titles.py), then manual overrides
  * reference lists, exercises and near-empty / all-number pieces are moved to
    <BOOK>_excluded.jsonl instead of being indexed (nothing is deleted, see the DROP_* switches)

Usage (from C:\\major):  python fix_chunks.py
Then re-check with:      python verify_chunks.py   (point CHUNKS_DIR at chunks_semantic_clean)
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
if not (BASE / "chunks_semantic").exists() and (BASE.parent / "chunks_semantic").exists():
    BASE = BASE.parent
SRC = BASE / "chunks_semantic"
DST = BASE / "chunks_semantic_clean"
BOOKS = ["DS", "C", "CPP", "CLRS"]

MAX_TOKENS = 230          # model limit is 256 incl. [CLS]/[SEP]; keep headroom
OVERLAP_WORDS = 30        # carry a short trailing sentence into the next piece
MIN_PIECE_WORDS = 25      # merge tiny leftovers into the previous piece

LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
             "\u00a0": " ", "\u00ad": "", "\u200b": ""}
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
CODE_CHARS = re.compile(r"[;{}=<>]")

# "Ch a p t e r 1 1 : A n O v e r v i e w" -- needs real gaps between letters so normal prose never matches
SPACED_CHAPTER = re.compile(
    r"(?<![A-Za-z])C\s?h\s+a\s+p\s+t\s+e\s+r(?:\s\d){1,3}\s?:?(?:\s\S(?=\s|$))*\s*")
LEAD_PAGENO = re.compile(r"^(\d{1,4})\s+(?=[A-Z][a-z])")
CH_HEAD = re.compile(r"(?m)^\s*Chapter\s+(\d+)\s*[:.\-\u2013\u2014]\s*([A-Z][^\n]{2,80}?)\s*$")
BAD_SECTION_TITLE = re.compile(r"(?i)^(table|figure|fig\.?|listing)\b")
MAX_SECTION_TITLE = 90    # DS section titles longer than this are treated as mislabelled
EXERCISE_HEAD = re.compile(r"(?m)^\s*Exercises?\b")
EXERCISE_MARK = re.compile(r"\[[A-H]\]\s+\S")
EXERCISE_Q = re.compile(
    r"(?im)^\s*(?:\(?[a-z]\)|\d{1,2}[.)])\s*(?:write|what|point out|find|attempt|is|are|how|which|why|state|fill|match)\b")
EXERCISE_HARD = re.compile(r"(?i)point out the errors?|attempt the following|answer the following|fill in the blanks")
EXERCISE_SOFT = re.compile(r"(?i)what (?:would|will) be the output|which of the following")
C_CHAPTER_LINE = re.compile(r"(?m)^Chapter \d+:\s*\S")

DROP_EXERCISES = True     # exercises go to <BOOK>_excluded.jsonl instead of the index
DROP_REFERENCES = True    # bibliography pieces go to <BOOK>_excluded.jsonl instead of the index
MAX_EXERCISE_RUN = 120    # safety cap on how many consecutive pieces one exercise block may swallow
MAX_QUIET_PIECES = 2      # an exercise block ends after this many pieces in a row with no exercise-like text
MIN_KEEP_WORDS = 15       # pieces shorter than this (figure axis labels, lone formulas) are excluded
REF_MARK = re.compile(r"(?:^|\s)\[\d{1,3}\]\s+[A-Z][^\[\]\n]{0,40}?[,.]\s")   # "[12] T. Cormen, ..." (author, then comma/period)
OUTLINE_LEADER = re.compile(r"(?:\.\s?){6,}")   # "15.4 The B+-tree ........ 15-10": per-chapter mini table of contents

# Numbered headings such as "3.2.1 List Representation" (DS, CLRS).
HEAD_LINE = re.compile(r"^(\d{1,2})\.(\d{1,2})(?:\.(\d{1,2}))?[ \t]+([A-Z][^\n]{2,100})$")
HEADING_BOOKS = ("DS", "CLRS")
LABEL_KEYS = ("chapter", "chapter_title", "section", "section_title", "subsection", "subsection_title")

# Symbol glyphs the CLRS PDF extracts as control characters. Only mappings actually seen in the
# text are listed (e.g. "2n \x00 1 bound", "n \x02 n matrix"); add more once you have checked them.
BOOK_CTRL_MAP = {"CLRS": {"\x00": "\u2212", "\x02": "\u00d7"}}

TITLES_JSON = BASE / "chapter_titles.json"   # written by make_chapter_titles.py

# Titles the PDF extraction cut off (multi-line chapter titles). Extend this from the book's table of contents.
CHAPTER_TITLE_OVERRIDES = {
    "CPP": {
        "4": "Arrays and Strings",
        "7": "Structures, Unions, Enumerations, and User-Defined Types",
        "13": "Arrays, Pointers, References, and the Dynamic Allocation Operators",
        "14": "Function Overloading, Copy Constructors, and Default Arguments",
        "20": "The C++ I/O System Basics",
        "22": "Run-Time Type ID and the Casting Operators",
        "23": "Namespaces, Conversion Functions, and Other Advanced Topics",
        # Still truncated in the chunks -- fill from the book's contents page (empty string = ignored):
        "26": "",
        "35": "",
        "38": "",
    },
}

CLRS_CHAPTERS = {
    1: "The Role of Algorithms in Computing", 2: "Getting Started", 3: "Growth of Functions",
    4: "Divide-and-Conquer", 5: "Probabilistic Analysis and Randomized Algorithms",
    6: "Heapsort", 7: "Quicksort", 8: "Sorting in Linear Time",
    9: "Medians and Order Statistics", 10: "Elementary Data Structures", 11: "Hash Tables",
    12: "Binary Search Trees", 13: "Red-Black Trees", 14: "Augmenting Data Structures",
    15: "Dynamic Programming", 16: "Greedy Algorithms", 17: "Amortized Analysis",
    18: "B-Trees", 19: "Fibonacci Heaps", 20: "van Emde Boas Trees", 21: "Data Structures for Disjoint Sets",
    22: "Elementary Graph Algorithms", 23: "Minimum Spanning Trees", 24: "Single-Source Shortest Paths",
    25: "All-Pairs Shortest Paths", 26: "Maximum Flow", 27: "Multithreaded Algorithms",
    28: "Matrix Operations", 29: "Linear Programming", 30: "Polynomials and the FFT",
    31: "Number-Theoretic Algorithms", 32: "String Matching", 33: "Computational Geometry",
    34: "NP-Completeness", 35: "Approximation Algorithms",
}   # 3rd edition; chapter_titles.json (from the PDF outline) takes precedence when present


def load_titles():
    """book -> {chapter number (str) -> title}. Precedence: manual overrides > PDF outline json > built-in."""
    titles = {b: {} for b in ("DS", "C", "CPP", "CLRS")}
    titles["CLRS"].update({str(k): v for k, v in CLRS_CHAPTERS.items()})
    if TITLES_JSON.exists():
        try:
            for b, m in json.loads(TITLES_JSON.read_text(encoding="utf-8")).items():
                titles.setdefault(b, {}).update({str(k): v for k, v in m.items() if v})
        except Exception as e:                                   # noqa: BLE001
            print(f"WARNING: could not read {TITLES_JSON}: {e}")
    for b, m in CHAPTER_TITLE_OVERRIDES.items():
        titles.setdefault(b, {}).update({str(k): v for k, v in m.items() if v})
    return titles


TITLES = load_titles()


# --------------------------------------------------------------------------- tokens
def make_token_counter():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        print("Using real MiniLM tokenizer for length checks.")
        return lambda s: len(tok.encode(s, add_special_tokens=False))
    except Exception:
        print("Tokenizer unavailable -> using a conservative estimate (safe, slightly smaller chunks).")
        return lambda s: int(max(len(s.split()) * 1.5, len(s) / 3.6))


# --------------------------------------------------------------------------- cleaning
def normalize_chars(t):
    for k, v in LIGATURES.items():
        t = t.replace(k, v)
    return t


def norm_line(s):
    return re.sub(r"\d+", "#", s.strip())


def is_header_candidate(p):
    p = p.strip()
    return (0 < len(p) <= 90 and "\n" not in p and re.search(r"\d", p)
            and not CODE_CHARS.search(p) and not re.match(r"(?i)(figure|fig\.|table|listing|exercise)", p)
            and re.search(r"[A-Za-z]{3,}", p))


def is_spaced_out(p):
    """Short paragraph that is mostly single letters separated by spaces: 'A n O v e r v i e w'."""
    toks = p.split()
    return (len(toks) >= 8 and len(p) <= 160
            and sum(len(t) == 1 and t.isalpha() for t in toks) / len(toks) >= 0.6)


def strip_lead_pageno(text, stats):
    m = LEAD_PAGENO.match(text)
    if m and not 1900 <= int(m.group(1)) <= 2099:       # keep years
        stats["page_numbers"] += 1
        return text[m.end():]
    return text


def find_headers(chunks):
    """Lines (paragraph-level) whose digit-normalised form repeats in >=3 chunks = running headers."""
    seen = defaultdict(set)
    for i, c in enumerate(chunks):
        for p in re.split(r"\n\s*\n", c["text"]):
            if is_header_candidate(p):
                seen[norm_line(p)].add(i)
    return {k for k, v in seen.items() if len(v) >= 3}


def clean_text(text, headers, raw_headers, stats, title="", ctrl_map=None):
    text = normalize_chars(text)
    for k, v in (ctrl_map or {}).items():                        # known symbol glyphs, e.g. CLRS minus / times
        text = text.replace(k, v)
    n_ctrl = len(CTRL_RE.findall(text))
    if n_ctrl:
        stats["ctrl_chars"] += n_ctrl
        text = CTRL_RE.sub("\ue000", text)          # placeholder, counted per piece then removed
    out, after_junk = [], False
    for p in re.split(r"\n\s*\n", text):
        s = p.strip()
        if not s:
            continue
        if re.fullmatch(r"\d{1,4}", s):                       # standalone page number
            stats["page_numbers"] += 1
            after_junk = True
            continue
        if is_spaced_out(s):                                    # whole paragraph is a spaced-out header
            stats["headers"] += 1
            after_junk = True
            continue
        s2 = SPACED_CHAPTER.sub("", s)                          # spaced-out "Chapter N: Title" glued into text
        if s2 != s:
            stats["headers"] += 1
            after_junk = True
            s = s2.strip()
            if not s:
                continue
        if is_header_candidate(s) and norm_line(s) in headers:  # running header paragraph
            stats["headers"] += 1
            raw_headers.add(s)
            after_junk = True
            continue
        for h in raw_headers:                                   # header glued to text start
            if s.startswith(h) and len(s) > len(h) + 1 and s[len(h)] in " \n\t":
                s = s[len(h):].lstrip()
                stats["headers"] += 1
                after_junk = True
                break
        if after_junk:                                          # "81 Expanding out..." page no. stuck to text
            s = re.sub(r"^\d{1,4}\s+(?=[A-Z])", "", s)
            after_junk = False
        m = re.match(r"Chapter \d+\s+", s)                     # stray "Chapter N " glued in front of body text
        if m and not s[m.end():].startswith(title[:15] or "\0"):
            s = s[m.end():]
            stats["headers"] += 1
        out.append(s)
    text = "\n\n".join(out)
    text = strip_lead_pageno(text, stats)                       # page number stuck to first word of chunk
    # drop caps: "T o understand" -> "To understand", "T he" -> "The"
    text = re.sub(r"(?m)^([A-Z]) ([a-z]) ", r"\1\2 ", text)
    text = re.sub(r"(?m)^([A-Z]) (he|his|hat|hese|hose|here|hen|hey)\b", r"\1\2", text)
    return text


# --------------------------------------------------------------------------- de-overlap
def strip_overlap(prev_text, cur_text):
    pw, spans = prev_text.split(), [m.end() for m in re.finditer(r"\S+", cur_text)]
    cw = cur_text.split()
    for k in range(min(300, len(pw), len(cw)), 7, -1):
        if pw[-k:] == cw[:k]:
            return cur_text[spans[k - 1]:].lstrip(), k
    return cur_text, 0


# --------------------------------------------------------------------------- splitting
def split_unit(u, budget, count):
    """Break an oversized paragraph on sentences/lines, then on words. Returns [(text, sep_before)]."""
    bits = re.split(r"((?<=[.!?])[ \t]+|\n)", u)
    parts, sep = [], "\n\n"
    for i in range(0, len(bits), 2):
        if bits[i].strip():
            parts.append((bits[i], sep))
        sep = "\n" if i + 1 < len(bits) and "\n" in bits[i + 1] else " "
    res = []
    for s, sp in parts:
        if count(s) <= budget:
            res.append((s, sp))
        else:
            cur = []
            for word in s.split():
                cur.append(word)
                if count(" ".join(cur)) > budget:
                    res.append((" ".join(cur[:-1]), sp if not res or sp != "\n\n" else " "))
                    cur, sp = [word], " "
            if cur:
                res.append((" ".join(cur), sp))
    if res:
        res[0] = (res[0][0], "\n\n")
    return res


def split_chunk(text, budget, count):
    units = []                                   # (text, separator placed before it)
    for p in re.split(r"\n\s*\n", text):
        if not p.strip():
            continue
        units.extend([(p.strip(), "\n\n")] if count(p) <= budget else split_unit(p, budget, count))
    if not units:
        return []
    toks = [count(u) for u, _ in units]
    words = [len(u.split()) for u, _ in units]
    cum = [0]
    for w in words:
        cum.append(cum[-1] + w)

    def join(a, b):
        out = units[a][0]
        for k in range(a + 1, b):
            out += units[k][1] + units[k][0]
        return out

    pieces, i = [], 0
    while i < len(units):
        j, t = i, 0
        while j < len(units) and (t + toks[j] <= budget or j == i):
            t += toks[j]
            j += 1
        pieces.append([i, j])
        if j >= len(units):
            break
        last = j - 1
        carry = last > i and words[last] <= OVERLAP_WORDS and not CODE_CHARS.search(units[last][0])
        i = last if carry else j
    if len(pieces) > 1 and (cum[pieces[-1][1]] - cum[pieces[-1][0]]) < MIN_PIECE_WORDS:
        a, b = pieces[-2], pieces[-1]
        if sum(toks[a[0]:b[1]]) <= budget:
            pieces[-2] = [a[0], b[1]]
            pieces.pop()
    return [(join(a, b), cum[a], cum[b], cum[-1]) for a, b in pieces]


# --------------------------------------------------------------------------- metadata
def recover_chapter_titles(chunks):
    """Fill empty chapter_title from the most common 'Chapter N: Title' line seen for that chapter."""
    votes = defaultdict(Counter)
    for c in chunks:
        for m in CH_HEAD.finditer(c["text"]):
            votes[m.group(1)][m.group(2).strip()] += 1
    titles = {ch: v.most_common(1)[0][0] for ch, v in votes.items()}
    filled = 0
    for c in chunks:
        if not (c.get("chapter_title") or "").strip():
            t = titles.get(str(c.get("chapter")))
            if t:
                c["chapter_title"] = t
                filled += 1
    return filled


def fix_ds_sections(chunks):
    """Table captions and overlong titles are not sections: fall back to the last good section of the chapter."""
    last_good, fixed = {}, 0
    for c in chunks:
        ch, title = c.get("chapter"), (c.get("section_title") or "").strip()
        if not c.get("section"):
            continue
        if BAD_SECTION_TITLE.match(title) or len(title) > MAX_SECTION_TITLE:
            c["section"], c["section_title"] = last_good.get(ch, ("", ""))
            fixed += 1
        else:
            last_good[ch] = (c["section"], title)
    return fixed


def build_breadcrumb(d):
    parts = [d.get("book_title") or d["book_id"]]
    ch, ct = d.get("chapter"), d.get("chapter_title")
    if ch:
        parts.append(f"Chapter {ch}: {ct}" if ct else f"Chapter {ch}")
    if d.get("section"):
        parts.append(f"Section {d['section']}: {d.get('section_title', '')}".rstrip(": "))
    if d.get("subsection"):
        parts.append(f"Subsection {d['subsection']}: {d.get('subsection_title', '')}".rstrip(": "))
    return " > ".join(parts)


def trim_reference_tail(text, stats):
    """A piece that ends the chapter: keep the prose, cut the bibliography that follows it."""
    text = re.sub(r"\n\s*(?:References|Bibliography)\s*$", "", text)     # heading left dangling at the end of a piece
    h = re.search(r"\bReferences\s+\[\d{1,3}\]", text)
    if h and h.start() > 0.35 * len(text):                  # "... References [1] ..." after real prose
        stats["ref_tail_trimmed"] += 1
        return text[:h.start()].rstrip()
    m = REF_MARK.search(text)
    if not m or m.start() <= 0.35 * len(text):              # (a piece that is mostly references is excluded later)
        return text
    tail = text[m.start():]
    marks = len(REF_MARK.findall(tail))
    if marks >= 3 and marks / max(len(tail.split()), 1) * 100 >= 2.0:
        stats["ref_tail_trimmed"] += 1
        return re.sub(r"\s*References\s*$", "", text[:m.start()]).rstrip()
    return text


def fix_bad_chapters(chunks):
    """A figure label such as '0.3 (a)' was read as a heading and made 'Chapter 0'. Inherit the previous labels."""
    last, fixed = None, 0
    for c in chunks:
        ch = str(c.get("chapter") or "")
        if ch in ("", "0") and last:
            for k in LABEL_KEYS:
                c[k] = last[k]
            fixed += 1
        elif ch not in ("", "0"):
            last = {k: c.get(k, "") for k in LABEL_KEYS}
    return fixed


def piece_headings(text):
    """Numbered headings inside a piece: [(offset, chapter, section, subsection, title)]."""
    out, pos = [], 0
    for line in text.split("\n"):
        s = line.strip()
        m = HEAD_LINE.match(s)
        if (m and int(m.group(1)) >= 1 and len(s.split()) <= 16 and not CODE_CHARS.search(s)
                and not s.endswith((".", ",", ";", ":"))):
            out.append((pos, m.group(1), m.group(2), m.group(3) or "", m.group(4).strip()))
        pos += len(line) + 1
    return out


def accept_heading(h, lab):
    """Headings must move forward (same chapter, later section, or the next chapter's first sections)."""
    _, ch, sec, _sub, _title = h
    try:
        cur_ch = int(lab.get("chapter") or 0)
        cur_sec = int(str(lab.get("section") or "0").split(".")[-1] or 0)
    except ValueError:
        return True
    ch_i, sec_i = int(ch), int(sec)
    if ch_i == cur_ch:
        return cur_sec <= sec_i <= cur_sec + 12
    return ch_i == cur_ch + 1 and sec_i <= 2


def apply_heading(book, lab, h):
    _, ch, sec, sub, title = h
    new = dict(lab)
    if ch != str(lab.get("chapter") or ""):
        new["chapter"], new["chapter_title"] = ch, TITLES.get(book, {}).get(ch, "")
    section = f"{ch}.{sec}"
    if sub:
        if section != lab.get("section"):
            new["section_title"] = ""
        new["section"] = section
        new["subsection"], new["subsection_title"] = f"{section}.{sub}", title
    else:
        new["section"], new["section_title"] = section, title
        new["subsection"] = new["subsection_title"] = ""
    return new


def relabel_piece(book, text, label):
    """(label the piece belongs to, label in force after the piece) using the headings inside the piece."""
    cur, first = dict(label), None
    for h in piece_headings(text):
        if accept_heading(h, cur):
            if first is None:
                first = h
            cur = apply_heading(book, cur, h)
    if first is not None and first[0] <= 120:            # heading opens the piece -> piece belongs to it
        return apply_heading(book, dict(label), first), cur
    return dict(label), cur


def fit_breadcrumb(parent, lab, text, count, limit=250):
    """Breadcrumb for this piece; shortened if the breadcrumb + text would exceed the model's 256 tokens."""
    d = dict(parent)
    d.update(lab)
    for drop in (0, 1, 2):
        if drop >= 1:
            d["subsection"] = d["subsection_title"] = ""
        if drop >= 2:
            d["section_title"] = ""
        bc = build_breadcrumb(d)
        st = f"[{bc}]\n\n{text}"
        if count(st) <= limit:
            break
    return d, bc, st


def fix_metadata(book, chunks, raw_headers):
    notes = []
    filled = recover_chapter_titles(chunks)
    if filled:
        notes.append(f"  chapter titles recovered from text: {filled} chunks")
    if book == "C":
        for c in chunks:
            c["chapter_title"] = re.sub(r"\s+\d+$", "", c.get("chapter_title", "")).strip()

    if book == "DS":
        notes.append(f"  DS chunks with bogus chapter 0 relabelled from the previous chunk: {fix_bad_chapters(chunks)}")
        notes.append(f"  DS section labels reset (table caption / overlong title): {fix_ds_sections(chunks)}")

    for c in chunks:                        # titles: overrides > PDF outline json > built-in > what the chunker found
        t = TITLES.get(book, {}).get(str(c.get("chapter")))
        if t:
            c["chapter_title"] = t

    if book == "CLRS":                      # original section/chapter labels are unreliable: rebuild from headings
        head = re.compile(r"(?m)^(\d+)\.(\d+)[ \t]+([A-Z][^\n]{2,80}?)(?:[ \t]{2,}|\n|$)")
        cur = None                          # (chapter, section, title)
        for c in chunks:
            heads = []
            for m in head.finditer(c["text"]):
                ch, sec, title = int(m.group(1)), int(m.group(2)), m.group(3).strip()
                if CODE_CHARS.search(title):
                    continue
                ref = heads[-1] if heads else cur
                if ref is None:
                    ok = ch == 1
                else:
                    ok = (ch, sec) >= (ref[0], ref[1]) and ch <= ref[0] + 1
                if ok:
                    heads.append((ch, sec, title, m.start()))
            if heads and heads[0][3] <= 250:
                use = heads[0][:3]
            else:
                use = cur or (heads[0][:3] if heads else (1, 0, ""))
            c["chapter"], c["section"] = str(use[0]), (f"{use[0]}.{use[1]}" if use[1] else "")
            c["section_title"] = use[2] if use[1] else ""
            c["chapter_title"] = TITLES["CLRS"].get(str(use[0]), "")
            if heads:
                cur = heads[-1][:3]
            elif cur is None:
                cur = use

    for c in chunks:                        # rebuild breadcrumb consistently
        c["breadcrumb"] = build_breadcrumb(c)
    return notes


# --------------------------------------------------------------------------- main
def process_book(book, count, report):
    path = SRC / f"{book}_chunks.jsonl"
    if not path.exists():
        report.append(f"{book}: MISSING {path}")
        return
    chunks = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    stats = Counter()
    headers = find_headers(chunks)
    raw_headers = set()
    ctrl_map = BOOK_CTRL_MAP.get(book)
    for c in chunks:                        # first pass registers raw header strings
        clean_text(c["text"], headers, raw_headers, Counter(), ctrl_map=ctrl_map)
    notes = fix_metadata(book, chunks, raw_headers)

    out, excluded, dropped_pages, prev_clean = [], [], set(), ""
    ex_run, run_len, quiet, last_chapter = False, 0, 0, None   # C: an exercise block runs on while pieces still look like exercises
    for c in chunks:
        cleaned = clean_text(c["text"], headers, raw_headers, stats, c.get("chapter_title", ""), ctrl_map)
        body, k = strip_overlap(prev_clean, cleaned) if prev_clean else (cleaned, 0)
        stats["overlap_words_removed"] += k
        body = strip_lead_pageno(body, stats)               # overlap removal can leave "834 Here..." at the start
        prev_clean = cleaned
        if not body.strip():
            continue
        bc_tokens = count(f"[{c['breadcrumb']}]\n\n")
        ps, pe = c["page_start"], c.get("page_end", c["page_start"])
        span = pe - ps + 1
        label = {k: c.get(k, "") for k in LABEL_KEYS}       # label in force at the start of this parent chunk
        for text, w0, w1, wt in split_chunk(body, MAX_TOKENS - bc_tokens - 12, count):
            wt = max(wt, 1)
            new = {k2: v for k2, v in c.items() if k2 not in ("text", "search_text", "chunk_id")}
            new["parent_chunk_id"] = c["chunk_id"]
            new["page_start"] = min(pe, ps + int(w0 / wt * span))
            new["page_end"] = min(pe, ps + int(max(w1 - 1, w0) / wt * span))
            new["pages_approx"] = span > 1
            garbled = text.count("\ue000") >= 3
            text = re.sub(r" {2,}", " ", text.replace("\ue000", ""))
            text = strip_lead_pageno(text, stats)               # splitting can also expose a page number
            text = trim_reference_tail(text, stats)             # prose followed by the chapter's reference list
            piece_label = label
            if book in HEADING_BOOKS:
                piece_label, label = relabel_piece(book, text, label)
                if piece_label != {k: c.get(k, "") for k in LABEL_KEYS}:
                    stats["relabelled"] += 1
            d, bc, search_text = fit_breadcrumb(c, piece_label, text, count)
            new.update({k: d.get(k, "") for k in LABEL_KEYS})
            new["breadcrumb"] = bc
            is_ex = False
            if book == "C":
                if C_CHAPTER_LINE.search(text) or new["chapter"] != last_chapter:
                    ex_run, quiet = False, 0                        # a new chapter starts here
                last_chapter = new["chapter"]
                strong = bool(EXERCISE_HEAD.search(text[:200]) or len(EXERCISE_MARK.findall(text)) >= 2
                              or EXERCISE_HARD.search(text))
                soft = len(EXERCISE_Q.findall(text)) >= 5 or len(EXERCISE_SOFT.findall(text)) >= 2
                if strong and not ex_run:
                    ex_run, run_len = True, 0                       # exercises run to the end of the chapter...
                    stats["exercise_runs"] += 1
                if ex_run:
                    run_len += 1
                    has_signal = bool(strong or soft or EXERCISE_MARK.search(text) or EXERCISE_Q.search(text)
                                      or EXERCISE_SOFT.search(text))
                    quiet = 0 if has_signal else quiet + 1
                    # the block ends when the text stops looking like exercises (appendices, tables and
                    # reference material that follow the last chapter's exercises stay in the index)
                    if quiet > MAX_QUIET_PIECES or run_len > MAX_EXERCISE_RUN:
                        ex_run = False
                is_ex = ex_run or soft
            if is_ex:
                stats["exercise_chunks"] += 1
            new["text"] = text
            new["search_text"] = search_text
            new["word_count"] = len(text.split())
            new["char_count"] = len(text)
            new["contains_figure"] = bool(re.search(r"\b(Figure|FIGURE|Fig\.)\s*\d", text))
            new["contains_code"] = bool(re.search(r"[;{}]\s*$|#include|\bint\s+main\b", text, re.M))
            new["garbled_math"] = bool(garbled)
            new["is_exercise"] = is_ex
            toks_w = text.split()
            n_words = len(toks_w)
            ref_marks = len(REF_MARK.findall(text))
            is_ref = ref_marks >= 3 and ref_marks / max(n_words, 1) * 100 >= 2.0
            numeric = sum(bool(re.fullmatch(r"[\d.,()%+\-\u2212]+", t)) for t in toks_w) / max(n_words, 1)
            new["is_reference"] = is_ref
            n_leaders = sum(bool(OUTLINE_LEADER.search(ln)) for ln in text.split("\n"))
            reason = ("reference_list" if is_ref else "exercise" if is_ex
                      else "outline" if n_leaders >= 3
                      else "tiny_or_numeric" if (n_words < MIN_KEEP_WORDS or (numeric > 0.5 and n_words < 80)) else "")
            if reason and (reason in ("tiny_or_numeric", "outline") or (reason == "reference_list" and DROP_REFERENCES)
                           or (reason == "exercise" and DROP_EXERCISES)):
                stats["excluded_" + reason] += 1
                excluded.append(dict(new, exclude_reason=reason))
                dropped_pages.update(range(new["page_start"], new["page_end"] + 1))
                continue
            out.append(new)

    for i, r in enumerate(out, 1):
        r["chunk_id"] = f"{book}_CHUNK_{i:05d}"
    DST.mkdir(exist_ok=True)
    with open(DST / f"{book}_chunks.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(DST / f"{book}_excluded.jsonl", "w", encoding="utf-8") as f:
        for r in excluded:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    toks = sorted(count(r["search_text"]) for r in out)
    gaps = []
    covered = sorted({p for r in out for p in range(r["page_start"], r["page_end"] + 1)} | dropped_pages)
    for a, b in zip(covered, covered[1:]):
        if b - a > 3:
            gaps.append((a, b))
    report += [
        f"\n{book}: {len(chunks)} old chunks -> {len(out)} new chunks",
        f"  tokens median/max: {toks[len(toks)//2]}/{toks[-1]}   over {MAX_TOKENS}: {sum(t > MAX_TOKENS for t in toks)}",
        f"  removed: {stats['headers']} running headers, {stats['page_numbers']} page numbers, "
        f"{stats['overlap_words_removed']} duplicated overlap words, {stats['ctrl_chars']} control chars",
        f"  chunks flagged garbled_math: {sum(r['garbled_math'] for r in out)}",
        f"  excluded from the index (see {book}_excluded.jsonl): {stats['excluded_reference_list']} reference lists, "
        f"{stats['excluded_exercise']} exercises, {stats['excluded_outline']} chapter outlines, "
        f"{stats['excluded_tiny_or_numeric']} tiny/all-number pieces",
        f"  exercise blocks started (C): {stats['exercise_runs']}",
        f"  pieces whose chapter/section label was re-derived from a heading inside them: {stats['relabelled']}",
        f"  reference-list tails trimmed off prose pieces: {stats['ref_tail_trimmed']}",
        f"  chunks still starting with a number + capital word: "
        f"{sum(bool(LEAD_PAGENO.match(r['text'])) for r in out)}",
        f"  chunks still containing a spaced-out 'Ch a p t e r': "
        f"{sum(bool(SPACED_CHAPTER.search(r['text'])) for r in out)}",
        f"  page gaps > 3 pages (excluded pages count as covered): {len(gaps)} {gaps[:6]}",
        f"  distinct breadcrumbs: {len({r['breadcrumb'] for r in out})}",
    ]
    report += notes
    empty = sorted({str(r["chapter"]) for r in out if r.get("chapter") and not r.get("chapter_title")},
                   key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x))
    if empty:
        shown = ", ".join(empty[:15]) + (" ..." if len(empty) > 15 else "")
        report.append(f"  chapters still without a title ({len(empty)}): {shown}")


if __name__ == "__main__":
    count = make_token_counter()
    report = []
    for b in BOOKS:
        process_book(b, count, report)
    DST.mkdir(exist_ok=True)
    (DST / "fix_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nWrote cleaned files to {DST}")