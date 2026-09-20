from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter
import json
import re
import statistics


# ============================================================
# CONFIG
# ============================================================

BASE = Path(__file__).resolve().parent.parent

TARGET = 800
MIN_WORDS = 250
MAX_WORDS = 1200
HARD_MAX = 1800
OVERLAP = 0.10

REBUILD_DS = True

# Front matter / TOC heuristics are only applied right at the content start. Applying them to
# every chunk dropped real prose (20-32% of DS/C/CPP); a "back of book" zone was tried too, but
# all four books have real content (chapters 59-64, appendices) in their last 8% of pages.
# The back-of-book index is handled by the "Index" marker in structure(), and bibliographies
# are handled in fix_chunks.py.
FRONT_ZONE_PAGES = 4       # chunks starting within this many pages of the content start
BACK_ZONE_FRAC = None      # None = never filter by position at the end of the book

BOOKS = {
    "DS": {
        "title": "Handbook of Data Structures and Applications",
        "input": BASE / "extracted" / "DS_raw.txt",
        "output": BASE / "chunks_semantic" / "DS_chunks.jsonl",
        "start": 20,
        "kind": "ds",
    },
    "C": {
        "title": "Let Us C",
        "input": BASE / "extracted" / "C_raw.txt",
        "output": BASE / "chunks_semantic" / "C_chunks.jsonl",
        "start": 16,
        "kind": "c",
    },
    "CPP": {
        "title": "C++: The Complete Reference",
        "input": BASE / "extracted" / "CPP_raw.txt",
        "output": BASE / "chunks_semantic" / "CPP_chunks.jsonl",
        "start": 37,
        "kind": "cpp",
    },
    "CLRS": {
        "title": "Introduction to Algorithms",
        "input": BASE / "extracted" / "APA_raw.txt",
        "output": BASE / "chunks_semantic" / "CLRS_chunks.jsonl",
        "start": 26,
        "kind": "clrs",
    },
}

PAGE_RE = re.compile(r"={3,}\s*PAGE\s+(\d+)\s*={3,}", re.I)
SEC_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})$")
SUB_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{1,2})$")
NUM_RE = re.compile(r"^\d{1,2}$")
C_CH_RE = re.compile(r"^Chapter\s+(\d{1,3})\s*:\s*(.+)$", re.I)
CPP_CH_RE = re.compile(r"^Chapter\s+(\d{1,3})$", re.I)

INDEX_START_RE = re.compile(
    r"^(?:Index|INDEX|Subject Index|Index of|General Index)\s*$",
    re.I
)
TOC_RE = re.compile(
    r"^(?:Contents|Table of Contents|CONTENTS)\s*$",
    re.I
)
COPYRIGHT_RE = re.compile(
    r"^(?:Copyright|©|All rights reserved|Published by|ISBN)\b",
    re.I
)

WORD_RE = re.compile(r"\S+")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")

PRE_RE = re.compile(r"^\s*#\s*(include|define|ifdef|ifndef|endif|pragma|undef)\b")
BRACE_RE = re.compile(r"^\s*[{}]\s*;?\s*$")
SEMI_RE = re.compile(r";\s*$")
CTRL_RE = re.compile(
    r"^\s*(if|for|while|switch|else|do|return|break|continue|case|default)"
    r"\s*(\(|\{|\b)"
)
OP_RE = re.compile(r"::|->")
ASSIGN_RE = re.compile(
    r"^\s*[A-Za-z_]\w*(?:\[[^\]]+\])?\s*"
    r"(?:=|\+=|-=|\*=|/=|%=)\s*.+[;]?\s*$"
)
COMMENT_RE = re.compile(r"^\s*(//|/\*|\*|\*/)")
FUNC_RE = re.compile(
    r"^\s*(?:[A-Za-z_][\w:<>,\[\]\s*&]*\s+)?"
    r"[A-Za-z_]\w*\s*\([^;{}]*\)\s*(?:\{|$)"
)

MATH_RE = re.compile(r"[∑∏√±∞≤≥≠≈ΘΩλμσ∈∫]")
MATH2_RE = re.compile(
    r"\b(?:O|o|Ω|Θ|T|t)\s*\([^)]*\)|\bn\^\s*\d+|\blog\s*n\b|\blim\b"
)

TABLE_RE = re.compile(r"^\s*(?:TABLE|Table)\s+\d+(?:\.\d+)*\b")
FIG_RE = re.compile(r"^\s*(?:FIGURE|Figure|FIG)\s+\d+(?:\.\d+)*\b")


@dataclass
class Page:
    number: int
    lines: list[str]


@dataclass
class Block:
    kind: str
    text: str
    start: int
    end: int

    @property
    def words(self):
        return len(WORD_RE.findall(self.text))


@dataclass
class Segment:
    chapter: str = ""
    chapter_title: str = ""
    section: str = ""
    section_title: str = ""
    subsection: str = ""
    subsection_title: str = ""
    heading: str = ""
    heading_page: int | None = None
    lines: list[tuple[str, int]] = field(default_factory=list)


def words(text):
    return len(WORD_RE.findall(text))


def parse_pages(text):
    hits = list(PAGE_RE.finditer(text))
    if not hits:
        return [Page(1, text.splitlines())]

    pages = []
    for i, hit in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        pages.append(Page(
            int(hit.group(1)),
            text[hit.end():end].splitlines()
        ))
    return pages


def clean_pages(pages):
    # Only repeated boundary lines are considered running headers.
    counts = Counter()

    for p in pages:
        lines = [x.strip() for x in p.lines if x.strip()]
        for x in set(lines[:5] + lines[-5:]):
            if 5 <= len(x) <= 100:
                counts[x] += 1

    threshold = max(6, int(max(1, len(pages)) * 0.35))
    repeated = {x for x, n in counts.items() if n >= threshold}

    for p in pages:
        cleaned = []

        for raw in p.lines:
            s = raw.strip()

            if not s:
                cleaned.append("")
                continue

            # Repeated running headers / footers.
            if s in repeated:
                continue

            # Page labels such as "12-13".
            if re.fullmatch(r"\d{1,4}\s*-\s*\d{1,4}", s):
                continue

            # Generic legal / publisher boilerplate.
            if COPYRIGHT_RE.match(s):
                continue
            if re.search(r"click here for terms of use", s, re.I):
                continue
            if re.search(r"all rights reserved", s, re.I):
                continue

            indent = len(raw) - len(raw.lstrip(" "))
            s = re.sub(r"[ \t]+", " ", s)
            cleaned.append(" " * min(indent, 8) + s)

        p.lines = heal_hyphenated_lines(cleaned)

    return pages


def heal_hyphenated_lines(lines: list[str]) -> list[str]:
    """Reconnect words broken across line wraps by soft hyphens in PDFs."""
    healed = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.rstrip()
        if (
            i + 1 < n
            and stripped.endswith("-")
            and re.search(r"[A-Za-z]-$", stripped)
        ):
            next_line = lines[i + 1]
            next_stripped = next_line.lstrip()
            m = re.match(r"^([a-z]+)(\b.*)$", next_stripped)
            if m:
                first_part = stripped[:-1]
                second_part = m.group(1)
                rest = m.group(2)
                healed_line = first_part + second_part
                if rest.strip():
                    healed_line += " " + rest.strip()
                healed.append(healed_line)
                i += 2
                continue
        healed.append(line)
        i += 1
    return healed


def title_like(s):
    s = s.strip()
    if not s or len(s) > 110:
        return False
    if s.endswith((".", ";", ":")):
        return False
    if s[0].islower():
        return False
    if not re.search(r"[A-Za-z]", s):
        return False
    if PRE_RE.match(s) or SEMI_RE.search(s) or BRACE_RE.fullmatch(s):
        return False
    return True


def next_nonempty(flat, i):
    j = i + 1
    while j < len(flat):
        if flat[j][0].strip():
            return (*flat[j], j)
        j += 1
    return None, None, None


def heading_ds(flat, i):
    s, page = flat[i][0].strip(), flat[i][1]
    m = SUB_RE.fullmatch(s)
    if m:
        title, _, j = next_nonempty(flat, i)
        if title and title_like(title):
            a, b, c = m.groups()
            return ("sub", a, f"{a}.{b}", f"{a}.{b}.{c}",
                    title.strip(), page, j + 1)

    m = SEC_RE.fullmatch(s)
    if m:
        title, _, j = next_nonempty(flat, i)
        if title and title_like(title):
            a, b = m.groups()
            return ("sec", a, f"{a}.{b}", "",
                    title.strip(), page, j + 1)
    return None


def heading_clrs(flat, i):
    h = heading_ds(flat, i)
    if h:
        return h

    s, page = flat[i][0].strip(), flat[i][1]
    if NUM_RE.fullmatch(s):
        n = int(s)
        if 1 <= n <= 40:
            title, _, j = next_nonempty(flat, i)
            if title and title_like(title):
                if not (
                    NUM_RE.fullmatch(title.strip())
                    or SEC_RE.fullmatch(title.strip())
                    or SUB_RE.fullmatch(title.strip())
                ):
                    return ("chapter", str(n), "", "", title.strip(), page, j + 1)
    return None


def heading_c(flat, i):
    s, page = flat[i][0].strip(), flat[i][1]
    m = C_CH_RE.fullmatch(s)
    if m:
        return ("chapter", m.group(1), "", "", m.group(2).strip(), page, i + 1)
    return None


def heading_cpp(flat, i):
    s, page = flat[i][0].strip(), flat[i][1]
    m = CPP_CH_RE.fullmatch(s)
    if m:
        title, _, j = next_nonempty(flat, i)
        if title and title_like(title):
            return ("chapter", m.group(1), "", "", title.strip(), page, j + 1)
    return None


def code_line(s):
    s = s.strip()
    if not s:
        return False
    return bool(
        PRE_RE.match(s)
        or BRACE_RE.fullmatch(s)
        or SEMI_RE.search(s)
        or CTRL_RE.match(s)
        or OP_RE.search(s)
        or ASSIGN_RE.match(s)
        or COMMENT_RE.match(s)
        or (FUNC_RE.match(s) and ("(" in s or "{" in s))
    )


def math_line(s):
    s = s.strip()
    if not s:
        return False
    if MATH_RE.search(s):
        return True
    if MATH2_RE.search(s):
        return sum(c in "=+-*/^()[]" for c in s) >= 2 or len(s.split()) <= 16
    return False


def likely_topic(flat, i):
    s = flat[i][0].strip()
    if not title_like(s):
        return False
    if len(s) > 75 or len(s.split()) > 12:
        return False
    if code_line(s) or math_line(s):
        return False
    if re.match(
        r"^(what|which|why|how|find|write|give|predict|consider|"
        r"explain|solve|state|describe)\b",
        s, re.I
    ):
        return False
    titleish = sum(
        w[:1].isupper() or w.lower() in
        {"of","and","or","to","in","on","for","with","a","an","the","by","from","as"}
        for w in s.split()
    )
    return titleish >= max(2, len(s.split()) // 2)


def structure(pages, kind, start):
    selected = [p for p in pages if p.number >= start]
    last_page = max((p.number for p in pages), default=0)
    ignored_index_marks = []

    # Flatten while preserving page numbers.
    flat = [(line, p.number) for p in selected for line in p.lines]

    if not flat:
        return []

    # Book-specific start/end boundaries.
    # These operate on extracted text and therefore do not depend on a
    # hard-coded physical PDF page being the exact first learning page.
    real_start = 0

    if kind == "c":
        # Start at the first real Chapter 1 heading.
        for i, (line, _) in enumerate(flat):
            if C_CH_RE.fullmatch(line.strip()):
                real_start = i
                break

    elif kind == "ds":
        # Start at the first numbered instructional section, not at a
        # dangling illustration/code fragment.
        for i, (line, _) in enumerate(flat):
            if SEC_RE.fullmatch(line.strip()) or SUB_RE.fullmatch(line.strip()):
                real_start = i
                break

    elif kind == "clrs":
        # Start at the first true chapter-title pair.
        for i, (line, _) in enumerate(flat):
            s = line.strip()
            if NUM_RE.fullmatch(s) and 1 <= int(s) <= 40:
                title, _, _ = next_nonempty(flat, i)
                if title and title_like(title):
                    real_start = i
                    break

    flat = flat[real_start:]

    result = []
    current = None

    chapter = chapter_title = ""
    section = section_title = ""
    subsection = subsection_title = ""

    i = 0
    while i < len(flat):
        s = flat[i][0].strip()

        # Stop before the back-of-book index -- but only if the marker is near the
        # end of the book. A stray line reading "Index"/"index" in a figure label or
        # running text used to end the whole book early (CLRS stopped at PDF page 269).
        if INDEX_START_RE.fullmatch(s):
            if flat[i][1] >= 0.8 * last_page:
                print(f"  Back-of-book index marker at PDF page {flat[i][1]}; stopping there.")
                break
            ignored_index_marks.append(flat[i][1])

        h = None

        if kind == "ds":
            h = heading_ds(flat, i)

        elif kind == "clrs":
            h = heading_clrs(flat, i)

        elif kind == "c":
            h = heading_c(flat, i)

        elif kind == "cpp":
            h = heading_cpp(flat, i)

        # Extra CLRS guard:
        # A numeric "chapter" marker must be followed by a plausible title,
        # and the title must not be an ordinary sentence.
        if h and kind == "clrs" and h[0] == "chapter":
            title = h[4].strip()
            if (
                len(title.split()) > 16
                or title.endswith((".", ";", ":"))
                or title.lower().startswith(("and ", "or ", "which ", "that "))
            ):
                h = None

        if h:
            if current and any(x.strip() for x, _ in current.lines):
                result.append(current)

            level, ch, sec, sub, title, page, consume = h

            if level == "chapter":
                chapter, chapter_title = ch, title
                section = section_title = subsection = subsection_title = ""

                heading_text = (
                    f"Chapter {chapter}: {title}"
                    if kind == "c"
                    else f"Chapter {chapter} {title}"
                )

            elif level == "sec":
                chapter, section, section_title = ch, sec, title
                subsection = subsection_title = ""
                heading_text = f"{sec} {title}"

            elif level == "sub":
                chapter, section, subsection, subsection_title = ch, sec, sub, title
                heading_text = f"{sub} {title}"

            else:
                continue

            current = Segment(
                chapter=chapter,
                chapter_title=chapter_title,
                section=section,
                section_title=section_title,
                subsection=subsection,
                subsection_title=subsection_title,
                heading=heading_text,
                heading_page=page,
            )

            i = consume
            continue

        if current is None:
            current = Segment()

        current.lines.append(flat[i])
        i += 1

    if current and any(x.strip() for x, _ in current.lines):
        result.append(current)

    if ignored_index_marks:
        print(f"  Ignored {len(ignored_index_marks)} early 'Index' line(s) on PDF pages "
              f"{ignored_index_marks[:8]} (not near the end of the book).")
    return result


def blocks(lines):
    out = []
    prose = []
    prose_pages = []

    def flush():
        nonlocal prose, prose_pages
        if not prose:
            return

        paragraphs, cur = [], []
        for line, page in zip(prose, prose_pages):
            if not line.strip():
                if cur:
                    paragraphs.append(cur)
                    cur = []
            else:
                cur.append((line.strip(), page))
        if cur:
            paragraphs.append(cur)

        for para in paragraphs:
            pieces, pages = [], []
            for text, page in para:
                if pieces and pieces[-1].endswith("-") and text[:1].islower():
                    pieces[-1] = pieces[-1][:-1] + text
                else:
                    pieces.append(text)
                pages.append(page)
            text = " ".join(pieces).strip()
            if text:
                out.append(Block("text", text, min(pages), max(pages)))

        prose, prose_pages = [], []

    i = 0
    n = len(lines)

    while i < n:
        line, page = lines[i]
        s = line.strip()

        if not s:
            prose.append(line)
            prose_pages.append(page)
            i += 1
            continue

        if code_line(line):
            flush()
            code, pages, strong = [], [], 0
            j, blanks = i, 0

            while j < n:
                l2, p2 = lines[j]
                s2 = l2.strip()

                if not s2:
                    blanks += 1
                    if blanks <= 1:
                        code.append("")
                        pages.append(p2)
                        j += 1
                        continue
                    break

                yes = code_line(l2)
                indented = len(l2) - len(l2.lstrip(" ")) >= 2

                if yes or indented or COMMENT_RE.match(s2):
                    code.append(l2.rstrip())
                    pages.append(p2)
                    strong += int(yes)
                    blanks = 0
                    j += 1
                else:
                    break

            while code and not code[-1].strip():
                code.pop()
                pages.pop()

            if code and strong:
                out.append(Block(
                    "code",
                    "\n".join(code).strip(),
                    min(pages),
                    max(pages)
                ))
                i = j
                continue

        if math_line(line):
            flush()
            math, pages = [s], [page]
            j = i + 1
            while j < n and lines[j][0].strip() and math_line(lines[j][0]):
                math.append(lines[j][0].strip())
                pages.append(lines[j][1])
                j += 1
            out.append(Block("math", "\n".join(math), min(pages), max(pages)))
            i = j
            continue

        if TABLE_RE.match(s) or FIG_RE.match(s):
            flush()
            special, pages = [s], [page]
            j = i + 1
            while j < n and len(special) < 30 and lines[j][0].strip():
                if code_line(lines[j][0]):
                    break
                special.append(lines[j][0].strip())
                pages.append(lines[j][1])
                j += 1
            kind = "table" if TABLE_RE.match(s) else "figure"
            out.append(Block(kind, "\n".join(special), min(pages), max(pages)))
            i = j
            continue

        prose.append(line)
        prose_pages.append(page)
        i += 1

    flush()
    return out


def split_large(block):
    if block.words <= HARD_MAX:
        return [block]

    sentences = [
        s.strip() for s in SENT_RE.split(block.text) if s.strip()
    ]

    result, cur, count = [], [], 0

    for s in sentences:
        w = words(s)
        if cur and count + w > TARGET:
            result.append(Block("text", " ".join(cur), block.start, block.end))
            cur, count = [s], w
        else:
            cur.append(s)
            count += w

    if cur:
        result.append(Block("text", " ".join(cur), block.start, block.end))

    return result


def pack(items):
    expanded = []
    for b in items:
        expanded.extend(split_large(b) if b.kind == "text" else [b])

    groups, cur, count = [], [], 0

    for b in expanded:
        if cur and count >= MIN_WORDS and count + b.words > MAX_WORDS:
            groups.append(cur)
            cur, count = [b], b.words
            continue

        cur.append(b)
        count += b.words

        if count >= TARGET:
            groups.append(cur)
            cur, count = [], 0

    if cur:
        groups.append(cur)

    if len(groups) >= 2:
        last = sum(b.words for b in groups[-1])
        prev = sum(b.words for b in groups[-2])
        if last < MIN_WORDS and prev + last <= HARD_MAX:
            groups[-2].extend(groups[-1])
            groups.pop()

    return groups


def build_chunks(segments, book_id, title):
    # Flatten all semantic blocks so small adjacent sections can be packed
    # together instead of creating one tiny chunk per section.
    # Oversized blocks are split before packing.
    units = []

    for seg in segments:
        for original in blocks(seg.lines):
            if not original.text.strip():
                continue

            expanded_blocks = (
                split_large(original)
                if original.kind == "text" and original.words > MAX_WORDS
                else [original]
            )

            for b in expanded_blocks:
                units.append({
                    "block": b,
                    "chapter": seg.chapter,
                    "chapter_title": seg.chapter_title,
                    "section": seg.section,
                    "section_title": seg.section_title,
                    "subsection": seg.subsection,
                    "subsection_title": seg.subsection_title,
                    "heading": seg.heading,
                })

    if not units:
        return []

    groups = []
    cur = []
    count = 0

    for unit in units:
        b = unit["block"]

        # Atomic block protection: If incoming block is code, math, table, or figure
        # and adding it exceeds TARGET, flush preceding prose early so the structured block
        # begins cleanly at the start of the next chunk.
        is_structured = b.kind in ("code", "math", "table", "figure")
        if cur and is_structured and (count + b.words > TARGET):
            groups.append(cur)
            cur, count = [], 0
        elif cur and count + b.words > MAX_WORDS:
            groups.append(cur)
            cur, count = [], 0

        cur.append(unit)
        count += b.words

        if count >= TARGET:
            groups.append(cur)
            cur, count = [], 0

    if cur:
        groups.append(cur)

    # Prevent a tiny final chunk when it can safely join the previous group.
    if len(groups) >= 2:
        last = sum(x["block"].words for x in groups[-1])
        prev = sum(x["block"].words for x in groups[-2])

        if last < MIN_WORDS and prev + last <= MAX_WORDS:
            groups[-2].extend(groups[-1])
            groups.pop()

    raw = []

    for group in groups:
        parts = []
        seen_headings = set()

        for unit in group:
            heading = unit["heading"].strip() if unit["heading"] else ""

            if heading and heading not in seen_headings:
                parts.append(heading)
                seen_headings.add(heading)

            parts.append(unit["block"].text)

        text = "\n\n".join(x for x in parts if x.strip()).strip()
        first = group[0]

        raw.append({
            "chapter": first["chapter"],
            "chapter_title": first["chapter_title"],
            "section": first["section"],
            "section_title": first["section_title"],
            "subsection": first["subsection"],
            "subsection_title": first["subsection_title"],
            "text": text,
            "page_start": min(x["block"].start for x in group),
            "page_end": max(x["block"].end for x in group),
            "contains_code": any(x["block"].kind == "code" for x in group),
        })

    # Add sentence overlap only when it fits inside MAX_WORDS.
    # The overlap is reduced automatically for chunks near the size limit.
    for i in range(1, len(raw)):
        if raw[i]["chapter"] != raw[i - 1]["chapter"]:
            continue

        current_words = words(raw[i]["text"])
        available = MAX_WORDS - current_words
        if available <= 0:
            continue

        previous = raw[i - 1]["text"]
        sents = [s.strip() for s in SENT_RE.split(previous) if s.strip()]
        if not sents:
            continue

        desired = max(1, int(words(previous) * OVERLAP))
        target = min(desired, available)

        tail = []
        count = 0

        for s in reversed(sents):
            w = words(s)

            if count + w > target:
                if not tail:
                    # Do not exceed MAX_WORDS just to force an overlap.
                    continue
                break

            tail.insert(0, s)
            count += w

            if count >= target:
                break

        if tail and count + current_words <= MAX_WORDS:
            overlap_text = " ".join(tail)
            raw[i]["text"] = overlap_text + "\n\n" + raw[i]["text"]

    # Final safety split: no serialized chunk is allowed above MAX_WORDS.
    # This is a last-resort guard for unusual PDF extraction cases.
    safe_raw = []

    for x in raw:
        if words(x["text"]) <= MAX_WORDS:
            safe_raw.append(x)
            continue

        b = Block("text", x["text"], x["page_start"], x["page_end"])
        pieces = split_large(b)

        for piece in pieces:
            y = dict(x)
            y["text"] = piece.text
            y["page_start"] = piece.start
            y["page_end"] = piece.end
            y["contains_code"] = x["contains_code"]
            safe_raw.append(y)

    result = []

    for i, x in enumerate(safe_raw, 1):
        text = x["text"]
        lower = text.lower()

        # Contextual breadcrumb generation for RAG grounding
        bc_parts = [title]
        if x["chapter"]:
            ch_str = f"Chapter {x['chapter']}"
            if x["chapter_title"]:
                ch_str += f": {x['chapter_title']}"
            bc_parts.append(ch_str)
        if x["section"]:
            sec_str = f"Section {x['section']}"
            if x["section_title"]:
                sec_str += f": {x['section_title']}"
            bc_parts.append(sec_str)
        if x["subsection"]:
            sub_str = f"Subsection {x['subsection']}"
            if x["subsection_title"]:
                sub_str += f": {x['subsection_title']}"
            bc_parts.append(sub_str)
        breadcrumb = " > ".join(bc_parts)
        search_text = f"[{breadcrumb}]\n\n{text}"

        result.append({
            "chunk_id": f"{book_id}_CHUNK_{i:05d}",
            "book_id": book_id,
            "book_title": title,
            "breadcrumb": breadcrumb,
            "search_text": search_text,
            "chapter": x["chapter"],
            "chapter_title": x["chapter_title"],
            "section": x["section"],
            "section_title": x["section_title"],
            "subsection": x["subsection"],
            "subsection_title": x["subsection_title"],
            "page_start": x["page_start"],
            "page_end": x["page_end"],
            "word_count": words(text),
            "char_count": len(text),
            "contains_code": x["contains_code"],
            "contains_equation": bool(MATH_RE.search(text) or MATH2_RE.search(text)),
            "contains_example": bool(re.search(r"\bexample\b", lower)),
            "contains_definition": bool(re.search(
                r"\bdefinition\b|\bdefined as\b|\bis called\b", lower
            )),
            "contains_algorithm": bool(re.search(
                r"\balgorithm\b|\bprocedure\b|\bpseudocode\b", lower
            )),
            "contains_table": bool(re.search(r"\btable\s+\d+", lower)),
            "contains_figure": bool(re.search(r"\bfigure\s+\d+", lower)),
            "text": text,
        })

    return result


def is_frontmatter(text):
    return bool(re.search(
        r"\bcopyright\b|all rights reserved|click here for terms of use|published by|library of congress cataloging",
        text, re.I
    ))


def is_toc_like(text):
    dotted_lines = len(re.findall(r"\.{4,}\s*\d+", text))
    has_toc_header = bool(re.search(r"(?im)^\s*(?:contents|table of contents|brief contents)\s*$", text))
    num_digits = len(re.findall(r"\b\d+\b", text))
    return (
        has_toc_header
        or dotted_lines >= 3
        or (bool(re.search(r"(?:\bcontents\b|\btable of contents\b)", text, re.I)) and num_digits >= 8)
    )


def is_index_like(text):
    index_patterns = len(re.findall(r"\b[A-Za-z]+(?:\s+[A-Za-z]+)?\s*,\s*\d+(?:\s*[-–,]\s*\d+)*", text))
    all_numbers = re.findall(r"\b\d{1,4}\b", text)
    num_words = max(words(text), 1)
    density = len(all_numbers) / num_words
    return (
        bool(re.search(r"(?im)^\s*(?:index|subject index|general index)\s*$", text))
        or len(re.findall(r"(?m)^[A-Z][A-Za-z'’\-]+\s*$", text)) >= 8
        or index_patterns >= 5
        or (density > 0.25 and num_words > 100)
    )


def noise_reasons(c, start_page, last_page):
    """Why a chunk looks like front matter / TOC / index -- only checked near the ends of the book."""
    in_front = c["page_start"] <= start_page + FRONT_ZONE_PAGES
    in_back = BACK_ZONE_FRAC is not None and c["page_end"] >= BACK_ZONE_FRAC * last_page
    if not (in_front or in_back):
        return []
    text = c.get("text", "")
    reasons = []
    if is_frontmatter(text):
        reasons.append("frontmatter")
    if is_toc_like(text):
        reasons.append("toc_like")
    if is_index_like(text):
        reasons.append("index_like")
    return reasons


def validate(chunks, start_page=0, last_page=10**9):
    ids = Counter(c["chunk_id"] for c in chunks)
    counts = [c["word_count"] for c in chunks]

    incomplete = sum(
        bool(
            c["text"].strip()
            and not c["contains_code"]
            and not c["contains_equation"]
            and not c["text"].strip().endswith(
                (".", "!", "?", '"', "'", ")", "]", "}")
            )
        )
        for c in chunks
    )

    word_mismatch = sum(
        c["word_count"] != words(c.get("text", ""))
        for c in chunks
    )

    frontmatter = sum("frontmatter" in noise_reasons(c, start_page, last_page) for c in chunks)
    toc_like = sum("toc_like" in noise_reasons(c, start_page, last_page) for c in chunks)
    index_like = sum("index_like" in noise_reasons(c, start_page, last_page) for c in chunks)

    return {
        "total": len(chunks),
        "avg": round(statistics.mean(counts), 1) if counts else 0,
        "min": min(counts) if counts else 0,
        "max": max(counts) if counts else 0,
        "short": sum(x < MIN_WORDS for x in counts),
        "large": sum(x > MAX_WORDS for x in counts),
        "hard_large": sum(x > HARD_MAX for x in counts),
        "incomplete": incomplete,
        "frontmatter": frontmatter,
        "toc_like": toc_like,
        "index_like": index_like,
        "missing": sum(
            not c.get("chunk_id")
            or not c.get("book_id")
            or not c.get("text", "").strip()
            for c in chunks
        ),
        "duplicates": sum(v > 1 for v in ids.values()),
        "bad_pages": sum(c["page_start"] > c["page_end"] for c in chunks),
        "word_mismatch": word_mismatch,
    }


def filter_contamination(chunks, start_page=0, last_page=10**9):
    """
    Split chunks into (kept, dropped) based on content-quality heuristics
    (front matter, table-of-contents pages, back-of-book index pages).
    These are noise to drop, not pipeline bugs — unlike duplicate IDs or
    bad page ranges, which indicate something is actually broken upstream.
    Each dropped entry carries a `reason` and the chunk it came from, so
    they can be inspected before being discarded for good.
    """
    kept, dropped = [], []

    for c in chunks:
        reasons = noise_reasons(c, start_page, last_page)

        if reasons:
            dropped.append({"chunk": c, "reasons": reasons})
        else:
            kept.append(c)

    return kept, dropped


def save(chunks, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def process(book_id, cfg):
    print("\n" + "=" * 72)
    print(f"PROCESSING {book_id}: {cfg['title']}")
    print("=" * 72)

    if not cfg["input"].exists():
        raise FileNotFoundError(
            f"Missing input file:\n{cfg['input']}"
        )

    text = cfg["input"].read_text(encoding="utf-8")
    pages = clean_pages(parse_pages(text))

    print(f"PDF pages: {len(pages)}")
    print(f"Content start page: {cfg['start']}")

    segs = structure(
        pages,
        cfg["kind"],
        cfg["start"],
    )

    print(f"Structural segments: {len(segs)}")

    chunks = build_chunks(
        segs,
        book_id,
        cfg["title"],
    )

    last_page = pages[-1].number if pages else 0
    kept_chunks, dropped = filter_contamination(chunks, cfg["start"], last_page)

    if dropped:
        print(f"\nDropping {len(dropped)} content-quality chunk(s) "
              f"(front matter / TOC / index noise):")
        for d in dropped[:10]:
            c = d["chunk"]
            preview = c["text"].strip().replace("\n", " ")[:100]
            print(f"  {c['chunk_id']} p{c['page_start']}-{c['page_end']} "
                  f"[{', '.join(d['reasons'])}]: {preview!r}")
        if len(dropped) > 10:
            print(f"  ... and {len(dropped) - 10} more")
        # Keep everything that was dropped so it can be eyeballed.
        save([dict(d["chunk"], drop_reasons=d["reasons"]) for d in dropped],
             cfg["output"].with_name(f"{book_id}_dropped.jsonl"))

    report = validate(kept_chunks, cfg["start"], last_page)

    print(f"\nChunks: {report['total']}")
    print(f"Average words: {report['avg']}")
    print(f"Min/Max words: {report['min']} / {report['max']}")
    print(f"Short (<{MIN_WORDS}): {report['short']}")
    print(f"Large (>{MAX_WORDS}): {report['large']}")
    print(f"Hard large (>{HARD_MAX}): {report['hard_large']}")
    print(f"Possible incomplete prose: {report['incomplete']}")
    print(f"Missing metadata: {report['missing']}")
    print(f"Duplicate IDs: {report['duplicates']}")
    print(f"Bad page ranges: {report['bad_pages']}")
    print(f"Word-count mismatches: {report['word_mismatch']}")
    print(f"Remaining frontmatter (should be 0): {report['frontmatter']}")
    print(f"Remaining TOC-like (should be 0): {report['toc_like']}")
    print(f"Remaining index-like (should be 0): {report['index_like']}")

    # Hard-fail only on signs of an actual pipeline bug — not on content
    # noise, which filter_contamination() already removed above.
    if (report["missing"] or report["duplicates"] or report["bad_pages"]
            or report["word_mismatch"] or report["hard_large"] or not kept_chunks):
        raise RuntimeError(
            f"Hard validation failed for {book_id}; file not written."
        )

    save(kept_chunks, cfg["output"])
    print(f"SUCCESS: {cfg['output']} ({len(kept_chunks)} chunks, "
          f"{len(dropped)} dropped as noise)")

    for c in kept_chunks[:2]:
        print(
            f"  {c['chunk_id']} | "
            f"pages {c['page_start']}-{c['page_end']} | "
            f"{c['word_count']} words | "
            f"{c['section']} {c['section_title']}"
        )


def main():
    print("=" * 72)
    print("ADAPTIVELEARN - ONE MASTER TEXTBOOK CHUNKER")
    print("=" * 72)

    if REBUILD_DS:
        process("DS", BOOKS["DS"])
    else:
        ds_out = BOOKS["DS"]["output"]
        if ds_out.exists():
            print(f"\nDS skipped; keeping existing validated file:\n{ds_out}")
        else:
            print("\nWARNING: DS_chunks.jsonl not found.")
            print("Set REBUILD_DS = True if DS must be regenerated.")

    for key in ("C", "CPP", "CLRS"):
        process(key, BOOKS[key])

    print("\n" + "=" * 72)
    print("MASTER CHUNKING COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()