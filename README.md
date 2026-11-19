# PrepPilot

PrepPilot is a study assistant for computer-science students. It reads four textbooks, indexes them into a
searchable database, and exposes a backend API that answers questions with citations and generates verified
practice questions. The UI is built separately and talks to this API.

**Textbooks indexed**

| ID | Book |
|---|---|
| `DS` | Handbook of Data Structures and Applications |
| `C` | Let Us C |
| `CPP` | C++: The Complete Reference |
| `CLRS` | Introduction to Algorithms (CLRS) |

---

## What works today

| Feature | Endpoint | Notes |
|---|---|---|
| Hybrid search | `POST /search` | Vector search + keyword search, merged with Reciprocal Rank Fusion |
| Q&A with citations | `POST /ask` | Answers only from retrieved textbook chunks, cites `[1]`, `[2]` with book / chapter / page |
| Chapter list | `GET /books/{book_id}/chapters` | For dropdowns in the UI |
| AI question generator | `POST /generate/questions` | MCQ, short and long questions; every question is verified against its source chunk |
| Question bank | `GET /questions` | Verified questions, filter by book / chapter / type |
| Health check | `GET /health` | Chunk count, keyword-search status, which LLM keys are set |

Planned (not built yet): syllabus intelligence, PYQ intelligence, prerequisite graph, knowledge model,
adaptive quiz, flashcards, likely questions, full mock exam, exam trend analyzer.

---

## How it works

```
PDF books
  -> Stage 1  text extraction (PyMuPDF)
  -> Stage 2  diagram / figure extraction
  -> Stage 3  semantic chunking, chapter titles from PDF bookmarks, chunk cleanup (fix_chunks.py)
  -> Stage 4  embeddings (all-MiniLM-L6-v2, 384-dim) into SQLite + sqlite-vec  ->  preppilot.db
  -> Stage 6  retrieval evaluation

preppilot.db  ->  api.py (FastAPI)  ->  UI
```

- **Database:** one SQLite file, `preppilot.db`, holds the chunks, their vectors (`vec_chunks`), extracted image
  paths, and the tables used by upcoming features. `api.py` adds a keyword index (`chunks_fts`, SQLite FTS5)
  automatically on first start.
- **Hybrid search:** the question is embedded and matched against vectors (top 30); the same question is also run as
  a BM25 keyword query (top 30). The two ranked lists are merged with Reciprocal Rank Fusion (k = 60). Each result
  has a `found_by` field: `vector`, `keyword` or `vector+keyword`.
- **Q&A:** the top chunks are sent to the LLM with a strict prompt: use only the excerpts, cite them, and say so
  when the excerpts do not contain the answer. If the best chunk is too dissimilar, the LLM is not called.
- **Question generator:** picks substantial, text-heavy chunks (preferring ones with no questions yet), asks the
  LLM to write questions from one chunk, then makes a second LLM pass that checks the answer is really supported by
  that chunk. Only verified questions are saved. MCQ options are shuffled so the answer letter is not predictable.
  Each chunk costs two LLM calls, so one request of 10 questions uses about 10-14 calls.
- **LLM:** free tier. Groq (`openai/gpt-oss-120b`, then `openai/gpt-oss-20b`) with Google Gemini
  (`gemini-2.5-flash`) as fallback. Model names change over time; override with `GROQ_MODEL` / `GEMINI_MODEL` in `.env`.

---

## Project structure

```
api.py                  FastAPI backend (the whole API in one file)
main.py                 master pipeline: extraction -> chunking -> embedding -> eval
fix_chunks.py           chunk cleanup (pipeline stage 3c)
scripts/                pipeline helpers, including db.py (connection + vector search)
chapter_titles.json     chapter titles read from PDF bookmarks
requirements.txt        Python dependencies
API_TESTING.txt         step-by-step manual tests for the API
.env.example            template for API keys (copy to .env)
```

Not in git (see below): `preppilot.db`, `extracted_images/`, `books/`, `.env`.

---

## Setup

Requires Python 3.10+ (developed on 3.13).

```bash
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

Edit `.env` and add at least one free key:

```
GROQ_API_KEY=...      # https://console.groq.com
GEMINI_API_KEY=...    # https://aistudio.google.com
```

### Data files (not in git)

`preppilot.db` (~70 MB) and `extracted_images/` are generated from copyrighted textbooks, so they are not
committed. To run the API you need them in the project root:

- **Teammates:** get `preppilot.db` and `extracted_images/` from the project owner (shared drive) and place both in the
  project root, next to `api.py`.
- **Rebuilding from scratch:** put the four PDFs in `books/` and run `python main.py`. If a `preppilot.db` already
  exists, use `python main.py --fresh-db`, otherwise stale vectors pile up and search returns too few results.

---

## Run the API

From the project root (the folder containing `api.py`):

```bash
python -m uvicorn api:app --reload
```

Wait for `Application startup complete` (the first start also builds the keyword index, about 10-20 seconds).
Interactive docs for testing and for the UI developer: **http://127.0.0.1:8000/docs**

If port 8000 is busy, add `--port 8001`.

---

## API reference

All request bodies are JSON. Swagger at `/docs` shows a working example for each endpoint.

### `POST /search`
```json
{"question": "What is a binary search tree?", "top_k": 5, "book_id": null}
```
Returns a ranked list of chunks with `text`, `book`, `chapter`, `chapter_title`, `section`, `pages`,
`similarity`, `found_by`, `location` (ready-made citation string) and `image_paths` (diagrams on that page).
`book_id` is optional: `DS`, `C`, `CPP` or `CLRS`.

### `POST /ask`
Same body as `/search`. Returns:
```json
{"answer": "... [2] ...", "cited": [2, 3], "sources": [ ...chunks with "cited": true/false... ], "used_llm": true}
```
If the LLM is unreachable the API returns HTTP 502 with a readable `detail` message.

### `GET /books/{book_id}/chapters`
Returns `[{"chapter": "12", "chapter_title": "Binary Search Trees", "chunks": 64}, ...]`.

### `POST /generate/questions`
```json
{"book_id": "CLRS", "chapter": "12", "qtype": "mcq", "count": 4, "difficulty": "medium"}
```
- `qtype`: `mcq` | `short` | `long`; `difficulty`: `easy` | `medium` | `hard`; `count`: 1-10 (a maximum, since
  unverified questions are discarded).
- `book_id` and `chapter` are optional (leave them out to sample any chunk).
- Returns `saved`, `rejected`, `errors` and the saved `questions`. For MCQs `answer` is the letter (`A`-`D`)
  into `options`.

### `GET /questions`
Query parameters: `book_id`, `chapter`, `qtype`, `limit` (default 20), `include_answers` (default `true`;
use `false` when showing a quiz to a student).

### `GET /health`
```json
{"status": "ok", "chunks": 12864, "keyword_search": true, "llm_keys": {"groq": true, "gemini": false}}
```

See `API_TESTING.txt` for a full manual test plan with expected results.

---

## Troubleshooting

| Problem | Cause / fix |
|---|---|
| `/docs` shows version 0.1.0 or `/health` returns only `{"status":"ok"}` | An old server is still running. Stop it, or start the new one on another port. |
| `/search` returns fewer than `top_k` results | Orphan vectors in `vec_chunks` from re-running the pipeline without `--fresh-db`. Rebuild with `python main.py --fresh-db`. |
| `/ask` or `/generate/questions` returns 502 | No valid LLM key, rate limit (429), or the model name was retired. Check `.env`, wait a minute, or set `GROQ_MODEL`. |
| `No module named 'scripts'` | Start the server from the project root, not from inside another folder. |
| `/generate/questions` says "No suitable chunks" | Check `book_id` / `chapter` values (see `/books/{book_id}/chapters`). Do not leave Swagger's placeholder `"string"` in the body. |
| `preppilot.db not found` | Copy the database into the project root (see Data files). |

---

## Tech stack

Python, FastAPI, SQLite + sqlite-vec + FTS5, sentence-transformers (all-MiniLM-L6-v2), PyMuPDF,
Groq / Gemini via the OpenAI-compatible API.
