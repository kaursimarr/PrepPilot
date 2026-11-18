from typing import Optional
from answer_generator import (SYSTEM_PROMPT, NO_CONTEXT_MSG, MIN_TOP_SIMILARITY,
                              build_user_message, cited_numbers)
from retriever import location_label
from app import llm
from app.services.search import search

def ask(question: str, top_k: int = 5, book_id: Optional[str] = None) -> dict:
    chunks = search(question, top_k, book_id)
    if not chunks or max(c["similarity"] for c in chunks) < MIN_TOP_SIMILARITY:
        return {"answer": NO_CONTEXT_MSG, "cited": [], "sources": [], "used_llm": False}
    answer = llm.chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, chunks)},
    ], max_tokens=1024).strip()
    cited = cited_numbers(answer)
    sources = [{
        "rank": c["rank"], "chunk_id": c["chunk_id"], "book_id": c["book_id"],
        "location": location_label(c), "pages": c["pages"],
        "similarity": round(c["similarity"], 3), "cited": c["rank"] in cited,
        "text": c["text"], "image_paths": c["image_paths"],
    } for c in chunks]
    return {"answer": answer, "cited": sorted(cited), "sources": sources, "used_llm": True}