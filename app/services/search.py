"""All retrieval runs on ONE dedicated thread, because the sqlite-vec connection
inside Retriever can't be shared across FastAPI's worker threads."""
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from retriever import Retriever

_pool = ThreadPoolExecutor(max_workers=1)
_retriever: Optional[Retriever] = None

def _get() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever

def warm_up():
    _pool.submit(lambda: _get().retrieve("warm up", top_k=1)).result()

def search(question: str, top_k: int = 5, book_id: Optional[str] = None):
    return _pool.submit(lambda: _get().retrieve(question, top_k, book_id)).result()