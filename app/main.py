from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.db import init_schema
from app.services import search as search_svc, ask as ask_svc


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_schema()
    search_svc.warm_up()      # loads the embedding model once
    yield


app = FastAPI(title="PrepPilot API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AskIn(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(5, ge=1, le=15)
    book_id: Optional[str] = Field(None, description="DS | C | CPP | CLRS")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(body: AskIn):
    return search_svc.search(body.question, body.top_k, body.book_id)


@app.post("/ask")
def ask(body: AskIn):
    return ask_svc.ask(body.question, body.top_k, body.book_id)