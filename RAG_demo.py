"""
PrepPilot RAG demo: type a question -> see the retrieved chunks -> get a grounded answer -> see the sources.

    python rag_demo.py                       # interactive
    python rag_demo.py "What is a splay tree?"
    python rag_demo.py --book C -k 5         # limit to one book
    python rag_demo.py --no-llm              # retrieval only (no API key needed)
Type  quit  to leave.
"""
from __future__ import annotations

import argparse
import sys

from answer_generator import cited_numbers, format_sources, generate_answer, make_client
from retriever import BOOK_TITLES, DEFAULT_TOP_K, Retriever, print_chunks

RULE = "=" * 78


def ask(retriever: Retriever, question: str, args, client) -> None:
    print(f"\n{RULE}\n RETRIEVED CHUNKS (top {args.top_k})\n{RULE}")
    chunks = retriever.retrieve(question, top_k=args.top_k, book_id=args.book)
    if not chunks:
        print("No chunks found.")
        return
    print_chunks(chunks, full=args.full)
    if args.no_llm:
        return

    print(f"{RULE}\n ANSWER\n{RULE}")
    try:
        result = generate_answer(question, chunks, model=args.model, client=client)
    except Exception as e:  # noqa: BLE001 - missing key, no network, API error...
        print(f"Could not generate an answer: {e}")
        return
    print(result["answer"])
    cited = result["cited"] or cited_numbers(result["answer"])
    print(f"\n{RULE}\n SOURCES  (* = cited in the answer)\n{RULE}")
    print("\n".join(format_sources(chunks, cited)))


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description="PrepPilot RAG demo")
    ap.add_argument("question", nargs="*", help="optional; omit for interactive mode")
    ap.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--book", choices=sorted(BOOK_TITLES), default=None)
    ap.add_argument("--model", default=None, help="override the LLM model name")
    ap.add_argument("--full", action="store_true", help="show full chunk text instead of a preview")
    ap.add_argument("--no-llm", action="store_true", help="retrieval only")
    args = ap.parse_args()

    client = None
    if not args.no_llm:
        try:
            client = make_client()
        except Exception as e:  # noqa: BLE001
            print(f"[!] {e}\n    Continuing with retrieval only.\n")
            args.no_llm = True

    retriever = Retriever()
    try:
        if args.question:
            ask(retriever, " ".join(args.question), args, client)
            return
        print("PrepPilot RAG demo. Ask a question about DS, C, C++ or CLRS (type 'quit' to exit).")
        while True:
            try:
                q = input("\nQuestion> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                ask(retriever, q, args, client)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()