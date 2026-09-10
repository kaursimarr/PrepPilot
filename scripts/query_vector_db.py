from pathlib import Path
import json
import sys

import numpy as np

try:
    import faiss
except ImportError:
    print("ERROR: faiss-cpu is not installed.")
    print("Run: python -m pip install faiss-cpu")
    raise SystemExit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers is not installed.")
    print("Run: python -m pip install -U sentence-transformers")
    raise SystemExit(1)


BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "vector_db" / "learning_material"

INDEX_FILE = VECTOR_DIR / "index.faiss"
METADATA_FILE = VECTOR_DIR / "metadata.jsonl"
CONFIG_FILE = VECTOR_DIR / "config.json"

TOP_K = 5
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("Enter your question: ").strip()

    if not query:
        print("No question entered.")
        return

    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        print("Vector database not found.")
        print("Run build_vector_db.py first.")
        return

    model_name = DEFAULT_MODEL
    if CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        model_name = config.get("embedding_model", DEFAULT_MODEL)

    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)

    index = faiss.read_index(str(INDEX_FILE))

    metadata = []
    with METADATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))

    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"Index/metadata mismatch: {index.ntotal} vs {len(metadata)}"
        )

    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, ids = index.search(
        query_vector,
        min(TOP_K, index.ntotal)
    )

    print("\n" + "=" * 72)
    print("TOP RETRIEVED CHUNKS (all books)")
    print("=" * 72)
    print(f"Query: {query}")

    for rank, (score, idx) in enumerate(
        zip(scores[0], ids[0]), 1
    ):
        idx = int(idx)
        if idx < 0:
            continue

        chunk = metadata[idx]

        print("\n" + "-" * 72)
        print(
            f"#{rank} {chunk['chunk_id']} "
            f"| book={chunk.get('book_id', '')} "
            f"({chunk.get('book_title', '')}) "
            f"| similarity={float(score):.4f}"
        )
        print(
            f"Section: {chunk.get('section', '')} "
            f"{chunk.get('section_title', '')}"
        )
        print(
            f"Subsection: {chunk.get('subsection', '')} "
            f"{chunk.get('subsection_title', '')}"
        )
        print(
            f"Pages: {chunk.get('page_start')}-{chunk.get('page_end')} "
            f"| Words: {chunk.get('word_count')}"
        )

        preview = chunk["text"][:900].replace("\n", " ")
        print(f"Text: {preview}")


if __name__ == "__main__":
    main()