from pathlib import Path
import json
import sys

import numpy as np


# ============================================================
# DEPENDENCIES
# ============================================================

try:
    import faiss
except ImportError:
    print("ERROR: faiss-cpu is not installed.")
    print("Run:")
    print("    python -m pip install faiss-cpu")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers is not installed.")
    print("Run:")
    print("    python -m pip install -U sentence-transformers")
    sys.exit(1)


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

CHUNKS_DIR = BASE_DIR / "chunks_semantic"

VECTOR_DIR = BASE_DIR / "vector_db" / "learning_material"

CHUNK_FILES = {
    "DS": CHUNKS_DIR / "DS_chunks.jsonl",
    "C": CHUNKS_DIR / "C_chunks.jsonl",
    "CPP": CHUNKS_DIR / "CPP_chunks.jsonl",
    "CLRS": CHUNKS_DIR / "CLRS_chunks.jsonl",
}

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Small enough for ordinary laptops, large enough to be reasonably fast.
BATCH_SIZE = 32


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path: Path, expected_book_id: str) -> list[dict]:
    """Load and validate one book's chunk file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing chunk file for {expected_book_id}:\n{path}\n\n"
            "Run semantic_chunker.py first."
        )

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name}, line {line_number}: {exc}"
                ) from exc

            required = ("chunk_id", "book_id", "text")

            for key in required:
                if key not in record:
                    raise ValueError(
                        f"{path.name}, line {line_number}: "
                        f"missing required field '{key}'."
                    )

            if record["book_id"] != expected_book_id:
                raise ValueError(
                    f"{path.name}, line {line_number}: "
                    f"expected book_id '{expected_book_id}', "
                    f"got '{record['book_id']}'."
                )

            if not str(record["text"]).strip():
                raise ValueError(
                    f"{path.name}, line {line_number}: empty text."
                )

            records.append(record)

    if not records:
        raise ValueError(
            f"No chunks found in {path}."
        )

    return records


# ============================================================
# VALIDATE GLOBAL DATASET
# ============================================================

def validate_global_chunks(chunks: list[dict]) -> None:
    """Check IDs, page ranges, and basic metadata before indexing."""
    seen = set()

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]

        if chunk_id in seen:
            raise ValueError(
                f"Duplicate chunk ID found: {chunk_id}"
            )

        seen.add(chunk_id)

        if (
            "page_start" in chunk
            and "page_end" in chunk
            and chunk["page_start"] > chunk["page_end"]
        ):
            raise ValueError(
                f"Bad page range in {chunk_id}: "
                f"{chunk['page_start']}-{chunk['page_end']}"
            )

    print(f"Global metadata validation: PASSED")
    print(f"Unique chunk IDs: {len(seen)}")


# ============================================================
# BUILD COMBINED FAISS INDEX
# ============================================================

def main() -> None:
    print()
    print("=" * 78)
    print("ADAPTIVELEARN - FINAL COMBINED LEARNING MATERIAL VECTOR DB")
    print("=" * 78)

    print("\nBooks:")
    print("  DS   -> Handbook of Data Structures and Applications")
    print("  C    -> Let Us C")
    print("  CPP  -> C++: The Complete Reference")
    print("  CLRS -> Introduction to Algorithms")
    print(f"\nEmbedding model: {MODEL_NAME}")

    # --------------------------------------------------------
    # Load all four books
    # --------------------------------------------------------

    all_chunks = []

    print("\nLoading chunk files...")

    per_book_counts = {}

    for book_id in ("DS", "C", "CPP", "CLRS"):
        path = CHUNK_FILES[book_id]

        chunks = load_jsonl(
            path,
            book_id
        )

        all_chunks.extend(chunks)
        per_book_counts[book_id] = len(chunks)

        print(
            f"  {book_id:<4} -> "
            f"{len(chunks):>6} chunks"
        )

    print(
        f"\nTOTAL LEARNING-MATERIAL CHUNKS: "
        f"{len(all_chunks)}"
    )

    validate_global_chunks(
        all_chunks
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print("\nLoading embedding model...")
    model = SentenceTransformer(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    print(
        f"\nCreating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"Unexpected embedding shape: "
            f"{embeddings.shape}"
        )

    if embeddings.shape[0] != len(all_chunks):
        raise RuntimeError(
            "Embedding count does not match chunk count."
        )

    dimension = int(
        embeddings.shape[1]
    )

    print(
        f"\nEmbedding matrix shape: "
        f"{embeddings.shape}"
    )

    # --------------------------------------------------------
    # Build FAISS
    # --------------------------------------------------------

    # Embeddings are normalized, so inner product = cosine similarity.
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    if index.ntotal != len(all_chunks):
        raise RuntimeError(
            f"FAISS vector count mismatch: "
            f"{index.ntotal} vs {len(all_chunks)}"
        )

    # --------------------------------------------------------
    # Save everything into ONE final folder
    # --------------------------------------------------------

    VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    index_path = (
        VECTOR_DIR / "index.faiss"
    )

    metadata_path = (
        VECTOR_DIR / "metadata.jsonl"
    )

    config_path = (
        VECTOR_DIR / "config.json"
    )

    faiss.write_index(
        index,
        str(index_path)
    )

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        for row in all_chunks:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

    config = {
        "dataset": "learning_material",
        "books": [
            "DS",
            "C",
            "CPP",
            "CLRS",
        ],
        "book_chunk_counts": per_book_counts,
        "total_chunks": len(all_chunks),
        "embedding_model": MODEL_NAME,
        "embedding_dimension": dimension,
        "faiss_index_type": "IndexFlatIP",
        "similarity": "cosine",
        "vectors_normalized": True,
        "metadata_order_matches_faiss": True,
    }

    config_path.write_text(
        json.dumps(
            config,
            indent=2
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Integrity test
    # --------------------------------------------------------

    # A vector should retrieve itself as rank 1 with the highest score.
    test_scores, test_ids = index.search(
        embeddings[:1],
        1
    )

    if int(test_ids[0][0]) != 0:
        raise RuntimeError(
            "FAISS integrity check failed."
        )

    print("\n" + "=" * 78)
    print("SUCCESS - ONE COMBINED VECTOR DATABASE CREATED")
    print("=" * 78)

    print(
        f"\nTotal vectors : {index.ntotal}"
    )
    print(
        f"Dimensions    : {dimension}"
    )

    print("\nBooks included:")
    for book_id in ("DS", "C", "CPP", "CLRS"):
        print(
            f"  {book_id:<4} "
            f"{per_book_counts[book_id]:>6} chunks"
        )

    print("\nFiles:")
    print(f"  FAISS index : {index_path}")
    print(f"  Metadata    : {metadata_path}")
    print(f"  Config      : {config_path}")

    print(
        "\nThis is the ONLY vector database you need for "
        "the Learning Material Dataset."
    )


if __name__ == "__main__":
    main()