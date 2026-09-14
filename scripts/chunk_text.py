# NOTE: simple word-count chunker (part of the basic pipeline:
# clean_text.py -> chunk_text.py). Its output ("../chunks") is NOT read by
# build_vector_db.py, which instead reads "chunks_semantic/*.jsonl" produced
# by semantic_chunker.py. Keep this only if you want a quick, structure-blind
# alternative to compare against; otherwise semantic_chunker.py is the one
# that actually feeds the vector DB.
import os
import json

INPUT_FOLDER = "../cleaned"
OUTPUT_FOLDER = "../chunks"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CHUNK_SIZE = 800      # words
OVERLAP = 100         # words

def chunk_words(words):
    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = words[start:end]
        chunks.append(chunk)

        start += CHUNK_SIZE - OVERLAP

    return chunks


for file in os.listdir(INPUT_FOLDER):

    if file.endswith(".txt"):

        with open(os.path.join(INPUT_FOLDER, file), "r", encoding="utf-8") as f:
            text = f.read()

        words = text.split()
        chunks = chunk_words(words)

        output_file = file.replace(".txt", ".jsonl")

        with open(os.path.join(OUTPUT_FOLDER, output_file), "w", encoding="utf-8") as out:

            for i, chunk in enumerate(chunks, start=1):

                record = {
                    "chunk_id": f"{file[:-4].upper()}_{i:04}",
                    "book_title": file[:-8],
                    "chunk_number": i,
                    "text": " ".join(chunk)
                }

                out.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"{file}: {len(chunks)} chunks created")

print("Chunking Complete!")