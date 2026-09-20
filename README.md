# PrepPilot 🚀
> **AI-Driven Adaptive Learning & Examination System**  
> Powered by Semantic RAG, Multimodal Diagram Extraction, and Personalized Student Knowledge Modeling.

---

## 📖 Overview
**PrepPilot** is an intelligent, textbook-grounded learning and assessment platform designed for core Computer Science topics (Data Structures, Algorithms, C, and C++). 

Unlike standard RAG pipelines that blindly chunk text, PrepPilot preserves semantic book context, extracts textbook diagrams and figures, tracks student concept mastery over time, and supports both zero-config local operation (**SQLite + sqlite-vec**) and enterprise cloud scaling (**AWS RDS PostgreSQL + pgvector**).

---

## 🌟 Key Features

1. **High-Fidelity Document Processing**:
   - **PDF Text Ingestion**: Extracts text while filtering running headers, footers, and table-of-contents noise.
   - **Line De-hyphenation**: Automatically reunites split words across line breaks (e.g., `algo-` + `rithm` ➔ `algorithm`).
   - **Atomic Code & Math Protection**: Keeps code blocks, algorithm pseudocode, and mathematical definitions intact within single chunks.

2. **Multimodal Diagram & Figure Extraction**:
   - **Dual Strategy**: Directly extracts embedded raster images (XObjects) and automatically crops vector diagrams/charts based on textbook figure captions (`Figure X.Y`) rendered at crisp 150 DPI.
   - Links extracted diagram image paths directly to their surrounding textbook text chunks.

3. **Semantic Chunking with Hierarchical Breadcrumbs**:
   - Injects contextual paths into each chunk (e.g., `[CLRS > Chapter 2: Getting Started > Section 2.1 Insertion Sort]`).
   - Uses dual representation: rich breadcrumbs for vector semantic retrieval, and clean text for LLM generation.

4. **Unified Relational + Vector Storage**:
   - **Local Mode (Zero Setup)**: SQLite with the `sqlite-vec` extension (`preppilot.db`) stores both relational student data and 384-dimensional dense vectors in a single lightweight file.
   - **Cloud Mode (AWS RDS)**: PostgreSQL 16 with `pgvector` and HNSW indexing for high-concurrency cloud deployments.

5. **Personalized Student Knowledge Modeling**:
   - Built-in schema for student profiles, topic-by-topic mastery tracking (`p_mastery`, confidence intervals, practice counters), and quiz attempt histories.

---

## 🏗️ Architecture & Database Schema

PrepPilot unifies relational entities and vector search under a single storage layer:

```
                          ┌────────────────────────┐
                          │   Textbook PDFs (CS)   │
                          └───────────┬────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
      [Text Extraction & Cleaning]              [Diagram & Figure Cropper]
                 │                                         │
                 ▼                                         │
     [Contextual Semantic Chunker]                         │
                 │                                         │
                 ▼                                         │
   [Sentence-Transformers (384-d)]                         │
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │       Unified Relational + Vector Database       │
             │   (Local SQLite vec0  /  AWS RDS PostgreSQL)     │
             ├──────────────────────────────────────────────────┤
             │ • chunks (text, breadcrumbs, embedding vector)   │
             │ • extracted_images (figure captions, image paths)│
             │ • students (user profiles, target exams)         │
             │ • topic_mastery (adaptive skill scoring)         │
             │ • pyq_questions (past examination questions)     │
             │ • quiz_attempts (answer history & feedback)      │
             └──────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
PrepPilot/
├── scripts/
│   ├── extract_pdf.py            # Extracts raw text and page markers from PDFs
│   ├── extract_images.py         # Extracts diagrams & crops figures (150 DPI)
│   ├── semantic_chunker.py       # Context-aware chunking with breadcrumbs & healing
│   ├── build_vector_db.py        # Legacy FAISS index builder
│   ├── query_vector_db.py        # Similarity query script for FAISS
│   ├── review_chunks.py          # Quality inspection and statistics tool
│   ├── db.py                     # Unified SQLite + AWS RDS database access layer
│   ├── migrate_to_sqlite.py      # Encodes chunks with MiniLM and populates SQLite
│   └── migrate_sqlite_to_rds.py  # Migrates SQLite data & vectors to AWS RDS pgvector
├── .env.example                  # Environment variables template
├── .gitignore                    # Prevents credentials, databases, and heavy PDFs from leaking
├── requirements.txt              # Production and development dependencies
└── README.md                     # Project documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Installation
Clone the repository and install dependencies:

```bash
# Clone repository
git clone https://github.com/kaursimarr/PrepPilot.git
cd PrepPilot

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the `.env.example` template:
```bash
cp .env.example .env
```
If using AWS RDS, fill in your database credentials in `.env`:
```ini
PGHOST=your-rds-endpoint.us-east-1.rds.amazonaws.com
PGPORT=5432
PGDATABASE=preppilot
PGUSER=postgres
PGPASSWORD=your-secure-password
```
*(If you are using only local SQLite, you can leave `.env` as is—the system defaults to `preppilot.db`).*

---

## ⚡ Data Pipeline Workflow

### 🚀 All-in-One Master Execution (Recommended)
Run the entire end-to-end pipeline (PDF extraction ➔ diagram cropping ➔ semantic chunking ➔ vector database embedding ➔ verification test) with a single command:

```bash
# Run full pipeline end-to-end
python main.py

# Check system health, dataset sizes, and database counts
python main.py --status

# Test semantic search across indexed textbook chunks
python main.py --query "how does binary search work?"

# Optional: Sync database to AWS RDS PostgreSQL
python main.py --rds

# Verify bytecode compilation across all Python files
python main.py --compile-check
```

---

### 🔧 Individual Step Execution (Optional)
If you prefer running individual stages separately:

#### Step 1: Extract Text & Page Markers
```bash
python main.py --extract-text
# or: python scripts/extract_pdf.py
```

#### Step 2: Extract Diagrams and Figures
```bash
python main.py --extract-images
# or: python scripts/extract_images.py
```

#### Step 3: Run Contextual Semantic Chunking
```bash
python main.py --chunk
# or: python scripts/semantic_chunker.py
```

#### Step 4: Populate the Unified Local Database
```bash
python main.py --embed
# or: python scripts/migrate_to_sqlite.py
```

#### Step 5: (Optional) Migrate to AWS RDS PostgreSQL
```bash
python main.py --rds
# or: python scripts/migrate_sqlite_to_rds.py
```

---

## 💻 Python Database API Usage

You can interact with the database directly using `scripts.db`:

```python
from scripts.db import get_connection, search_chunks_by_embedding, record_quiz_attempt

# Connect to local database (or use get_rds_connection for AWS)
conn = get_connection("preppilot.db")

# Semantic vector search
results = search_chunks_by_embedding(
    conn,
    query_embedding=[0.023, -0.045, ...], # 384-dimensional float vector
    top_k=5,
    book_filter="CLRS"
)

for r in results:
    print(f"[{r['book']} - {r['chapter']}] (Score: {r['distance']:.4f})")
    print(r['text'][:150], "...\n")
```

---

## 🔒 Security & Best Practices
- **Credentials Protection**: Database secrets and connection URLs are loaded strictly through environment variables. The `.env` file and SQLite database files (`*.db`) are ignored by Git.
- **Atomic Operations**: Chunk migrations and mastery updates utilize transactional boundaries (`BEGIN / COMMIT`) to guarantee data integrity.
