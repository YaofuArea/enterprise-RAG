# Enterprise RAG

A production-grade Retrieval-Augmented Generation (RAG) system built from scratch, with dense inline comments explaining every technical decision. Designed as a learning resource for engineers building enterprise knowledge base Q&A systems.

## What's covered

| Module | Techniques |
|--------|-----------|
| Document Ingestion | Multi-format parsing (PDF/Word/TXT), recursive character splitting, semantic chunking, chunk overlap |
| Vector Indexing | FAISS (Flat / IVF / HNSW trade-offs), multi-granularity index (Parent Document Retriever) |
| Sparse Indexing | BM25 Okapi, Chinese tokenization |
| Hierarchical Retrieval | **RAPTOR** — GMM clustering + recursive summarization + Collapsed Tree search |
| Query Enhancement | **HyDE**, **Multi-Query**, **Step-Back Prompting**, coreference resolution |
| Hybrid Search | Dense + Sparse + **RRF (Reciprocal Rank Fusion)** |
| Re-ranking | **Cross-Encoder Reranker** (Bi-Encoder vs Cross-Encoder explained) |
| Diversity | **MMR (Maximal Marginal Relevance)** |
| Semantic Cache | FAISS similarity matching + LFU eviction + TTL expiry + distributed design notes |
| Generation | **Lost-in-the-Middle** reordering, token budget management, streaming output (SSE) |
| Multi-turn Dialogue | Sliding window + summary compression, session management |
| API Server | FastAPI + SSE + dependency injection + health checks |

## Project structure

```
enterprise_rag/
├── data/                      # Sample knowledge base documents
│   ├── hr_policy.txt          # HR policy (leave, compensation, attendance)
│   ├── it_system_manual.txt   # ERP system operations manual
│   ├── product_faq.txt        # Product FAQ
│   └── security_policy.txt    # Information security policy
│
├── ingestion.py               # Document parsing + chunking strategies
├── indexing.py                # FAISS vector index + BM25 sparse index
├── raptor.py                  # RAPTOR hierarchical retrieval
├── query_enhancement.py       # HyDE / Multi-Query / Step-Back
├── retrieval.py               # Hybrid search + Reranker + MMR
├── cache.py                   # Semantic cache
├── generation.py              # Prompt engineering + streaming generation
├── conversation.py            # Multi-turn conversation management
├── pipeline.py                # Full pipeline assembly
├── server.py                  # FastAPI service
└── requirements.txt
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run each module in order

Each file runs standalone with a built-in demo — no API key required.

```bash
export OMP_NUM_THREADS=1   # Required on macOS (fixes FAISS + PyTorch OpenMP conflict)

python ingestion.py         # Chunking strategies demo
python indexing.py          # Index build + vector vs BM25 comparison
python raptor.py            # RAPTOR tree build + hierarchical retrieval demo
python query_enhancement.py # Query enhancement demo (mock mode, no API key needed)
python retrieval.py         # Hybrid search + RRF + MMR demo
python cache.py             # Semantic cache + threshold sensitivity demo
python generation.py        # Prompt structure + Lost-in-the-Middle demo
python conversation.py      # Multi-turn history management demo
python pipeline.py          # Full RAG pipeline (mock mode)
```

### 3. Connect a real LLM (optional)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export USE_REAL_LLM=1

OMP_NUM_THREADS=1 uvicorn server:app --port 8000
```

Visit http://localhost:8000/docs for the interactive API docs.

### 4. API usage

```bash
# Non-streaming Q&A
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of annual leave do I get?", "session_id": "user_001"}'

# Upload a new document
curl -X POST http://localhost:8000/ingest \
  -F "file=@your_document.pdf"

# Clear the semantic cache
curl -X DELETE http://localhost:8000/cache
```

## Key design decisions

### Why RRF instead of weighted score fusion

Vector scores (cosine, ~0.5–0.9) and BM25 scores (unbounded, 0–20+) live on completely different scales. Weighted fusion requires manual tuning and behaves inconsistently across queries. RRF uses rank positions instead of raw scores — naturally comparable, with k=60 empirically optimal across many datasets.

### Why RAPTOR is rebuilt in-process instead of loaded from disk

Python pickle binds to the module path at save time. Loading from a different entry point causes `AttributeError: Can't get attribute 'RaptorTree' on <module '__main__'>`. The mock summarizer rebuilds the tree in ~2s, which is acceptable. For production persistence, serialize the tree to JSON instead.

### OMP_NUM_THREADS=1 on macOS

FAISS (uses OpenMP) and PyTorch (also uses OpenMP) conflict on macOS, causing a segfault when both are loaded in the same process. Setting `OMP_NUM_THREADS=1` disables FAISS multithreading and resolves the crash. This is a macOS-only issue — Linux production environments are unaffected.

### Semantic cache threshold calibration

The `bge-small-zh` model produces cosine similarities of ~0.88–0.92 for semantically equivalent questions. Thresholds can't be transferred between embedding models — always calibrate on your own data by collecting historical query pairs, labeling which should match, and finding the F1-optimal threshold on a precision-recall curve.

## Tech stack

| Library | Purpose |
|---------|---------|
| `sentence-transformers` | Local embedding model, free, no API required |
| `faiss-cpu` | Vector indexing (Meta open source) |
| `rank-bm25` | BM25 sparse retrieval |
| `scikit-learn` | GMM clustering (used by RAPTOR) |
| `anthropic` | LLM generation (swappable for OpenAI) |
| `fastapi` + `uvicorn` | API server |
| `PyMuPDF` | PDF parsing |
| `tiktoken` | Token counting |

## License

MIT
