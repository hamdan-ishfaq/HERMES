# Hermes — Agentic RAG Research Assistant

> Agentic RAG over PDFs, URLs, and YouTube with grounded citations. Hybrid dense + BM25 retrieval with Reciprocal Rank Fusion and cross-encoder reranking, orchestrated as a LangGraph retrieval + synthesis pipeline behind a JWT-secured FastAPI, with semantic caching and RAGAS-based quality tracking.

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-FF4F00?style=for-the-badge)
![Qdrant](https://img.shields.io/badge/Qdrant-1A1A1A?style=for-the-badge)
![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor=white)

---

## What it does

- **Ingest** PDFs, web pages, and YouTube transcripts into a hybrid vector store.
- **Ask** natural-language questions and get answers grounded in the ingested sources, with citation cards that link back to the page, URL, or video timestamp.
- **Retrieve** with hybrid dense + sparse search, fuse with RRF, and rerank with a cross-encoder before any LLM call.
- **Cache** semantically similar questions to skip the pipeline on repeats.
- **Track** answer quality with RAGAS (faithfulness, answer relevancy, context precision/recall).

---

## Architecture

```mermaid
flowchart TD
    UI[React Frontend] -->|JWT + query| API[FastAPI Router]
    API --> Cache{cache_check<br>semantic cache}

    Cache -->|hit ≥ 0.95| Return[Return cached answer<br>bypass retrieval + generation]
    Return --> UI

    Cache -->|miss| Supervisor[supervisor<br>classify complexity]
    Supervisor --> Research[research_agent]

    subgraph Retrieval
        Research --> Embed[nomic-embed-text dense<br>+ BM25 sparse]
        Embed --> Qdrant[(Qdrant)]
        Qdrant -->|dense + sparse prefetch| RRF[RRF fusion]
        RRF --> Rerank[Cross-encoder rerank<br>ms-marco-MiniLM-L-6-v2]
        Rerank --> Parent[Parent-context expansion]
    end

    Parent --> Draft[Draft answer + citations]
    Draft -->|simple| UI
    Draft -->|multi-hop / synthesis| Synthesis[synthesis_agent]
    Synthesis --> UI
```

The LangGraph pipeline is: `START → cache_check → [END on hit | supervisor → research_agent → (synthesis_agent) → END]`. The supervisor classifies query complexity; simple queries finish after research, while multi-hop/synthesis queries get a second synthesis pass.

---

## Core components

### Hybrid retrieval (`backend/src/rag/retriever.py`)
- **Dense** embeddings via `nomic-embed-text` (Ollama) capture semantic meaning.
- **Sparse** BM25 vectors via `fastembed` capture exact lexical / keyword matches.
- Qdrant runs both as prefetches and fuses them with **Reciprocal Rank Fusion (RRF)**.
- **Parent-child chunking** (`chunker.py`): small child chunks are indexed for precise retrieval, but the larger parent chunk is returned to the LLM for context. Parent text is persisted in the Qdrant payload so expansion works across processes and restarts.

### Cross-encoder reranking (`backend/src/rag/reranker.py`)
Vector similarity scores candidates independently. The `ms-marco-MiniLM-L-6-v2` cross-encoder rescores the query against each candidate jointly, and contexts below `MIN_RERANK_SCORE` (default `0.35`) are dropped before the prompt is built.

### Semantic cache (`backend/src/rag/cache.py`)
A Redis-backed cache embeds each query and compares against stored queries by cosine similarity (threshold `0.95`). On a hit, `cache_check` (the first graph node) returns the stored answer and **bypasses retrieval and generation** — an honest latency optimization, not a bypass of "the entire pipeline" before classification.

### LangGraph agents (`backend/src/agents/`)
- `cache_check.py` — semantic cache gate (entry node).
- `supervisor.py` — classifies complexity (simple / multi-hop / synthesis).
- `research.py` — retrieves, reranks, and drafts an answer with citations.
- `synthesis.py` — refines multi-hop / cross-document answers.

### FastAPI + JWT (`backend/src/`)
- `POST /api/auth/register`, `POST /api/auth/login` — JWT auth.
- `POST /api/research` — run a query (Bearer token required).
- `POST /api/ingest/pdf`, `/api/ingest/url`, `/api/ingest/youtube` — ingestion.
- `GET /api/eval/dashboard`, `POST /api/eval/run` — RAGAS reporting.

---

## Evaluation (RAGAS)

Quality is measured with RAGAS using a local Ollama judge (`llama3.1:8b`) over a golden Q&A set. Latest run (`backend/eval_report.json`, 10 questions):

| Metric | Score |
|---|---|
| Faithfulness | 0.75 |
| Answer relevancy | 0.72 |
| Context precision | 0.72 |
| Context recall | 0.81 |

Run an evaluation:

```bash
# via API (background job, JWT required)
curl -X POST localhost:8000/api/eval/run -H "Authorization: Bearer <token>"

# or directly
cd backend && uv run python -m src.evaluation.ragas_eval
```

RAGAS is not part of CI (it depends on a running Ollama judge and takes several minutes). CI runs the mocked unit tests; an integration test exercises the real retriever locally.

---

## Setup

### 1. Start infrastructure

```bash
docker compose up -d postgres redis qdrant
# optional: run a local LLM/embedding server too
# docker compose --profile local up -d
```

You also need an Ollama server reachable at `OLLAMA_API_BASE` serving `nomic-embed-text` (embeddings) and `llama3.1:8b` / `llama3.2:3b` (generation).

### 2. Configure environment

```bash
cp backend/.env.example backend/.env
# edit values as needed (SECRET_KEY is required in production)
```

### 3. Backend

```bash
cd backend
uv sync
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Tests

```bash
cd backend
uv run pytest -m "not integration"      # unit tests (mocked, used in CI)
uv run pytest -m integration            # real Qdrant + Ollama + Redis required
```

---

## Project structure

```text
hermes/
├── backend/
│   ├── src/
│   │   ├── agents/
│   │   │   ├── cache_check.py   # semantic cache gate (entry node)
│   │   │   ├── supervisor.py    # complexity classification + routing
│   │   │   ├── research.py      # retrieval + draft answer + citations
│   │   │   ├── synthesis.py     # multi-hop / cross-doc refinement
│   │   │   └── graph.py         # LangGraph wiring
│   │   ├── rag/
│   │   │   ├── factory.py       # shared retriever singleton
│   │   │   ├── retriever.py     # Qdrant hybrid search + RRF
│   │   │   ├── reranker.py      # cross-encoder reranking
│   │   │   ├── chunker.py       # parent-child chunking
│   │   │   └── cache.py         # Redis semantic cache
│   │   ├── ingestion/           # pdf / url / youtube loaders
│   │   ├── routers/             # FastAPI endpoints
│   │   ├── evaluation/          # RAGAS eval + golden dataset
│   │   ├── auth.py              # JWT auth
│   │   └── main.py             # FastAPI app (entrypoint: src.main:app)
│   └── tests/
├── frontend/                    # React + Vite UI
└── docker-compose.yml
```

---

## Scope

This is a focused implementation of agentic RAG. Intentionally **not** included: CRAG/Self-RAG reflection loops, long-term conversational memory, query expansion, and token-level streaming. These are possible future extensions, not current features.
