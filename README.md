# Hermes — Agentic RAG Research Assistant

> Agentic RAG over PDFs, URLs, and YouTube with grounded citations. Hybrid dense + BM25 retrieval with Reciprocal Rank Fusion and cross-encoder reranking, orchestrated as a LangGraph retrieval + synthesis pipeline behind a JWT-secured FastAPI, with semantic caching and RAGAS-based quality tracking.

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-FF4F00?style=for-the-badge)
![Qdrant](https://img.shields.io/badge/Qdrant-1A1A1A?style=for-the-badge)
![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor=white)

---

## Portfolio positioning

HERMES is the **Applied AI / agentic RAG** portfolio star (foundation-model stack, hybrid retrieval, LangGraph agents, semantic cache, honest RAGAS).
---

## What it does

- **Ingest** PDFs, web pages, and YouTube transcripts into a hybrid vector store.
- **Ask** natural-language questions and get answers grounded in the ingested sources, with citation cards that link back to the page, URL, or video timestamp.
- **Retrieve** with hybrid dense + sparse search, fuse with RRF, and rerank with a cross-encoder before any LLM call.
- **Cache** semantically similar questions to skip the pipeline on repeats.
- **Track** answer quality with RAGAS (faithfulness, answer relevancy, context precision/recall).
- **Multi-turn memory:** last N turns in Postgres feed a query-rewrite step so follow-ups retrieve with prior entities (not long-term user profiles). Disable with `HERMES_MULTI_TURN=0`.
- **Named tools + traces:** research always calls `hybrid_search` first; API/UI return `tool_trace` (empty-KB answers do not invent sources).
- **Workspace-scoped KB:** ingest stamps `user_id` on Qdrant payloads; query filters by the JWT user. Re-ingest after upgrading from pre-ACL collections.
- **Real token streaming:** `POST /api/research/stream` emits SSE token events (JSON `POST /api/research` still available).
- **MCP:** `uv run python -m src.mcp.server` exposes `hermes_search` / `hermes_research` over stdio for MCP Inspector.
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
        Research --> Embed[BGE-m3 dense<br>+ BM25 sparse]
        Embed --> Qdrant[(Qdrant)]
        Qdrant -->|dense + sparse prefetch| RRF[RRF fusion]
        RRF --> Rerank[Cross-encoder rerank<br>BAAI/bge-reranker-v2-m3]
        Rerank --> Parent[Parent-context expansion]
    end

    Parent --> Draft[Draft answer + citations]
    Draft -->|simple| UI
    Draft -->|multi-hop / synthesis| Synthesis[synthesis_agent]
    Synthesis --> UI
```

The LangGraph pipeline is: `START → cache_check → [END on hit | supervisor → query_rewrite → research_agent → (synthesis_agent) → END]`. The supervisor classifies query complexity; simple queries finish after research, while multi-hop/synthesis queries get a second synthesis pass.

---

## Core components

### Hybrid retrieval (`backend/src/rag/retriever.py`)
- **Dense** embeddings via `BAAI/bge-m3` (local SentenceTransformer, 1024-dim) capture semantic meaning.
- **Sparse** BM25 vectors via `fastembed` capture exact lexical / keyword matches.
- Qdrant runs both as prefetches and fuses them with **Reciprocal Rank Fusion (RRF)**.
- **Parent-child chunking** (`chunker.py`): small child chunks are indexed for precise retrieval, but the larger parent chunk is returned to the LLM for context. Parent text is persisted in the Qdrant payload so expansion works across processes and restarts.

### Cross-encoder reranking (`backend/src/rag/reranker.py`)
Vector similarity scores candidates independently. The `BAAI/bge-reranker-v2-m3` cross-encoder rescores the query against each candidate jointly, and contexts below `MIN_RERANK_SCORE` (default `0.0`) are dropped before the prompt is built.

### Semantic cache (`backend/src/rag/cache.py`)
A Redis-backed cache embeds each query and compares against stored queries by cosine similarity (threshold `0.95`). On a hit, `cache_check` (the first graph node) returns the stored answer and **bypasses retrieval and generation** — an honest latency optimization, not a bypass of "the entire pipeline" before classification.

### LangGraph agents (`backend/src/agents/`)
- `cache_check.py` — semantic cache gate (entry node).
- `supervisor.py` — classifies complexity (simple / multi-hop / synthesis).
- `query_rewrite.py` — expands follow-ups using prior turns before retrieval.
- `research.py` — forced `hybrid_search` tool, rerank, draft answer + citations + `tool_trace`.
- `synthesis.py` — refines multi-hop / cross-document answers.
- `stream_research.py` — real token SSE event generator.

### FastAPI + JWT (`backend/src/`)
- `POST /api/auth/register`, `POST /api/auth/login` — JWT auth.
- `POST /api/research` — JSON Q&A (Bearer token required).
- `POST /api/research/stream` — SSE token stream (same auth/body).
- `POST /api/ingest/pdf`, `/api/ingest/url`, `/api/ingest/youtube` — ingestion (stamps `user_id`).
- `GET /api/eval/dashboard`, `POST /api/eval/run` — RAGAS reporting.

### MCP (`backend/src/mcp/server.py`)
```bash
cd backend && HERMES_MCP_USER_ID=<user id> uv run python -m src.mcp.server
# Connect MCP Inspector via stdio — tools: hermes_search, hermes_research
```
Both tools run under the ACL of `HERMES_MCP_USER_ID`, which is required — stdio MCP carries no request-level identity, so if it is unset both tools return an error rather than running unscoped. `hermes_research` has no JWT; do not expose it unauthenticated in production.
---

## Evaluation (RAGAS)

The latest run is recorded in `backend/eval_report.json` (experiment `clean_slate_winning`, 2026-07-14). It scored **20 questions**; the judge is recorded as `levers.judge` = `openrouter/google/gemini-2.5-flash-lite`:

| Metric | Score |
|---|---|
| Faithfulness | 1.0 |
| Answer relevancy | 0.8711 |
| Context precision | 0.8327 |
| Context recall | 1.0 |

Levers recorded for that run, in full, from `levers` in the same file:

| Lever | Value | | Lever | Value |
|---|---|---|---|---|
| `MIN_RERANK_SCORE` | 0.0 | | `CHILD_CHUNK_SIZE` | 150 |
| `RETRIEVAL_CANDIDATES` | 50 | | `CHILD_CHUNK_OVERLAP` | 50 |
| `HERMES_MULTI_QUERY` | 1 | | `EMBED_MODEL` | bge-m3 |
| `HERMES_CRAG_LITE` | 0 | | `RERANK_MODEL` | BAAI/bge-reranker-v2-m3 |
| `HERMES_GRAPH_RAG` | 0 | | `RAGAS_MAX_WORKERS` | 4 |
| `CONTEXT_PACK_TOP_K` | 5 | | `RAGAS_BUILD_WORKERS` | 1 |
| `HERMES_SIMPLE_MODEL` | complex | | `LLM_PROVIDER` | openrouter |
| `CHUNK_STRATEGY` | fixed_large | | `OLLAMA_API_BASE` | http://localhost:11434 |


**What this number is, stated plainly:**

- It is a **20-question smoke test**, not a benchmark. Twenty questions is far too small a sample to estimate a quality metric.
- **13 of the 20 questions are answerable from `backend/eval/kb/hermes_architecture.md`** — a file authored for this repository, describing HERMES' own behaviour. Only the remaining 7 test anything beyond "can the system read back a document about itself".
- **The results are not independent.** The questions, the corpus and the system under test were all written by the same author, and the tuned defaults above were selected by running this same evaluation. There is no held-out set and no external judge of the questions themselves, so the scores measure self-consistency, not general retrieval quality. Treat 1.0 faithfulness as "no answer contradicted its own context", not as a quality claim.

Run an evaluation:

```bash
# via API (background job, JWT required, and EVAL_ADMIN_EMAILS must list you —
# an empty allow-list denies everyone)
curl -X POST localhost:8000/api/eval/run -H "Authorization: Bearer <token>"

# or directly
cd backend && uv run python -m src.evaluation.ragas_eval
```

RAGAS is not part of CI (it needs a judge model and takes several minutes). CI runs the mocked unit tests; an integration test exercises the real retriever locally.

---

## Setup

### 1. Start infrastructure

```bash
docker compose up -d postgres redis qdrant
# optional: run a local LLM/embedding server too
# docker compose --profile local up -d
```

Generation goes through LiteLLM. By default the tiers are Groq-hosted `groq/openai/gpt-oss-120b` (complex/long-doc) and `groq/openai/gpt-oss-20b` (simple/classify), so set `GROQ_API_KEY`. Override any tier with `GROQ_MODEL_SIMPLE`, `GROQ_MODEL_COMPLEX`, `GROQ_MODEL_LONG_DOC`, `GROQ_MODEL_CLASSIFY`, `GROQ_MODEL_OFFLINE`, or set `LLM_PROVIDER=openrouter` with `OPENROUTER_MODEL_*`. Embeddings and reranking are local (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`) and need no API key. Set `OLLAMA_API_BASE` if you want the `offline` tier to stay on a local Ollama server.

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

| Suite | Command | Requires |
|---|---|---|
| Unit (CI) | `cd backend && uv run pytest -m "not integration"` | Postgres (+ Redis optional) |
| Integration | `cd backend && uv run pytest -m integration` | Qdrant + Ollama + Redis |
| RAGAS | `cd backend && uv run python -m src.evaluation.ragas_eval` | Full stack + judge model |

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
│   │   │   ├── query_rewrite.py # multi-turn retrieval rewrite
│   │   │   ├── research.py      # hybrid_search tool + draft + citations
│   │   │   ├── synthesis.py     # multi-hop / cross-doc refinement
│   │   │   ├── stream_research.py
│   │   │   └── graph.py         # LangGraph wiring
│   │   ├── tools/               # hybrid_search, fetch_parent, web_fetch
│   │   ├── mcp/                 # MCP stdio server
│   │   ├── rag/
│   │   │   ├── factory.py       # shared retriever singleton
│   │   │   ├── retriever.py     # Qdrant hybrid search + RRF + ACL filter
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

This is a focused implementation of agentic RAG. Intentionally **not** included: CRAG/Self-RAG reflection loops, long-term user-profile personalization, Juris-class RBAC/SSO/SCIM/WORM, and competing with enterprise legal platforms. Workspace-scoped KB filtering is not a full RBAC platform.
