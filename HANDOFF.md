# HERMES — Complete Project Handoff Document

**Version:** 0.1.0 (post Resume-Honest Fix)  
**Last updated:** June 2026  
**Repository:** `/home/mhamd/HERMES-clean`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What HERMES Does](#2-what-hermes-does)
3. [System Architecture](#3-system-architecture)
4. [The Life of a Query](#4-the-life-of-a-query)
5. [Backend Components](#5-backend-components)
6. [Frontend](#6-frontend)
7. [Data Stores](#7-data-stores)
8. [API Reference](#8-api-reference)
9. [Configuration Reference](#9-configuration-reference)
10. [Performance & Quality Metrics](#10-performance--quality-metrics)
11. [Running the Project](#11-running-the-project)
12. [Testing](#12-testing)
13. [Recent Fixes (Resume-Honest Fix)](#13-recent-fixes-resume-honest-fix)
14. [Known Limitations & Out of Scope](#14-known-limitations--out-of-scope)
15. [Troubleshooting](#15-troubleshooting)
16. [File Map](#16-file-map)

---

## 1. Executive Summary

HERMES is an **agentic Retrieval-Augmented Generation (RAG) research assistant**. Users ingest documents (PDFs, web pages, YouTube transcripts), then ask natural-language questions and receive grounded answers with source citations.

The system is built as a portfolio-grade demonstration of modern RAG engineering:

| Capability | Implementation |
|---|---|
| Multi-source ingestion | PDF, URL (trafilatura), YouTube (transcript API) |
| Hybrid retrieval | Dense (`nomic-embed-text`) + sparse (BM25) in Qdrant, fused with RRF |
| Reranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) with configurable score threshold |
| Agent orchestration | LangGraph: cache gate → supervisor → research → synthesis |
| API & auth | FastAPI + JWT (Bearer tokens) |
| Semantic caching | Redis, cosine similarity ≥ 0.95 |
| Quality tracking | RAGAS metrics (faithfulness, relevancy, precision, recall) |

**Stack:** Python 3.11+ (FastAPI, LangGraph) · React 19 (Vite) · PostgreSQL · Redis · Qdrant · Ollama (local LLM/embeddings)

---

## 2. What HERMES Does

### User-facing workflow

1. **Register / log in** — JWT stored in browser `localStorage`.
2. **Ingest knowledge** — Upload a PDF, paste a URL, or submit a YouTube link. Content is chunked, embedded, and stored in Qdrant.
3. **Ask questions** — Chat interface sends queries through the LangGraph pipeline. Answers include citation cards linking to the source page, URL, or video timestamp.
4. **View analytics** — Dashboard shows query volume, cache hit rate, and latest RAGAS evaluation scores.

### The four resume claims (all verified live)

1. **Q&A over PDF/URL/YouTube with citations** — Ingestion loaders + citation metadata (`url`, `title`, `timestamp`) propagated to the UI.
2. **Hybrid dense+BM25, RRF, cross-encoder reranking** — Real implementation in `retriever.py` + `reranker.py`; weak contexts filtered by `MIN_RERANK_SCORE`.
3. **LangGraph retrieval + synthesis via JWT FastAPI** — `POST /api/research` runs the full agent graph behind Bearer auth.
4. **Semantic caching + RAGAS** — Cache hit bypasses supervisor/retrieval/generation; RAGAS scores documented honestly (~0.72–0.81).

---

## 3. System Architecture

### High-level diagram

```mermaid
flowchart TB
    subgraph Client
        UI[React Frontend<br>localhost:5173]
    end

    subgraph API["FastAPI (localhost:8000)"]
        Auth[JWT Auth]
        IngestRouter[Ingest Router]
        ResearchRouter[Research Router]
        EvalRouter[Eval Router]
    end

    subgraph Agents["LangGraph Pipeline"]
        CacheCheck[cache_check]
        Supervisor[supervisor]
        Research[research_agent]
        Synthesis[synthesis_agent]
    end

    subgraph RAG["RAG Layer"]
        Retriever[HermesRetriever<br>factory singleton]
        Reranker[Cross-Encoder]
        Chunker[Parent-Child Chunker]
        SemCache[Semantic Cache]
    end

    subgraph Infra["Infrastructure"]
        PG[(PostgreSQL<br>users, jobs, logs)]
        Redis[(Redis<br>semantic cache)]
        Qdrant[(Qdrant<br>hermes_docs)]
        Ollama[Ollama<br>embeddings + LLM]
    end

    UI -->|Bearer JWT| API
    API --> Agents
    CacheCheck -->|miss| Supervisor
    CacheCheck -->|hit| UI
    Supervisor --> Research
    Research -->|multi-hop| Synthesis
    Research --> Retriever
    Retriever --> Qdrant
    Retriever --> Reranker
    Retriever --> Ollama
    CacheCheck --> SemCache
    SemCache --> Redis
    Research --> Ollama
    IngestRouter --> Retriever
    IngestRouter --> PG
    ResearchRouter --> PG
```

### Technology choices

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI + Uvicorn | Async, OpenAPI docs, JWT middleware |
| Agents | LangGraph | Explicit state machine, conditional routing |
| Vector DB | Qdrant | Native hybrid dense+sparse, RRF fusion |
| Embeddings | Ollama `nomic-embed-text` | Local, 768-dim, no API cost |
| Sparse | fastembed `Qdrant/bm25` | Keyword/exact-match retrieval |
| Reranker | sentence-transformers cross-encoder | Joint query-passage scoring |
| LLM routing | LiteLLM | Unified interface + fallback chain |
| Cache | Redis + cosine similarity | Semantic dedup of paraphrased queries |
| Auth | python-jose + passlib bcrypt | Standard JWT pattern |
| Frontend | React 19 + Vite + Tailwind 4 | Fast dev, modern UI |
| Package mgmt | uv (backend), npm (frontend) | Fast Python dependency resolution |

---

## 4. The Life of a Query

This is the most important section for understanding how HERMES works end-to-end.

### Step-by-step flow

```
User types question in ResearchView
        │
        ▼
POST /api/research  { query, session_id? }
        │  (JWT validated, query logged to Postgres)
        ▼
┌───────────────────────────────────────────────────┐
│  LangGraph: START → cache_check                     │
│                                                     │
│  Embeds query, compares to Redis cache entries.     │
│  If similarity ≥ 0.95 AND cached answer exists:     │
│    → return cached answer + citations, END          │
│    (skips supervisor, retrieval, and generation)    │
└───────────────────────────────────────────────────┘
        │ cache miss
        ▼
┌───────────────────────────────────────────────────┐
│  supervisor_node                                    │
│                                                     │
│  llama3.2:3b classifies complexity:                 │
│    simple | multi_hop | synthesis                   │
│  Sets next_agent → research_agent                   │
└───────────────────────────────────────────────────┘
        ▼
┌───────────────────────────────────────────────────┐
│  research_node                                      │
│                                                     │
│  1. retriever.query(question, top_k=5)              │
│     a. Embed question (dense + sparse)              │
│     b. Qdrant hybrid prefetch (dense + BM25)        │
│     c. RRF fusion → candidate child chunks          │
│     d. Expand to parent_text from Qdrant payload    │
│     e. Cross-encoder rerank → top results           │
│  2. Filter contexts below MIN_RERANK_SCORE (0.35)    │
│  3. Build citations (source, url, title, timestamp) │
│  4. llama3.1:8b generates draft answer              │
│  5. If simple → cache result, END                   │
│     If multi_hop/synthesis → synthesis_agent        │
└───────────────────────────────────────────────────┘
        │ (optional)
        ▼
┌───────────────────────────────────────────────────┐
│  synthesis_node                                     │
│                                                     │
│  Refines draft for multi-hop / cross-doc queries.   │
│  Caches final answer, END.                          │
└───────────────────────────────────────────────────┘
        ▼
Response: { answer, citations[], model_used, cache_hit, session_id }
```

### Ingestion flow (separate path)

```
POST /api/ingest/{pdf|url|youtube}
        │
        ▼
Loader extracts text + metadata
        │
        ▼
HierarchicalChunker
  Parent chunks (~1000 chars) ── stored in Qdrant payload as parent_text
  Child chunks  (~200 chars)  ── embedded and indexed for retrieval
        │
        ▼
Embed: dense (nomic-embed-text) + sparse (BM25)
        │
        ▼
Upsert to Qdrant collection "hermes_docs"
        │
        ▼
Log IngestionJob in Postgres (status, chunks_stored)
```

### Parent-child chunking (critical design)

Small **child** chunks (~200 chars) are used for precise vector matching. When a child matches, the larger **parent** chunk (~1000 chars) is returned to the LLM for richer context. Parent text is persisted in the Qdrant point payload (`parent_text` field) so expansion works across process restarts and the shared retriever singleton.

---

## 5. Backend Components

### 5.1 Entry point

| File | Role |
|---|---|
| `src/main.py` | FastAPI app, CORS, lifespan (creates DB tables), mounts routers |

**Start command:** `uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload`

### 5.2 Authentication (`src/auth.py`)

- JWT tokens, HS256, 24-hour expiry.
- `SECRET_KEY` from env; startup **fails** if default key used with `ENV=production`.
- `get_current_user` dependency validates Bearer token on protected routes.

### 5.3 Database (`src/db.py`)

Async SQLAlchemy with `asyncpg`. Three tables — see [Section 7](#7-data-stores).

### 5.4 LangGraph agents (`src/agents/`)

| File | Node | What it does |
|---|---|---|
| `graph.py` | — | Wires the state machine, exports `run_research()` and `get_graph()` |
| `state.py` | — | `ResearchState` TypedDict + `Citation` type |
| `cache_check.py` | `cache_check` | **Entry node.** Semantic cache lookup; hit → END, miss → supervisor |
| `supervisor.py` | `supervisor` | Classifies query complexity via `llama3.2:3b` |
| `research.py` | `research_agent` | Retrieves, reranks, generates draft answer + citations |
| `synthesis.py` | `synthesis_agent` | Refines multi-hop answers, caches result |

**Graph topology:**

```
START → cache_check → [END | supervisor → research_agent → (synthesis_agent) → END]
```

Routing is conditional on `state["next_agent"]`. `MemorySaver` checkpointer enables `session_id` threading (used by frontend, but multi-turn memory is not a product feature).

### 5.5 RAG layer (`src/rag/`)

| File | Class/Function | What it does |
|---|---|---|
| `factory.py` | `get_retriever()` | **Single process-wide singleton.** All code paths (ingest, research, eval) share one `HermesRetriever` instance. |
| `retriever.py` | `HermesRetriever` | Hybrid search (dense+sparse prefetch, RRF fusion), parent expansion, delegates to reranker. **No cache logic** (moved to graph layer). |
| `reranker.py` | `rerank()` | Lazy-loads `cross-encoder/ms-marco-MiniLM-L-6-v2`, scores query×passage pairs, returns top_k sorted by `reranker_score`. |
| `chunker.py` | `HierarchicalChunker` | Parent (1000 char) / child (200 char) splitting via LangChain `RecursiveCharacterTextSplitter`. |
| `cache.py` | `SemanticCache` | Redis-backed. Embeds query, scans cached entries, returns hit if cosine ≥ 0.95 (instant return at ≥ 0.99). TTL 24h. |

**Qdrant collection schema (`hermes_docs`):**

| Field | Type | Purpose |
|---|---|---|
| `dense` vector | 768-dim cosine | Semantic similarity |
| `sparse` vector | BM25 indices/values | Keyword matching |
| `text` | string | Child chunk text |
| `parent_id` | string | Link to parent chunk |
| `parent_text` | string | Full parent context for LLM |
| `source`, `url`, `title`, `page_num`, `timestamp`, `type` | metadata | Citation rendering |

### 5.6 Ingestion loaders (`src/ingestion/`)

| Loader | Library | Metadata keys |
|---|---|---|
| `pdf_loader.py` | pypdf (+ unstructured fallback for scanned PDFs) | `source` (filename), `page_num`, `type: pdf` |
| `url_loader.py` | trafilatura | `source`, `url`, `title`, `page_num`, `type: url` |
| `youtube_loader.py` | youtube-transcript-api + yt-dlp (title) | `source`, `url` (with `&t=` timestamp), `title`, `timestamp`, `type: youtube` |

Failed URL/YouTube ingest returns `status: "failed"` and the API responds with HTTP 422 (job marked `error` in Postgres).

### 5.7 LLM providers (`src/llm/providers.py`)

LiteLLM wrapper with model map and automatic fallback:

| Complexity | Primary model | Fallback chain |
|---|---|---|
| `simple` | `ollama/llama3.1:8b` | → Groq instant |
| `complex` | `groq/llama-3.3-70b-versatile` | → Gemini → Ollama 8b |
| `long_doc` | `gemini/gemini-2.0-flash-exp` | — |
| `classify` | `ollama/llama3.2:3b` | — (supervisor only) |

Requires `GROQ_API_KEY` / `GEMINI_API_KEY` only if you want hosted fallbacks. Local-only mode works with Ollama alone.

### 5.8 Evaluation (`src/evaluation/`)

| File | Role |
|---|---|
| `golden_dataset.py` | 10 Q&A pairs on RAG topics (ground truth for RAGAS) |
| `ragas_eval.py` | Full eval pipeline: flush state → ingest 3 URLs → run 10 questions through graph → score with local Ollama judge |

**Trigger eval:**
- CLI: `uv run python -m src.evaluation.ragas_eval`
- API: `POST /api/eval/run` (JWT; optionally restricted via `EVAL_ADMIN_EMAILS`)

### 5.9 Routers (`src/routers/`)

See [Section 8](#8-api-reference).

### 5.10 Utilities

| File | Purpose |
|---|---|
| `src/verify.py` | One-shot connectivity check for Ollama, Qdrant, Redis, Postgres, Groq |

---

## 6. Frontend

**Location:** `frontend/`  
**Dev server:** `npm run dev` → `http://localhost:5173`  
**API client:** `src/api/client.js` — Axios, base URL `http://localhost:8000/api`, auto-attaches Bearer token.

### Routes

| Path | Page | Function |
|---|---|---|
| `/auth` | `LoginRegister` | Register or log in |
| `/` | `ResearchView` | Main chat — ask questions, view answers + citations |
| `/knowledge-base` | `KnowledgeBaseView` | Ingest URLs, YouTube links, PDFs |
| `/analytics` | `AnalyticsView` | Query stats + RAGAS radar/bar charts |

All routes except `/auth` require authentication (`PrivateRoute` wrapper in `App.jsx`).

### Key UI behaviors

- **ResearchView:** Persists chat history and `session_id` in `sessionStorage`. Renders markdown answers. Shows cache-hit badge and model name. Citation cards display source links (web/YouTube) or page numbers (PDF).
- **KnowledgeBaseView:** Auto-detects YouTube URLs and routes to the correct ingest endpoint. Drag-and-drop PDF upload.
- **AnalyticsView:** Fetches `/api/eval/dashboard`, renders Recharts visualizations.

### Auth flow

1. User registers/logs in → JWT returned.
2. `AuthContext` stores token in `localStorage` key `hermes_access_token`.
3. Axios interceptor attaches `Authorization: Bearer <token>` to every request.

---

## 7. Data Stores

### PostgreSQL (application data)

**Connection:** `postgresql+asyncpg://hermes:hermes_pass@localhost:5432/hermes_db`

#### `users`
| Column | Type | Notes |
|---|---|---|
| id | SERIAL PK | |
| email | VARCHAR(255) UNIQUE | Login identifier |
| hashed_password | VARCHAR(255) | bcrypt |
| is_active | BOOLEAN | Default true |
| created_at | TIMESTAMP | UTC |

#### `ingestion_jobs`
| Column | Type | Notes |
|---|---|---|
| id | SERIAL PK | |
| user_id | FK → users | |
| source_type | VARCHAR(20) | `pdf`, `url`, `youtube` |
| source_ref | TEXT | Filename or URL |
| status | VARCHAR(20) | `pending`, `ok`, `error` |
| chunks_stored | INTEGER | Child chunks written to Qdrant |
| error_msg | TEXT | Populated on failure |
| created_at | TIMESTAMP | |

#### `query_logs`
| Column | Type | Notes |
|---|---|---|
| id | SERIAL PK | |
| user_id | FK → users | |
| query | TEXT | User question |
| answer | TEXT | Final answer string |
| model_used | VARCHAR(100) | e.g. `ollama/llama3.1:8b` |
| cache_hit | BOOLEAN | Whether semantic cache was used |
| faithfulness | FLOAT | Reserved (not auto-populated) |
| created_at | TIMESTAMP | |

### Qdrant (vector store)

- **Collection:** `hermes_docs`
- **URL:** `http://localhost:6333`
- Hybrid vectors: dense (768-dim cosine) + sparse (BM25)
- Payload includes `parent_text` for parent-child expansion
- Dashboard: `http://localhost:6333/dashboard`

### Redis (semantic cache)

- **URL:** `redis://localhost:6379`
- Key pattern: `hermes:cache:index` (list of JSON entries with query embedding + result)
- Each entry: `{ query, embedding, result: { answer, citations, contexts } }`
- TTL: 24 hours

---

## 8. API Reference

**Base URL:** `http://localhost:8000`  
**Auth:** Bearer JWT on all `/api/*` routes except auth endpoints.  
**OpenAPI docs:** `http://localhost:8000/docs`

### Health

```
GET /health
→ { "status": "ok", "service": "hermes" }
```

### Auth

```
POST /api/auth/register
Body: { "email": "user@example.com", "password": "pass1234" }
→ 201 { "access_token": "eyJ..." }

POST /api/auth/login
Body: { "email": "user@example.com", "password": "pass1234" }
→ 200 { "access_token": "eyJ..." }
```

### Ingestion (Bearer required)

```
POST /api/ingest/url
Body: { "url": "https://example.com/article" }
→ { "job_id": 1, "status": "ok", "chunks_stored": 42 }
→ 422 if content extraction fails

POST /api/ingest/youtube
Body: { "url": "https://youtube.com/watch?v=VIDEO_ID" }
→ { "job_id": 2, "status": "ok", "chunks_stored": 18 }
→ 422 if transcript unavailable

POST /api/ingest/pdf
Body: multipart/form-data, field "file" (.pdf only)
→ { "job_id": 3, "status": "ok", "chunks_stored": 55 }
→ 400 if not a PDF
```

### Research (Bearer required)

```
POST /api/research
Body: { "query": "What is RAG?", "session_id": "optional-uuid" }
→ {
    "answer": "...",
    "citations": [
      {
        "source": "https://en.wikipedia.org/...",
        "title": "Retrieval-augmented generation",
        "url": "https://en.wikipedia.org/...",
        "page_num": null,
        "timestamp": null,
        "context": "...",
        "score": 5.651
      }
    ],
    "model_used": "ollama/llama3.1:8b",
    "cache_hit": false,
    "session_id": "uuid"
  }
```

### Evaluation (Bearer required)

```
POST /api/eval/run?n_questions=10
→ { "status": "started", "n_questions": 10 }
   (background job; takes 15–20 min with local Ollama judge)
→ 403 if EVAL_ADMIN_EMAILS is set and user not in list

GET /api/eval/status
→ { "running": false }

GET /api/eval/dashboard
→ {
    "total_queries": 42,
    "cache_hit_rate": 0.3333,
    "ragas": { "faithfulness": 0.7499, ... }
  }

GET /api/eval/golden
→ raw eval_report.json contents
```

---

## 9. Configuration Reference

Copy `backend/.env.example` → `backend/.env` and edit.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://hermes:hermes_pass@localhost:5432/hermes_db` | Postgres connection |
| `REDIS_URL` | `redis://localhost:6379` | Semantic cache |
| `QDRANT_URL` | `http://localhost:6333` | Vector database |
| `QDRANT_API_KEY` | (empty) | Only for Qdrant Cloud |
| `OLLAMA_API_BASE` | `http://localhost:11434` | Embeddings + local LLM |
| `GROQ_API_KEY` | (empty) | Optional hosted fallback |
| `GEMINI_API_KEY` | (empty) | Optional hosted fallback |
| `SECRET_KEY` | (must change in prod) | JWT signing key |
| `ENV` | `development` | Set `production` to enforce SECRET_KEY |
| `MIN_RERANK_SCORE` | `0.35` | Cross-encoder threshold for keeping contexts |
| `EVAL_ADMIN_EMAILS` | (empty) | Comma-separated allow-list for eval runs |

### Hardcoded tuning defaults (not env-configurable)

| Parameter | Value | Location |
|---|---|---|
| Cache similarity threshold | 0.95 | `rag/cache.py` |
| Instant cache return | ≥ 0.99 | `rag/cache.py` |
| Cache TTL | 24 hours | `rag/cache.py` |
| Parent chunk size / overlap | 1000 / 100 chars | `rag/chunker.py` |
| Child chunk size / overlap | 200 / 20 chars | `rag/chunker.py` |
| Embedding model | `nomic-embed-text` | `retriever.py` |
| Embedding dimensions | 768 | `retriever.py` |
| Sparse model | `Qdrant/bm25` | `retriever.py` |
| Reranker model | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `reranker.py` |
| Research retrieval top_k | 5 | `agents/research.py` |
| Hybrid prefetch multiplier | top_k × 4 | `retriever.py` |
| JWT expiry | 24 hours | `auth.py` |
| YouTube chunk window | 120 seconds | `youtube_loader.py` |

---

## 10. Performance & Quality Metrics

### Live end-to-end benchmarks (verified June 2026)

Measured on WSL2 Ubuntu with local Ollama (CPU-bound; GPU would be faster):

| Scenario | Latency | Notes |
|---|---|---|
| URL ingest (Wikipedia RAG article) | ~14s | 115 child chunks stored |
| Research query (cache miss) | ~150s | Includes supervisor classify + hybrid retrieval + rerank + llama3.1:8b generation |
| Research query (cache hit, paraphrase) | **~0.27s** | Semantic cache bypasses entire pipeline |
| Integration test: parent expansion | ~6 min total | Includes first-time BM25 + reranker model download |
| Integration test: cache round-trip | < 1s | Redis + embedding similarity only |

**Cache speedup:** ~560× faster on semantic cache hit (150s → 0.27s).

### RAGAS evaluation scores

**Source:** `backend/eval_report.json` (10 questions, local Ollama `llama3.1:8b` judge)

| Metric | Score | Internal target |
|---|---|---|
| Faithfulness | **0.7499** | ≥ 0.83 |
| Answer Relevancy | **0.7172** | ≥ 0.80 |
| Context Precision | **0.7222** | ≥ 0.80 |
| Context Recall | **0.8071** | ≥ 0.80 |

These are honest, measured scores — not aspirational targets. Context recall is the strongest metric; faithfulness and relevancy have room for improvement (better chunking, more diverse KB, or a stronger judge model).

### Integration test results

```
tests/test_integration_retriever.py::test_parent_expansion_returns_more_than_child  PASSED
tests/test_integration_retriever.py::test_semantic_cache_round_trip               PASSED
```

Proves: (1) parent context returned to LLM is larger than the matched child chunk, (2) semantic cache stores and retrieves answers for paraphrased queries.

### Unit tests (CI)

11 tests, all mocked (no live services required):

```
test_auth.py       — 5 tests (register, login, duplicate, wrong password, protected route)
test_ingest.py     — 3 tests (URL success, auth required, PDF type check)
test_research.py   — 3 tests (research response, auth required, health)
```

---

## 11. Running the Project

### Prerequisites

- Docker Desktop with WSL2 integration (for Postgres, Redis, Qdrant)
- Ollama with models: `nomic-embed-text`, `llama3.1:8b`, `llama3.2:3b`
- `uv` (Python package manager)
- Node.js 18+ (via nvm recommended)

### Step-by-step

```bash
# 1. Infrastructure
cd /home/mhamd/HERMES-clean
docker compose up -d postgres redis qdrant

# 2. Ollama (if not already running)
# Option A: docker compose --profile local up -d ollama
# Option B: native install (see ~/ollama-local/ from setup session)
ollama pull nomic-embed-text
ollama pull llama3.1:8b
ollama pull llama3.2:3b

# 3. Backend config
cp backend/.env.example backend/.env
# edit backend/.env if needed

# 4. Backend
cd backend
uv sync
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Frontend (separate terminal)
cd frontend
npm install
npm run dev
# → http://localhost:5173

# 6. Verify connectivity (optional)
cd backend && uv run python -m src.verify
```

### Docker Compose services

| Service | Image | Port | Profile |
|---|---|---|---|
| postgres | postgres:15 | 5432 | default |
| redis | redis:7-alpine | 6379 | default |
| qdrant | qdrant/qdrant | 6333, 6334 | default |
| ollama | ollama/ollama | 11434 | `local` |

Start Ollama container: `docker compose --profile local up -d ollama`

---

## 12. Testing

```bash
cd backend

# Unit tests (mocked, runs in CI)
uv run pytest -m "not integration" -v

# Integration tests (requires live Qdrant + Ollama + Redis)
uv run pytest -m integration -v

# All tests
uv run pytest -v
```

**CI** (`.github/workflows/ci.yml`): Runs on push/PR to `main`. Spins up Postgres + Redis service containers. Runs `uv run pytest -m "not integration"`. Does **not** run RAGAS or integration tests (Ollama dependency too heavy).

---

## 13. Recent Fixes (Resume-Honest Fix)

These fixes were applied to make the codebase truthfully match the portfolio description. A developer inheriting this project should understand what was broken and what changed.

### Critical bugs fixed (Phase 0)

| Bug | Impact | Fix |
|---|---|---|
| **Dual retriever singleton** | Ingest wrote to retriever A's in-memory parent store; queries used retriever B (empty store). Parent expansion never fired in the running server. | `src/rag/factory.py` — single shared `get_retriever()` |
| **Parent text in-memory only** | Lost on restart; invisible across singletons | `parent_text` persisted in Qdrant payload on ingest, read in `query()` |
| **Silent ingest failures** | URL/YouTube loader returned `status: "failed"` but API marked job `ok` | HTTP 422 on failed ingest |
| **Broken retriever cache** | `retriever.query()` cached contexts only (no answer); research node expected `answer` key | Cache removed from retriever; lives only at graph layer |
| **No-op rerank filter** | `> -100` threshold let everything through | `MIN_RERANK_SCORE=0.35` (env-configurable) |

### Feature alignment (Phase 1)

| Change | Detail |
|---|---|
| Citation metadata | `url`, `title`, `timestamp` propagated loaders → Qdrant → API → frontend |
| Cache before supervisor | `cache_check` node is graph entry point; hits skip classification + retrieval + generation |
| RAGAS wired honestly | Uses shared retriever; `POST /api/eval/run` endpoint; README cites actual scores |
| Docker compose | Qdrant added; n8n removed; optional Ollama profile |
| Integration tests | Parent expansion + cache round-trip, marked `@integration`, excluded from CI |

### Cleanup (Phase 2)

- README rewritten to match four resume bullets (removed "enterprise-grade", "mathematically verified", fake memory claims).
- Deleted dead code: `backend/main.py` stub, `llm/router.py` duplicate classifier, fake SSE streaming endpoint.
- Production `SECRET_KEY` guard added.
- `EVAL_ADMIN_EMAILS` allow-list for eval endpoint.

---

## 14. Known Limitations & Out of Scope

### Current limitations

| Limitation | Detail |
|---|---|
| **No multi-turn memory** | `MemorySaver` checkpointer exists for `session_id` threading, but the research node does not meaningfully use conversation history for retrieval. Frontend persists chat UI state only. |
| **CPU-bound generation** | Without GPU passthrough to Ollama in WSL, first query takes ~150s. GPU (RTX 4050) dramatically improves this. |
| **No streaming** | Answers return as complete JSON responses. No token-level SSE. |
| **Single collection** | All users share one Qdrant collection (`hermes_docs`). No per-user ACL or document isolation. |
| **Cache is process-local stats** | `SemanticCache.hits/misses` counters are in-memory; only query_logs in Postgres tracks cache hits persistently. |
| **RAGAS not in CI** | Eval requires running Ollama judge (~15–20 min). Run manually or via API. |
| **Old Qdrant points** | Documents ingested before the parent_text fix lack `parent_text` in payload. Re-ingest for full parent expansion (fallback to child text still works). |

### Explicitly not built (future v2 candidates)

- CRAG / Self-RAG reflection loops
- Long-term conversational memory (Mem0, Postgres checkpointer)
- Query expansion / HyDE
- Real SSE token streaming
- OpenRouter migration
- Per-user document ACL
- Langfuse observability
- 50+ golden question CI gate
- Adaptive routing beyond simple/multi_hop/synthesis

---

## 15. Troubleshooting

### `docker info` fails in WSL

Reinstall Docker Desktop via winget. Enable WSL integration for Ubuntu in Docker Desktop settings → Resources → WSL Integration.

### Ollama not reachable

```bash
curl http://localhost:11434/api/tags
# Should return {"models": [...]}

# If empty, pull required models:
ollama pull nomic-embed-text
ollama pull llama3.1:8b
ollama pull llama3.2:3b
```

If running native install without sudo: binary at `~/ollama-local/bin/ollama serve`.

### Qdrant connection refused

```bash
docker compose up -d qdrant
curl http://localhost:6333/collections
```

### Research returns "No relevant information found"

1. Check that documents were ingested: `curl http://localhost:6333/collections/hermes_docs`
2. Verify Ollama embeddings work: `curl -X POST http://localhost:11434/api/embed -d '{"model":"nomic-embed-text","input":"test"}'`
3. Lower `MIN_RERANK_SCORE` temporarily (e.g. `0.1`) if reranker is filtering too aggressively.
4. Re-ingest documents if they predate the `parent_text` fix.

### Cache hit returns stale/wrong answer

```bash
docker exec hermes-clean-redis-1 redis-cli FLUSHDB
```

### SECRET_KEY error on startup

Set a unique `SECRET_KEY` in `backend/.env`, or keep `ENV=development`.

### Slow first query

Expected on CPU. Cross-encoder reranker (~80MB) and BM25 model download on first use add latency. Subsequent queries are faster. GPU passthrough to Ollama is the biggest win.

### Integration tests skipped

Tests auto-skip if Qdrant or Ollama are unreachable. Ensure both are running before `pytest -m integration`.

---

## 16. File Map

```
HERMES-clean/
├── HANDOFF.md                  ← this document
├── README.md                   ← public-facing project description
├── docker-compose.yml          ← Postgres, Redis, Qdrant, Ollama (profile)
├── fix.md                      ← internal roadmap notes (not cited in README)
│
├── backend/
│   ├── .env.example            ← configuration template
│   ├── pyproject.toml          ← Python deps + pytest config
│   ├── eval_report.json        ← latest RAGAS aggregate scores
│   ├── eval_details.json       ← per-question RAGAS breakdown
│   │
│   ├── src/
│   │   ├── main.py             ← FastAPI entrypoint
│   │   ├── auth.py             ← JWT authentication
│   │   ├── db.py               ← SQLAlchemy models
│   │   ├── verify.py           ← service connectivity checker
│   │   │
│   │   ├── agents/             ← LangGraph pipeline
│   │   │   ├── graph.py        ← state machine wiring
│   │   │   ├── state.py        ← ResearchState + Citation types
│   │   │   ├── cache_check.py  ← semantic cache gate (entry)
│   │   │   ├── supervisor.py   ← complexity classifier
│   │   │   ├── research.py     ← retrieve + generate + cite
│   │   │   └── synthesis.py    ← multi-hop refinement
│   │   │
│   │   ├── rag/                ← retrieval layer
│   │   │   ├── factory.py      ← shared retriever singleton
│   │   │   ├── retriever.py    ← Qdrant hybrid search + RRF
│   │   │   ├── reranker.py     ← cross-encoder scoring
│   │   │   ├── chunker.py      ← parent-child chunking
│   │   │   └── cache.py        ← Redis semantic cache
│   │   │
│   │   ├── ingestion/          ← document loaders
│   │   │   ├── pdf_loader.py
│   │   │   ├── url_loader.py
│   │   │   └── youtube_loader.py
│   │   │
│   │   ├── llm/
│   │   │   └── providers.py    ← LiteLLM model map + fallbacks
│   │   │
│   │   ├── routers/            ← API endpoints
│   │   │   ├── auth.py
│   │   │   ├── ingest.py
│   │   │   ├── research.py
│   │   │   └── eval.py
│   │   │
│   │   └── evaluation/         ← RAGAS quality tracking
│   │       ├── golden_dataset.py
│   │       └── ragas_eval.py
│   │
│   └── tests/
│       ├── conftest.py         ← fixtures (mocked retriever, auth token)
│       ├── test_auth.py
│       ├── test_ingest.py
│       ├── test_research.py
│       └── test_integration_retriever.py  ← live stack tests
│
└── frontend/
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx             ← routing + auth guard
        ├── api/client.js       ← Axios HTTP client
        ├── context/AuthContext.jsx
        ├── components/
        │   ├── Sidebar.jsx
        │   └── Auth/LoginRegister.jsx
        └── pages/
            ├── ResearchView.jsx      ← chat UI
            ├── KnowledgeBaseView.jsx ← ingestion UI
            └── AnalyticsView.jsx     ← metrics dashboard
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│  HERMES Quick Start                                     │
├─────────────────────────────────────────────────────────┤
│  Infra:    docker compose up -d postgres redis qdrant │
│  Ollama:   nomic-embed-text, llama3.1:8b, llama3.2:3b │
│  Backend:  cd backend && uv run uvicorn src.main:app    │
│  Frontend: cd frontend && npm run dev                   │
│  UI:       http://localhost:5173                        │
│  API docs: http://localhost:8000/docs                   │
│  Qdrant:   http://localhost:6333/dashboard              │
├─────────────────────────────────────────────────────────┤
│  Tests:    uv run pytest -m "not integration"         │
│  Live:     uv run pytest -m integration                 │
│  Eval:     uv run python -m src.evaluation.ragas_eval   │
│  Health:   uv run python -m src.verify                  │
└─────────────────────────────────────────────────────────┘
```

---

*This document is the authoritative handoff reference for the HERMES project. For the public-facing summary, see `README.md`. For internal roadmap ideas, see `fix.md` (not production scope).*
