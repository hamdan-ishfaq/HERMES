# HERMES — Complete Code Walkthrough

> **Audience:** Someone new to the project who wants to understand every file, function, and how data moves through the system.  
> **Companion docs:** `README.md` (public summary) · `HANDOFF.md` (operations reference)

---

## How to use this document

1. Read [Section 1](#1-the-big-picture) for the 30-second mental model.
2. Follow [Section 2](#2-complete-data-flow-two-main-paths) for ingest vs query flows with function names.
3. Use [Section 3](#3-backend-file-by-file) as a dictionary — open each file alongside this section.
4. Use [Section 4](#4-frontend-file-by-file) for the React UI.
5. [Section 5](#5-tracing-a-query-in-the-code) walks one real query line-by-line through the call stack.

---

## 1. The big picture

HERMES is a **RAG (Retrieval-Augmented Generation) app**:

```
User uploads documents  →  stored as vectors in Qdrant
User asks a question    →  find relevant chunks  →  LLM writes answer  →  show citations
```

Three layers:

| Layer | Folder | Job |
|---|---|---|
| **UI** | `frontend/src/` | Login, chat, upload docs, view metrics |
| **API + Agents** | `backend/src/` | HTTP routes, LangGraph pipeline, business logic |
| **Infrastructure** | Docker | Postgres (users/logs), Redis (cache), Qdrant (vectors), Ollama (LLM/embeddings) |

The **brain** of the backend is the LangGraph state machine in `backend/src/agents/graph.py`. Every research question flows through it.

---

## 2. Complete data flow (two main paths)

### Path A — Ingestion (writing to the knowledge base)

```
KnowledgeBaseView.jsx
    │  POST /api/ingest/url | /youtube | /pdf  (+ JWT)
    ▼
routers/ingest.py  →  ingest_*_endpoint()
    │  creates IngestionJob in Postgres (status: pending)
    ▼
ingestion/{url,youtube,pdf}_loader.py  →  ingest_*()
    │  extracts plain text + metadata (source, url, title, timestamp, page_num)
    ▼
rag/factory.py  →  get_retriever()          ← single shared instance
    ▼
rag/retriever.py  →  HermesRetriever.ingest()
    │  ├─ rag/chunker.py  →  HierarchicalChunker.chunk()
    │  │     parents (~1000 chars) + children (~200 chars)
    │  ├─ Ollama nomic-embed-text  →  dense vectors (768-dim)
    │  ├─ fastembed BM25  →  sparse vectors
    │  └─ Qdrant upsert  →  collection "hermes_docs"
    │        payload: { text, parent_id, parent_text, ...metadata }
    ▼
routers/ingest.py  →  job.status = "ok", chunks_stored = N
    ▼
JSON response → frontend toast
```

### Path B — Research query (reading from the knowledge base)

```
ResearchView.jsx
    │  POST /api/research  { query, session_id? }  (+ JWT)
    ▼
routers/research.py  →  research()
    │  logs QueryLog to Postgres
    ▼
agents/graph.py  →  run_research(query, session_id)
    │
    ├─ NODE 1: agents/cache_check.py  →  cache_check_node()
    │     └─ rag/cache.py SemanticCache.get()
    │           hit (≥0.95 similarity) → return answer, END
    │           miss → continue
    │
    ├─ NODE 2: agents/supervisor.py  →  supervisor_node()
    │     └─ llm/providers.py classify via llama3.2:3b
    │           sets query_complexity: simple | multi_hop | synthesis
    │
    ├─ NODE 3: agents/research.py  →  research_node()
    │     ├─ rag/retriever.py query()
    │     │     ├─ embed question (dense + sparse)
    │     │     ├─ Qdrant hybrid prefetch + RRF fusion
    │     │     ├─ expand parent_text from payload
    │     │     └─ rag/reranker.py rerank() → cross-encoder scores
    │     ├─ filter by MIN_RERANK_SCORE (default 0.35)
    │     ├─ build citations[] with url/title/timestamp
    │     └─ llm/providers.py get_completion() → draft answer (llama3.1:8b)
    │           simple → cache result, END
    │           multi_hop/synthesis → synthesis_agent
    │
    └─ NODE 4 (optional): agents/synthesis.py  →  synthesis_node()
          └─ refine draft, cache, END
    ▼
routers/research.py  →  { answer, citations, model_used, cache_hit, session_id }
    ▼
ResearchView.jsx  →  render markdown + CitationCards
```

---

## 3. Backend file-by-file

### 3.1 Application entry

#### `backend/src/main.py`

**Role:** FastAPI application factory.

| Piece | What it does |
|---|---|
| `lifespan()` | On startup: `create_tables()` in Postgres |
| `app` | FastAPI instance, CORS for `localhost:5173` |
| Router includes | `auth`, `ingest`, `research`, `eval` |
| `GET /health` | Liveness probe |

**Data in:** HTTP requests  
**Data out:** HTTP responses, DB tables created once at boot

---

#### `backend/src/auth.py`

**Role:** JWT utilities + `get_current_user` dependency.

| Function | Input → Output |
|---|---|
| `hash_password(plain)` | Plain text → bcrypt hash |
| `verify_password(plain, hashed)` | bool |
| `create_access_token(data)` | Payload dict → signed JWT string |
| `get_current_user(credentials, db)` | Bearer token → `User` ORM object (HTTP 401) |

**Used by:** Every protected router via `Depends(get_current_user)`.

**Security note:** If `ENV=production` and `SECRET_KEY` is still the default, import raises `RuntimeError`.

---

#### `backend/src/db.py`

**Role:** SQLAlchemy async models + session factory.

| Model | Table | Key columns |
|---|---|---|
| `User` | `users` | email, hashed_password |
| `IngestionJob` | `ingestion_jobs` | source_type, source_ref, status, chunks_stored |
| `QueryLog` | `query_logs` | query, answer, model_used, cache_hit |

| Function | Purpose |
|---|---|
| `get_db()` | FastAPI dependency — yields async session |
| `create_tables()` | `Base.metadata.create_all` on startup |

---

#### `backend/src/verify.py`

**Role:** One-shot connectivity script. Pings Ollama, Qdrant, Redis, Postgres, Groq.

Run: `uv run python -m src.verify`

---

### 3.2 HTTP routers (`backend/src/routers/`)

Routers are thin — they validate auth, call domain logic, persist audit rows, return JSON.

#### `auth.py`

| Endpoint | Handler | Flow |
|---|---|---|
| `POST /api/auth/register` | `register()` | Check duplicate email → hash password → insert User → return JWT |
| `POST /api/auth/login` | `login()` | Lookup User → verify password → return JWT |

#### `ingest.py`

| Endpoint | Handler | Loader called |
|---|---|---|
| `POST /api/ingest/url` | `ingest_url_endpoint()` | `ingest_url(url, get_retriever())` |
| `POST /api/ingest/youtube` | `ingest_youtube_endpoint()` | `ingest_youtube(url, get_retriever())` |
| `POST /api/ingest/pdf` | `ingest_pdf_endpoint()` | Save temp file → `ingest_pdf(path, get_retriever())` |

**Error handling:** Loader returns `status: "failed"` → HTTP 422, job marked `error`.

#### `research.py`

| Endpoint | Handler | Core call |
|---|---|---|
| `POST /api/research` | `research()` | `run_research(query, session_id)` → log `QueryLog` |

#### `eval.py`

| Endpoint | Handler | Purpose |
|---|---|---|
| `POST /api/eval/run` | `run_eval()` | Background RAGAS job (admin-gated if `EVAL_ADMIN_EMAILS` set) |
| `GET /api/eval/status` | `eval_status()` | Is eval running? |
| `GET /api/eval/dashboard` | `eval_dashboard()` | Query stats + `eval_report.json` |
| `GET /api/eval/golden` | `golden_report()` | Raw RAGAS report |

---

### 3.3 LangGraph agents (`backend/src/agents/`)

#### `state.py` — the shared state object

```python
class ResearchState(TypedDict):
    query: str                    # user question
    query_complexity: str         # simple | multi_hop | synthesis
    next_agent: str               # routing key for conditional edges
    retrieved_contexts: list      # raw retriever output
    citations: list[Citation]     # formatted for API/frontend
    draft_answer: str
    final_answer: str
    cache_hit: bool
    model_used: str
    error: Optional[str]
```

Every node receives `ResearchState`, returns an updated copy.

#### `graph.py` — wires nodes together

```python
# Topology:
START → cache_check → [END | supervisor → research_agent → (synthesis_agent) → END]
```

| Function | Purpose |
|---|---|
| `build_graph()` | Registers nodes + conditional edges, compiles with `MemorySaver` |
| `get_graph()` | Lazy singleton compiled graph |
| `run_research(query, session_id)` | Builds initial state, `graph.invoke()`, returns final state |

**Routing functions:** `route_after_cache`, `route_after_supervisor`, `route_after_research` — each returns `state["next_agent"]`.

#### `cache_check.py`

```python
def cache_check_node(state) -> ResearchState:
    cached = retriever.cache.get(state["query"])
    if cached and "answer" in cached:
        return { ..., "next_agent": "END", "cache_hit": True }
    return { ..., "next_agent": "supervisor", "cache_hit": False }
```

**Why first?** Skips the supervisor LLM call (~1s) and entire retrieval+generation (~minutes) on paraphrased repeats.

#### `supervisor.py`

```python
def classify_complexity(query) -> str:
    # llama3.2:3b returns "simple", "multi_hop", or "synthesis"

def supervisor_node(state) -> ResearchState:
    complexity = classify_complexity(state["query"])
    return { ..., "query_complexity": complexity, "next_agent": "research_agent" }
```

#### `research.py` — the heavy lifter

```python
def research_node(state) -> ResearchState:
    contexts = retriever.query(state["query"], top_k=5)
    contexts = [c for c in contexts if c["reranker_score"] >= MIN_RERANK_SCORE]

    citations = [{ source, title, url, page_num, timestamp, context, score }, ...]

    draft = get_completion(messages, complexity=model_complexity)

    if next_agent == "END":
        retriever.cache.set(query, { contexts, answer: draft, citations })

    return { ..., "draft_answer": draft, "final_answer": draft, "next_agent": ... }
```

#### `synthesis.py`

Only runs for `multi_hop` / `synthesis` complexity. Takes draft + citations, asks the complex model to refine, caches final answer.

---

### 3.4 RAG layer (`backend/src/rag/`)

#### `factory.py`

```python
_retriever = None

def get_retriever() -> HermesRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HermesRetriever(use_cache=True, use_reranker=True)
    return _retriever
```

**Critical design decision:** One retriever for ingest AND query. Previously each module had its own singleton — parent expansion broke silently.

#### `chunker.py`

```python
class HierarchicalChunker:
    def chunk(text, metadata) -> list[ParentChunk]:
        # parent_splitter: 1000 chars
        # child_splitter: 200 chars per parent
        # each ChildChunk has parent_id linking back
```

#### `retriever.py` — `HermesRetriever`

| Method | Steps |
|---|---|
| `ingest(text, metadata)` | chunk → embed dense+sparse → build Qdrant points with `parent_text` in payload → upsert |
| `query(question, top_k)` | embed question → hybrid prefetch (dense+sparse) → RRF → dedupe by parent → read `parent_text` from payload → rerank → return top_k dicts |

**Each result dict:**
```python
{
    "context": "<parent text ~1000 chars>",
    "child_text": "<matched child ~200 chars>",
    "score": 0.82,              # vector/RRF score
    "reranker_score": 5.65,     # cross-encoder score
    "parent_id": "uuid",
    "metadata": { source, url, title, ... }
}
```

#### `reranker.py`

```python
def rerank(query, candidates, top_k=3):
    pairs = [(query, c["context"]) for c in candidates]
    scores = CrossEncoder("ms-marco-MiniLM-L-6-v2").predict(pairs)
    # attach reranker_score, sort descending, return top_k
```

#### `cache.py` — `SemanticCache`

```python
def get(query):
    query_vec = embed(query)
    for entry in redis_index:
        sim = cosine(query_vec, entry["embedding"])
        if sim >= 0.95: return entry["result"]  # includes answer, citations
    return None

def set(query, result):
    redis.lpush(index, { query, embedding, result })
```

---

### 3.5 Ingestion loaders (`backend/src/ingestion/`)

#### `url_loader.py`

```python
def fetch_url(url) -> (text, title):
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded)
    title = trafilatura.extract_metadata(downloaded).title

def ingest_url(url, retriever):
    text, title = fetch_url(url)
    if len(text) < 100: return { status: "failed" }
    retriever.ingest(text, metadata={ source: url, url, title, type: "url" })
```

#### `youtube_loader.py`

```python
def fetch_transcript(video_id) -> segments
def segments_to_chunks(segments, chunk_seconds=120) -> [{ text, timestamp, start_seconds }]
def ingest_youtube(url, retriever):
    for chunk in chunks:
        retriever.ingest(chunk["text"], metadata={
            url: f"https://youtube.com/watch?v={id}&t={start}s",
            timestamp: "2:05", ...
        })
```

#### `pdf_loader.py`

```python
def load_pdf(path) -> [{ text, page_num, source, total_pages }]
def ingest_pdf(path, retriever):
    for page in pages:
        retriever.ingest(page["text"], metadata={ source, page_num, type: "pdf" })
```

---

### 3.6 LLM layer (`backend/src/llm/providers.py`)

```python
MODEL_MAP = {
    "simple":   "ollama/llama3.1:8b",
    "complex":  "groq/llama-3.3-70b-versatile",
    "classify": "ollama/llama3.2:3b",
}

def get_completion(messages, complexity="simple") -> str:
    return litellm.completion(model=MODEL_MAP[complexity], messages=messages)
```

LiteLLM handles Ollama `api_base` and automatic fallbacks to Groq/Gemini on errors.

---

### 3.7 Evaluation (`backend/src/evaluation/`)

#### `golden_dataset.py`

Static list `GOLDEN_QA` — 10 question/ground_truth pairs about RAG topics.

#### `ragas_eval.py`

```python
def run_evaluation(n_questions=10):
    flush Redis + delete Qdrant collection
    ingest 3 Wikipedia/tech URLs via shared get_retriever()
    for each golden question:
        state = run_research(question)   # full pipeline
        collect answer + contexts
    score with RAGAS (faithfulness, relevancy, precision, recall)
    write eval_report.json
```

---

### 3.8 Tests (`backend/tests/`)

| File | What it proves |
|---|---|
| `conftest.py` | Postgres tables, auth token fixture, mocks for retriever/ingest/research |
| `test_auth.py` | Register, login, duplicate email, 401 cases |
| `test_ingest.py` | URL ingest mocked, auth required, PDF extension check |
| `test_research.py` | Research response shape, auth, health |
| `test_integration_retriever.py` | **Live stack:** parent expansion + cache round-trip |

---

## 4. Frontend file-by-file

### `frontend/src/main.jsx`

React root — mounts `<App />` into `#root`.

### `frontend/src/App.jsx`

| Piece | Role |
|---|---|
| `AuthProvider` | Wraps app with JWT context |
| `PrivateRoute` | Redirects to `/auth` if no token |
| Routes | `/` Research, `/knowledge-base`, `/analytics`, `/auth` |

### `frontend/src/api/client.js`

```javascript
const client = axios.create({ baseURL: "http://localhost:8000/api" });
// Request interceptor: attach Authorization: Bearer <token from localStorage>
```

### `frontend/src/context/AuthContext.jsx`

| Function | API call | Side effect |
|---|---|---|
| `login(email, password)` | `POST /auth/login` | Save token to `localStorage` key `hermes_access_token` |
| `register(email, password)` | `POST /auth/register` | Same |
| `logout()` | — | Clear token |

### `frontend/src/pages/ResearchView.jsx`

**State:**
- `messages[]` — chat history (persisted in `sessionStorage`)
- `sessionId` — sent to backend for LangGraph threading
- `loading` — spinner while waiting

**On submit:**
```javascript
const res = await client.post("/research", { query: input, session_id: sessionId });
// Append { role: "assistant", content: res.data.answer, citations, cache_hit, model }
```

**`CitationCard`:** Renders `citation.url` as link (web/YouTube) or `page_num` for PDFs.

### `frontend/src/pages/KnowledgeBaseView.jsx`

| Action | Endpoint |
|---|---|
| URL input (non-YouTube) | `POST /ingest/url` |
| YouTube URL detected | `POST /ingest/youtube` |
| PDF drop | `POST /ingest/pdf` (multipart) |

### `frontend/src/pages/AnalyticsView.jsx`

`GET /eval/dashboard` → stat cards + Recharts for RAGAS metrics.

### `frontend/src/components/Sidebar.jsx`

Navigation links + logout button.

### `frontend/src/components/Auth/LoginRegister.jsx`

Toggle login/register form, calls `AuthContext`.

---

## 5. Tracing a query in the code

**User asks:** *"How does hybrid search combine dense and sparse retrieval?"*

### Step 1 — Frontend

```
ResearchView.handleSubmit()
  → client.post("/research", { query, session_id })
```

### Step 2 — API layer

```
routers/research.research()
  → run_research(query, session_id)          # agents/graph.py
  → QueryLog inserted (cache_hit=False later updated from state)
```

### Step 3 — Graph invoke

```
graph.invoke(initial_state, config={ thread_id: session_id })
```

### Step 4 — cache_check_node

```
get_retriever().cache.get(query)
  → embed query with nomic-embed-text
  → scan Redis entries, cosine similarity
  → MISS (first time) → next_agent = "supervisor"
```

### Step 5 — supervisor_node

```
classify_complexity(query)
  → llama3.2:3b returns "simple"
  → next_agent = "research_agent"
```

### Step 6 — research_node

```
retriever.query(query, top_k=5)
  1. _embed([question])           # dense 768-dim
  2. _sparse_embed([question])     # BM25 sparse vector
  3. client.query_points(
       prefetch=[dense limit=20, sparse limit=20],
       query=FusionQuery(RRF),
       limit=20
     )
  4. For each hit:
       context_text = payload["parent_text"] or child text
  5. rerank(question, candidates, top_k=5)
       → ms-marco cross-encoder scores each (query, parent_context) pair
  6. Filter: reranker_score >= 0.35

citations = [{ source: wikipedia url, url: ..., score: 5.651, ... }]

get_completion(system+user messages, complexity="simple")
  → llama3.1:8b generates answer from context

cache.set(query, { answer, citations, contexts })
  → embed query, push to Redis index

next_agent = "END"   # simple query, no synthesis
```

### Step 7 — Response

```json
{
  "answer": "Hybrid vector approaches combine dense and sparse...",
  "citations": [{ "source": "https://en.wikipedia.org/...", "score": 5.651 }],
  "model_used": "ollama/llama3.1:8b",
  "cache_hit": false,
  "session_id": "uuid"
}
```

### Step 8 — Paraphrased repeat

User asks: *"In what way does hybrid search merge dense and sparse methods?"*

```
cache_check_node → similarity 0.96 → HIT
  → return cached answer in ~0.27s
  → supervisor, research, LLM all skipped
```

---

## 6. Key data structures

### Qdrant point payload (one child chunk)

```json
{
  "text": "small child chunk for retrieval...",
  "parent_id": "abc-123",
  "parent_text": "larger parent paragraph sent to LLM...",
  "source": "https://en.wikipedia.org/wiki/...",
  "url": "https://en.wikipedia.org/wiki/...",
  "title": "Retrieval-augmented generation",
  "type": "url",
  "page_num": 1
}
```

### Citation (API → frontend)

```json
{
  "source": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
  "title": "Retrieval-augmented generation",
  "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
  "page_num": null,
  "timestamp": null,
  "context": "parent text excerpt...",
  "score": 5.651
}
```

### Redis cache entry

```json
{
  "query": "original question text",
  "embedding": [0.12, -0.34, ...],
  "result": {
    "answer": "full generated answer",
    "citations": [...],
    "contexts": [...]
  }
}
```

---

## 7. Comment convention in the codebase

Every major file has a **module docstring** at the top with four sections:

```
What this file does:
Where it sits in the HERMES pipeline:
What calls this:
What this calls:
```

Functions have docstrings explaining parameters and return values. Frontend files use `/** ... */` block comments with **Role in the UI** and **API endpoints**.

When reading code, start at the module docstring — it tells you where to look next.

---

## 8. Quick reference — "where is X handled?"

| Question | File | Function |
|---|---|---|
| User logs in | `routers/auth.py` | `login()` |
| PDF uploaded | `routers/ingest.py` | `ingest_pdf_endpoint()` |
| Text chunked | `rag/chunker.py` | `HierarchicalChunker.chunk()` |
| Vectors stored | `rag/retriever.py` | `HermesRetriever.ingest()` |
| Cache checked | `agents/cache_check.py` | `cache_check_node()` |
| Query classified | `agents/supervisor.py` | `classify_complexity()` |
| Hybrid search | `rag/retriever.py` | `HermesRetriever.query()` |
| Reranking | `rag/reranker.py` | `rerank()` |
| Answer generated | `agents/research.py` | `research_node()` |
| LLM called | `llm/providers.py` | `get_completion()` |
| Graph wired | `agents/graph.py` | `build_graph()` |
| API entry for Q&A | `routers/research.py` | `research()` |
| Chat UI | `pages/ResearchView.jsx` | `handleSubmit()` |
| JWT validated | `auth.py` | `get_current_user()` |
| RAGAS eval | `evaluation/ragas_eval.py` | `run_evaluation()` |

---

## 9. Suggested learning path for your friend

1. **Start UI:** Open `ResearchView.jsx` — see what the user experiences.
2. **Follow one API call:** `client.post("/research")` → `routers/research.py` → `run_research()`.
3. **Read the graph:** `agents/graph.py` — understand node order.
4. **Read one node deeply:** `agents/research.py` — retrieval + generation.
5. **Understand retrieval:** `rag/retriever.py` `query()` method — the RAG core.
6. **Understand ingestion:** Pick one loader → `retriever.ingest()` → `chunker.chunk()`.
7. **Run integration tests:** `uv run pytest -m integration -v` with stack up.
8. **Read HANDOFF.md** for ops, env vars, and performance numbers.

---

*This document explains **how the code works**. For deployment, env vars, and metrics, see `HANDOFF.md`. For the public project summary, see `README.md`.*
