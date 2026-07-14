# REST & API Design for RAG Systems

> Endpoint naming, PUT vs PATCH, pagination, and the API patterns production RAG services need — with authoritative references.

*Last reviewed: 2026-06-22*

Your RAG pipeline can be world-class, but if your API is wrong, clients will misuse it, databases will degrade at scale, and partial updates will silently overwrite production data. These rules apply to any REST API — and RAG services have specific endpoints worth getting right.

---

## Contents

- [Core REST Principles](#core-rest-principles)
- [1. Endpoint Naming](#1-endpoint-naming)
- [2. PUT vs PATCH](#2-put-vs-patch)
- [3. Pagination](#3-pagination)
- [4. RAG-Specific API Patterns](#4-rag-specific-api-patterns)
- [5. gRPC vs REST](#5-grpc-vs-rest)
- [Top Reference Repositories](#top-reference-repositories)
- [Production Checklist](#production-checklist)
- [Further Reading](#further-reading)

---

## Core REST Principles

**The rule:** URL is the resource. HTTP method is the action. Do not mix them.

| HTTP Method | Semantics | Idempotent? |
| :--- | :--- | :--- |
| `GET` | Read resource | Yes |
| `POST` | Create resource | No |
| `PUT` | Replace entire resource | Yes |
| `PATCH` | Partial update | No* |
| `DELETE` | Remove resource | Yes |

*PATCH is not strictly idempotent in all implementations, but should be designed to be safely retryable.

---

## 1. Endpoint Naming

### The mistake (obvious)

```http
GET /users/getActive        ❌
GET /orders/fetchPending    ❌
POST /documents/uploadFile  ❌
```

### The mistake (hidden)

Many teams fix `getUsers` but still write:

```http
GET /users/getActive        ❌  verb in URL
GET /users/active           ✅  adjective as filter
GET /users?status=active    ✅  query param filter
```

**Rule:** If there is a verb in your URL that is not an HTTP method, your design is wrong.

### Correct patterns

```http
GET    /users                    # List users
GET    /users/{id}               # Get one user
POST   /users                    # Create user
PUT    /users/{id}               # Replace user
PATCH  /users/{id}               # Partial update
DELETE /users/{id}               # Delete user

GET    /users?status=active      # Filter (not /users/getActive)
GET    /documents/{id}/chunks    # Sub-resource
POST   /queries                  # RAG query submission (noun, not /askQuestion)
```

### Naming conventions

| Pattern | Example | Notes |
| :--- | :--- | :--- |
| Plural nouns | `/documents`, `/sessions` | Collections are plural |
| Nested resources | `/sessions/{id}/messages` | Shows ownership |
| Actions as sub-resources | `POST /documents/{id}/reindex` | Rare exceptions for non-CRUD operations |
| Versioning | `/v1/queries` | Prefix, not suffix (`/queries/v1` ❌) |

**Authoritative reference:** [GitHub REST API — Resources](https://docs.github.com/en/rest/overview/resources-in-the-rest-api)

---

## 2. PUT vs PATCH

| | PUT | PATCH |
| :--- | :--- | :--- |
| **Semantics** | Replace the **entire** resource | Update **only what you send** |
| **Missing fields** | Become null / default | Remain unchanged |
| **Use when** | Client sends complete representation | Client sends partial changes |

### PATCH done right

```
1. Fetch existing resource
2. Merge changes into existing data
3. Save merged result
```

### PATCH done wrong (PUT in disguise)

```python
# ❌ Overwrites entire document with only the fields sent
def patch_user(user_id, updates):
    db.save(user_id, updates)  # Lost all other fields!

# ✅ Merge then save
def patch_user(user_id, updates):
    existing = db.get(user_id)
    merged = {**existing, **updates}
    db.save(user_id, merged)
```

**This bug shows up in production, not in testing** — because tests often create fresh records where all fields are present.

### JSON Merge Patch vs JSON Patch

| Standard | Format | Complexity |
| :--- | :--- | :--- |
| [JSON Merge Patch (RFC 7396)](https://datatracker.ietf.org/doc/html/rfc7396) | `{"name": "new"}` | Simple, common |
| [JSON Patch (RFC 6902)](https://datatracker.ietf.org/doc/html/rfc6902) | `[{"op": "replace", "path": "/name", "value": "new"}]` | Precise, array-safe |

**GitHub API uses PATCH** for partial updates (e.g., issue title + body independently).

---

## 3. Pagination

### Offset pagination

```http
GET /documents?offset=100&limit=20
```

**Problem:** The database still scans and skips the first 100 records to reach your page. Cost grows with total data, not page size. Gets slower as your dataset grows.

**When acceptable:** Small datasets (<10K rows), admin UIs where users jump to arbitrary pages.

### Cursor pagination

```http
GET /documents?after=eyJpZCI6MTIzfQ&limit=20
```

**How it works:** Database uses an indexed column to jump directly to the cursor position. Cost is O(page_size), not O(offset).

**Requirements:**

1. **Cursor field must be indexed** — without an index, you lose most of the benefit
2. **Cursor field should be unique** — or at least consistently ordered (e.g., `created_at + id`)
3. **Cannot jump to page 7 directly** — cursor pagination is sequential only
4. **Use `Link` headers** — do not construct next-page URLs manually

**GitHub pattern (authoritative):**

```http
Link: <https://api.github.com/resource?after=CURSOR>; rel="next",
      <https://api.github.com/resource?before=CURSOR>; rel="prev"
```

**Implementation tips:**

- Return `next_cursor` in response body for mobile/JSON clients
- Return `Link` header for HTTP-native clients
- Include `has_more: true/false` for simple client logic
- Never expose internal row IDs as cursors — use opaque encoded tokens

**References:**

- [GitHub: Using Pagination in the REST API](https://docs.github.com/en/rest/using-the-rest-api/using-pagination-in-the-rest-api)
- [GitHub: Best Practices for Using the REST API](https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-the-rest-api)

### Rate limiting headers

Pair pagination with rate limit transparency:

```http
X-RateLimit-Limit: 5000
X-RateLimit-Remaining: 4999
X-RateLimit-Reset: 1372700873
Retry-After: 60
```

For RAG APIs: rate limit per user AND per token budget (see [rag-failure-handling.md](rag-failure-handling.md#6-runaway-cost-token-bill-shock)).

---

## 4. RAG-Specific API Patterns

### Query endpoint

```http
POST /v1/queries
Content-Type: application/json

{
  "question": "What is the refund policy?",
  "session_id": "sess_abc123",
  "filters": { "department": "support" },
  "stream": true
}
```

**Why POST not GET:** Queries have long text, filters, and conversation context — exceeds URL length limits. POST body is appropriate.

### Streaming response (SSE)

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"token": "The", "type": "content"}
data: {"token": " refund", "type": "content"}
data: {"type": "citation", "source_id": "doc_42", "page": 3}
data: {"type": "done", "grounding_score": 0.92}
```

### Document ingestion

```http
POST   /v1/documents              # Upload + async ingest
GET    /v1/documents              # List (cursor paginated)
GET    /v1/documents/{id}         # Status + metadata
DELETE /v1/documents/{id}         # Remove from index
POST   /v1/documents/{id}/reindex # Trigger re-embedding
```

**Async pattern:** Return `202 Accepted` with `job_id` for long ingestion. Poll `GET /v1/jobs/{job_id}`.

### Session / memory

```http
GET    /v1/sessions/{id}/messages     # Cursor paginated history
DELETE /v1/sessions/{id}/memory       # GDPR erasure
GET    /v1/users/{id}/memories         # User's stored facts
PATCH  /v1/users/{id}/memories/{mid}  # Correct a stored fact
```

### Evaluation

```http
POST /v1/evaluations
{
  "dataset_id": "golden_v2",
  "metrics": ["faithfulness", "answer_relevance"]
}
```

---

## 5. gRPC vs REST

| | REST | gRPC |
| :--- | :--- | :--- |
| **Transport** | HTTP/1.1 + JSON | HTTP/2 + Protobuf |
| **Streaming** | SSE (one-way) | Bidirectional native |
| **Schema** | OpenAPI (optional) | Protobuf (required) |
| **Browser support** | Native | Needs gRPC-Web proxy |
| **Best for** | Public APIs, web clients | Internal microservices, high-throughput |

**RAG recommendation:**

- **REST + SSE** for user-facing chat APIs (browser compatibility)
- **gRPC** for internal retriever ↔ reranker ↔ LLM gateway communication
- **OpenAPI spec** for all REST endpoints — generate client SDKs

**Streaming note:** gRPC server-streaming is more efficient than SSE for internal token pipelines. Expose REST/SSE at the edge, gRPC internally.

---

## Top Reference Repositories

| Repository | API Patterns Demonstrated |
| :--- | :--- |
| [SciPhi-AI/R2R](https://github.com/SciPhi-AI/R2R) | Full RESTful RAG API with OpenAPI |
| [Naresh1401/Enterprise-Rag-Pipeline](https://github.com/Naresh1401/Enterprise-Rag-Pipeline) | FastAPI + OpenAPI docs |
| [ara-5/Enterprise-Agentic-RAG-Platform](https://github.com/ara-5/Genai-rag-agent) | FastAPI + Pydantic validation |
| [prabhaharanv/production-hybrid-rag](https://github.com/prabhaharanv/production-hybrid-rag) | API key auth, rate limiting, streaming |
| [Hamzakhan001/production-rag-platform](https://github.com/Hamzakhan001/production-rag-platform) | FastAPI + Next.js, event-driven ingestion |
| [github/rest-api-description](https://github.com/github/rest-api-description) | GitHub's OpenAPI spec — gold standard REST design |
| [tiangolo/fastapi](https://github.com/tiangolo/fastapi) | Modern Python API framework with auto OpenAPI |

---

## Production Checklist

- [ ] Nouns in URLs, HTTP methods for actions — no verbs in paths
- [ ] PATCH merges updates; never blind overwrites
- [ ] Cursor pagination on all list endpoints (indexed cursor field)
- [ ] `Link` headers or `next_cursor` in responses
- [ ] Rate limiting with `Retry-After` header
- [ ] RAG query via POST with streaming SSE option
- [ ] Async ingestion with job status endpoint
- [ ] OpenAPI spec published and versioned (`/v1/`)
- [ ] API key or OAuth on all endpoints
- [ ] Per-user token budget headers exposed to clients

---

## Further Reading

- [GitHub REST API Documentation](https://docs.github.com/en/rest)
- [Microsoft REST API Guidelines](https://github.com/microsoft/api-guidelines)
- [Zalando RESTful API Guidelines](https://opensource.zalando.com/restful-api-guidelines/)
- [production-rag-pipeline.md — Streaming](production-rag-pipeline.md#12-streaming-sse)
- [rag-security.md](rag-security.md) — Securing your RAG API

([back to main resource](README.md))
