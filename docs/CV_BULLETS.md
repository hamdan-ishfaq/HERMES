# CV bullets — HERMES elevation

Use numbers **only** if they match committed [`backend/eval_report.json`](../backend/eval_report.json).

## HERMES (Applied AI / agentic RAG star)

1. Built an agentic RAG research assistant over PDFs, URLs, and YouTube with grounded citations (source URL / page / timestamp).
2. Hybrid retrieval: dense (`nomic-embed-text`) + BM25 sparse in Qdrant with Reciprocal Rank Fusion and `ms-marco-MiniLM-L-6-v2` cross-encoder reranking.
3. LangGraph pipeline (`cache_check → supervisor → query_rewrite → research → synthesis`) behind JWT-secured FastAPI; research uses a forced `hybrid_search` tool with `tool_trace`.
4. Redis semantic cache (cosine ≥ 0.95) that bypasses retrieval and generation on paraphrased repeats.
5. RAGAS evaluation on a fixed gold set — cite faithfulness / relevancy / precision / recall from `eval_report.json` only.
6. Multi-turn retrieval rewrite (Postgres turns), MCP `hermes_search` / `hermes_research`, workspace-scoped Qdrant `user_id` filter, and real token SSE (`POST /api/research/stream`).

## JurisGuard (supporting — max 2 bullets)

1. On-prem / air-gapped deployment constraints for regulated environments.
2. Eval discipline and access-control notes — not a lead “legal intelligence platform” story.

## Suggested ordering on the CV

- **Applied AI / Optimization:** HaulRank → HERMES → JurisGuard (short) → TALASH
- **Agentic systems:** NEXUS → HERMES (expanded) → JurisGuard (short) → HaulRank

## Do not claim

- “Enterprise-grade” or “mathematically verified” grounding
- Fake conversational memory / UI-only chat history as graph memory
- GPU production latency if still CPU-bound (~minutes cold)
- SSO/SCIM/WORM or multi-tenant SaaS for HERMES
