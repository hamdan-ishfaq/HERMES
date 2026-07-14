# CV bullets — HERMES elevation

Use numbers **only** if they match committed [`backend/eval_report.json`](../backend/eval_report.json) (`clean_slate_winning`).

## HERMES (Applied AI / agentic RAG star)

1. Built an agentic RAG research assistant over PDFs, URLs, and YouTube with grounded citations (source URL / page / timestamp).
2. Hybrid retrieval: dense **BGE-m3** (1024-d) + BM25 sparse in Qdrant with Reciprocal Rank Fusion and **bge-reranker-v2-m3** cross-encoder reranking; parent-child chunking (`fixed_large`).
3. LangGraph pipeline (`cache_check → supervisor → query_rewrite → research → synthesis`) behind JWT FastAPI with forced `hybrid_search` + `tool_trace` and token SSE streaming.
4. Redis semantic cache (cosine ≥ 0.95) that bypasses retrieval and generation on paraphrased repeats.
5. RAGAS on a fixed 20Q gold set (`gold_v2`) — **faithfulness 1.00 · answer relevancy 0.87 · context precision 0.83 · context recall 1.00** (industry targets met; cite `eval_report.json` only).
6. Retrieval-only eval gate (hit_rate 1.00), multi-query expansion, MCP `hermes_search`, workspace-scoped Qdrant `user_id` ACL.

## JurisGuard (supporting — max 2 bullets)

1. On-prem / air-gapped deployment constraints for regulated environments.
2. Eval discipline and access-control notes — not a lead “legal intelligence platform” story.

## Suggested ordering on the CV

- **Applied AI / Optimization:** HaulRank → HERMES → JurisGuard (short) → TALASH
- **Agentic systems:** NEXUS → HERMES (expanded) → JurisGuard (short) → HaulRank

## Do not claim

- “Enterprise-grade” or “mathematically verified” grounding
- Old 10Q scores (~0.72–0.81) as current
- Fake conversational memory / UI-only chat history as long-term memory
- GPU production latency if measuring CPU-only cold starts
- SSO/SCIM/WORM or multi-tenant SaaS for HERMES
