# HERMES RAGAS evaluation log

CV and README must cite numbers **only** from committed [`backend/eval_report.json`](../backend/eval_report.json).

## How to run

```bash
cd backend
# Full rebuild of KB + score (default 20 questions from eval/gold_v2.json)
uv run python -m src.evaluation.ragas_eval --n 20 --exp-name baseline

# After a clean ingest, score without rebuilding KB
RAGAS_MAX_WORKERS=4 RAGAS_BUILD_WORKERS=1 \
  uv run python -m src.evaluation.ragas_eval --n 20 --exp-name clean_slate_winning --no-fresh-kb

# Retrieval-only (no generation / no judge)
uv run python -m src.evaluation.retrieval_eval --n 20 --exp-name clean_slate_retrieval_v1
```

Requires: Postgres/Redis/Qdrant up; local CUDA for BGE-m3 + bge-reranker-v2-m3; OpenRouter for answer gen + RAGAS judge.

Env levers tracked in `eval_report.json`:

| Env var | Default | Purpose |
|---|---|---|
| `MIN_RERANK_SCORE` | 0.0 | Post-rerank filter (soft floor −2.0 if empty) |
| `RETRIEVAL_CANDIDATES` | 50 | Hybrid search candidate pool |
| `HERMES_MULTI_QUERY` | 1 | Multi-query paraphrase retrieval |
| `HERMES_CRAG_LITE` | 0 | Pre-gen relevance grade + 1 retry |
| `CONTEXT_PACK_TOP_K` | 5 | Dedup cap before LLM |
| `HERMES_SIMPLE_MODEL` | complex | Route simple path to complex model |
| `CHUNK_STRATEGY` | fixed_large | Parent 1200/120, child 150/50 |
| `EMBED_MODEL` | bge-m3 | Dense dim 1024 |
| `RERANK_MODEL` | BAAI/bge-reranker-v2-m3 | Cross-encoder rerank |
| `RAGAS_MAX_WORKERS` | 4 | Parallel OpenRouter judge jobs |
| `RAGAS_BUILD_WORKERS` | 1 | Serial answer build (GPU VRAM-safe) |

Redis flush is **Hermes-key scoped** (`hermes:cache:index`, `hermes:query:*`) — never `FLUSHDB`.

## Gold set v2.1

- File: [`backend/eval/gold_v2.json`](../backend/eval/gold_v2.json)
- Size: **20** items with `gold_contexts`
- Sources: Wikipedia RAG + Qdrant hybrid + SBERT cross-encoder + [`backend/eval/kb/hermes_architecture.md`](../backend/eval/kb/hermes_architecture.md)
- Every `gold_contexts` term verified as a substring of ingested plain text

## Industry targets vs clean-slate win

| Metric | Target | `clean_slate_winning` |
|---|---|---|
| Faithfulness | ≥ 0.90 | **1.0000** |
| Answer relevancy | ≥ 0.80 | **0.8711** |
| Context precision | ≥ 0.75 | **0.8327** |
| Context recall | ≥ 0.75 | **1.0000** |

Retrieval-only gate (`clean_slate_retrieval_v1`): hit_rate **1.000**, mean gold-term recall **1.000**.

## Experiment history

| Exp | Lever | Faithfulness | Relevancy | Precision | Recall | Notes |
|---|---|---|---|---|---|---|
| hist | 10Q historical | 0.7499 | 0.7172 | 0.7222 | 0.8071 | Pre-v2 |
| 0 | baseline_or_cloud | 0.9288 | 0.6828 | 0.5410 | 0.6500 | Gold/KB mismatch |
| 1 | rerank_0.20_candidates_50 | 0.8596 | 0.7558 | 0.5660 | 0.6000 | |
| 2 | improved_stack | 0.9444 | 0.6826 | 0.5375 | 0.6500 | |
| **win** | **clean_slate_winning** | **1.0000** | **0.8711** | **0.8327** | **1.0000** | Fresh KB + gold fix + soft gate + BGE |

## Winning config

```
CHUNK_STRATEGY=fixed_large
EMBED_MODEL=bge-m3
EMBED_DEVICE=cuda
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_DEVICE=cuda
MIN_RERANK_SCORE=0.0
CONTEXT_PACK_TOP_K=5
RETRIEVAL_CANDIDATES=50
HERMES_MULTI_QUERY=1
HERMES_CRAG_LITE=0
HERMES_GRAPH_RAG=0
HERMES_SIMPLE_MODEL=complex
LLM_PROVIDER=openrouter
judge=openrouter/google/gemini-2.5-flash-lite
```

Soft retrieval gate: if hard filter empties results, keep top-1 when score ≥ −2.0; `hybrid_search` top_k=10.

## Done criteria (HE-1)

- [x] ≥20 gold items with gold_contexts
- [x] One-command eval + scoped Redis flush
- [x] Exp table filled from real runs
- [x] All four industry targets met
- [x] Winning `eval_report.json` written (`clean_slate_winning`)
