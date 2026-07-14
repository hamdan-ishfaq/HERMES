#!/usr/bin/env bash
# Run RAGAS eval grid — one experiment per invocation block.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="HERMES_SIMPLE_MODEL=complex MIN_RERANK_SCORE=0.25 RETRIEVAL_CANDIDATES=50 \
HERMES_MULTI_QUERY=1 HERMES_CRAG_LITE=1 CONTEXT_PACK_TOP_K=3"

run_exp() {
  local name="$1"
  shift
  echo "========== $name =========="
  env $BASE "$@" uv run python -m src.evaluation.ragas_eval --n 20 --exp-name "$name"
}

# E1 — improved stack (fixed 200/40, nomic, no graph)
run_exp improved_stack \
  CHUNK_STRATEGY=fixed CHILD_CHUNK_SIZE=200 CHILD_CHUNK_OVERLAP=40 \
  HERMES_GRAPH_RAG=0 EMBED_MODEL=ollama

# E2 — chunk fixed_large
run_exp chunk_fixed_150_50 \
  CHUNK_STRATEGY=fixed_large HERMES_GRAPH_RAG=0 EMBED_MODEL=ollama

# E3 — semantic + graph
run_exp chunk_semantic_graph \
  CHUNK_STRATEGY=semantic HERMES_GRAPH_RAG=1 EMBED_MODEL=ollama

# E4 — best chunk + bge-m3 (no graph)
run_exp bge_m3_fixed_large \
  CHUNK_STRATEGY=fixed_large HERMES_GRAPH_RAG=0 \
  EMBED_MODEL=bge-m3 RERANK_MODEL=BAAI/bge-reranker-v2-m3 RERANK_DEVICE=cuda

# E5 — target winning config
run_exp e5_bge_graph \
  CHUNK_STRATEGY=fixed_large HERMES_GRAPH_RAG=1 \
  EMBED_MODEL=bge-m3 RERANK_MODEL=BAAI/bge-reranker-v2-m3 RERANK_DEVICE=cuda

echo "Eval grid complete. See backend/eval/experiments/"
