# HERMES RAG Architecture Notes

This document describes the HERMES production-style retrieval-augmented generation
(RAG) stack used for evaluation. Every gold question about HERMES-specific
behaviours should be answerable from this page.

## Hybrid retrieval and BM25

HERMES uses hybrid search: dense semantic embeddings plus sparse lexical matching.
BM25 is a classic sparse ranking function based on term frequency and inverse
document frequency. It excels at exact keyword and rare-term matching and is
combined with dense vectors so both meaning and exact keywords contribute to ranking.

Dense vectors capture semantic similarity. Sparse vectors (including BM25-style
signals) excel at keyword matching. Hybrid search combines both signals so
synonym and paraphrase queries as well as exact-term queries work well.

## Reciprocal Rank Fusion (RRF)

Reciprocal Rank Fusion (RRF) merges ranked lists from multiple retrievers by
combining reciprocal ranks. RRF produces a single fused ranking without needing
calibrated similarity scores across dense and sparse channels.

## Parent-child chunking

Parent-child chunking indexes small child chunks for precise retrieval while
returning larger parent chunks to the language model for richer generation
context. Child segments improve retrieval focus; the parent expands context for
generation. A typical pipeline retrieves precise child segments then expands to
larger parents before generation.

## Chunk size trade-offs

Chunking strategy affects retrieval quality. Chunks that are too small lose
context, while chunks that are too large reduce precision and may exceed the
model context window. Chunk size controls the trade-off between precision and
context: small chunks improve retrieval focus but risk incomplete answers; large
chunks add context but dilute relevance. Hybrid or RRF fusion then compensates by
blending complementary signals over the chosen chunk index.

## Cross-encoder reranking

Hybrid retrieval is fast but approximate. After retrieving a candidate set, a
cross-encoder reranker rescores fused candidates jointly with the query to
improve precision of the final contexts sent to the LLM. Cross-encoders are
typically applied after retrieving a limited candidate set (for example top-20)
because scoring every document jointly with the query is too slow for full-corpus
search.

Parent-child chunking and cross-encoder reranking work together: children are
retrieved for precision, parents expand context, then the cross-encoder reranks
those candidates by jointly scoring query and passage so only the most relevant
parents reach the LLM.

## Semantic caching

Semantic caching stores embeddings of prior queries and returns a cached answer
when a new query is sufficiently similar, avoiding repeated retrieval and
generation. Cache hits rely on embedding similarity between the new query and
stored query vectors.

## Evaluation benchmarks (BEIR)

RAG systems are commonly evaluated using benchmarks such as BEIR, a suite of
information retrieval datasets designed to test retrievability, retrieval
accuracy, and generative quality. BEIR is used to compare retrieval methods
across diverse domains and tasks.

## End-to-end HERMES flow

1. Expand or rewrite the user query (optional multi-query).
2. Run hybrid dense + BM25 sparse search over child chunks.
3. Fuse rankings with Reciprocal Rank Fusion (RRF).
4. Expand hits to parent chunks (parent-child chunking).
5. Rerank with a cross-encoder for precision.
6. Optionally serve a semantic cache hit when a similar query embedding matches.
7. Generate the answer grounded in the packed parent contexts.
