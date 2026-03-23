"""
Cross-encoder reranker.

Retrieval (bi-encoder) is fast but imprecise — it compares query and
document embeddings independently. A cross-encoder reads query + document
together, giving much better relevance scores at the cost of speed.

Pattern:
  1. Retrieve top-20 candidates cheaply via vector search
  2. Rerank with cross-encoder, keep top-3
  3. Send top-3 parent contexts to LLM

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 80MB, runs on CPU
  - Trained on MS MARCO passage ranking
  - Standard choice for RAG reranking
"""

from sentence_transformers import CrossEncoder

_model = None  # lazy load — only initialise on first use

def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        print("Loading reranker model (first time only)...")
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Reranker loaded")
    return _model


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """
    Rerank retrieved candidates using cross-encoder.

    Args:
        query: the user question
        candidates: list of dicts with 'context' and other fields
        top_k: how many to return after reranking

    Returns:
        top_k candidates sorted by reranker score (highest first)
    """
    if not candidates:
        return []

    reranker = get_reranker()

    # Cross-encoder scores query against each candidate's context
    pairs = [(query, c["context"]) for c in candidates]
    scores = reranker.predict(pairs)

    # Attach reranker scores
    for i, candidate in enumerate(candidates):
        candidate["reranker_score"] = round(float(scores[i]), 4)
        candidate["vector_score"] = candidate.get("score", 0)

    # Sort by reranker score descending
    reranked = sorted(candidates, key=lambda x: x["reranker_score"], reverse=True)

    return reranked[:top_k]


if __name__ == "__main__":
    # Smoke test
    candidates = [
        {"context": "RAG combines retrieval with generation for better answers.", "score": 0.82},
        {"context": "The weather in London is often cloudy and rainy.", "score": 0.61},
        {"context": "Parent-child chunking improves RAG retrieval precision.", "score": 0.79},
        {"context": "Vector databases store embeddings for semantic search.", "score": 0.75},
    ]

    query = "How does RAG improve language model accuracy?"
    results = rerank(query, candidates, top_k=3)

    print(f"Query: {query}\n")
    for i, r in enumerate(results):
        print(f"Rank {i+1} (reranker: {r['reranker_score']} | vector: {r['vector_score']}):")
        print(f"  {r['context']}")
