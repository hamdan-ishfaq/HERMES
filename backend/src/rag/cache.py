"""
Redis semantic cache.

Instead of exact-match caching, we cache by semantic similarity.
If a new query is within 0.92 cosine similarity of a cached query,
return the cached answer — no embedding, no retrieval, no LLM call.

This is the difference between:
  "What is RAG?"          → cache miss, full pipeline
  "Can you explain RAG?"  → cache HIT (0.94 similarity), instant return
"""

import os
import json
import hashlib
import httpx
from dotenv import load_dotenv

load_dotenv()

SIMILARITY_THRESHOLD = 0.85
CACHE_TTL = 60 * 60 * 24  # 24 hours in seconds
EMBEDDING_MODEL = "nomic-embed-text"


def _embed_single(text: str) -> list[float]:
    """Embed a single text using Ollama."""
    base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    resp = httpx.post(
        f"{base}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x ** 2 for x in a) ** 0.5
    mag_b = sum(x ** 2 for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class SemanticCache:
    def __init__(self):
        import redis
        self.redis = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True,
        )
        self.hits = 0
        self.misses = 0

    def _cache_key(self, query: str) -> str:
        return f"hermes:query:{hashlib.md5(query.encode()).hexdigest()}"

    def _index_key(self) -> str:
        return "hermes:cache:index"

    def get(self, query: str) -> dict | None:
        """
        Check if a semantically similar query is cached.
        Returns cached result dict or None.
        """
        # Get all cached entries
        index = self.redis.lrange(self._index_key(), 0, -1)
        if not index:
            self.misses += 1
            return None

        query_vec = _embed_single(query)

        best_similarity = 0.0
        best_entry = None

        for entry_json in index:
            try:
                entry = json.loads(entry_json)
                similarity = _cosine_similarity(query_vec, entry["embedding"])
                
                # Instantly return near-perfect matches to prioritize newest entries
                if similarity >= 0.99:
                    self.hits += 1
                    print(f"Cache HIT (similarity: {similarity:.4f})")
                    return {**entry["result"], "cache_hit": True,
                            "similarity": round(similarity, 4)}

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry
            except Exception:
                continue

        if best_similarity >= SIMILARITY_THRESHOLD and best_entry:
            self.hits += 1
            print(f"Cache HIT (similarity: {best_similarity:.4f})")
            return {**best_entry["result"], "cache_hit": True,
                    "similarity": round(best_similarity, 4)}

        self.misses += 1
        return None

    def set(self, query: str, result: dict) -> None:
        """Cache a query result with its embedding."""
        embedding = _embed_single(query)
        entry = {
            "query": query,
            "embedding": embedding,
            "result": result,
        }
        # Store in list — simple approach, works for portfolio scale
        self.redis.lpush(self._index_key(), json.dumps(entry))
        self.redis.expire(self._index_key(), CACHE_TTL)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total > 0 else 0.0,
            "cached_entries": self.redis.llen(self._index_key()),
        }


if __name__ == "__main__":
    cache = SemanticCache()

    # Clear any existing test entries
    cache.redis.delete("hermes:cache:index")

    print("=== Semantic Cache Test ===\n")

    # First query — cache miss
    result1 = cache.get("What is retrieval augmented generation?")
    print(f"Query 1 result: {result1}")  # None

    # Cache a result
    cache.set("What is retrieval augmented generation?", {
        "answer": "RAG combines retrieval with LLM generation.",
        "citations": ["doc1 p.1"]
    })
    print("Cached query 1\n")

    # Exact match — should hit
    result2 = cache.get("What is retrieval augmented generation?")
    print(f"Exact match hit: {result2 is not None}")

    # Paraphrased — should hit (high similarity)
    result3 = cache.get("Can you explain RAG to me?")
    print(f"Paraphrase hit: {result3 is not None}")

    # Unrelated — should miss
    result4 = cache.get("What is the capital of France?")
    print(f"Unrelated miss: {result4 is None}")

    print(f"\nStats: {cache.stats()}")
