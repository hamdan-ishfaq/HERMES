"""
Retriever v2 — hybrid search + reranking + semantic cache.

Query flow:
  1. Check semantic cache → return instantly if hit
  2. Embed question with nomic-embed-text
  3. Hybrid search Qdrant (dense + sparse vectors)
  4. Fetch parent contexts for retrieved children
  5. Rerank with cross-encoder
  6. Store result in cache
  7. Return ranked contexts with citations
"""

import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseIndexParams,
    SparseVector, Prefetch, FusionQuery, Fusion,
)
from src.rag.chunker import HierarchicalChunker, ParentChunk

load_dotenv()

COLLECTION_NAME = "hermes_docs"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_DIM = 768


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed texts using Ollama nomic-embed-text."""
    import httpx
    base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    vectors = []
    for text in texts:
        resp = httpx.post(
            f"{base}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        vectors.append(resp.json()["embeddings"][0])
    return vectors


def _sparse_embed(texts: list[str]) -> list[SparseVector]:
    """Generate sparse BM25 vectors using fastembed."""
    from fastembed import SparseTextEmbedding
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    results = []
    for embedding in sparse_model.embed(texts):
        results.append(SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        ))
    return results


class HermesRetriever:
    def __init__(self, use_cache: bool = True, use_reranker: bool = True):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.chunker = HierarchicalChunker()
        self._parent_store: dict[str, ParentChunk] = {}
        self.use_cache = use_cache
        self.use_reranker = use_reranker

        if use_cache:
            from src.rag.cache import SemanticCache
            self.cache = SemanticCache()

        if use_reranker:
            from src.rag.reranker import rerank
            self._rerank = rerank

        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection with both dense and sparse vectors."""
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": VectorParams(
                        size=VECTOR_DIM,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            print(f"Created collection: {COLLECTION_NAME} (hybrid)")
        else:
            print(f"Collection exists: {COLLECTION_NAME}")

    def ingest(self, text: str, metadata: dict = {}) -> dict:
        """Chunk, embed (dense + sparse), store in Qdrant."""
        parents = self.chunker.chunk(text, metadata)

        all_children = []
        for parent in parents:
            self._parent_store[parent.id] = parent
            all_children.extend(parent.children)

        if not all_children:
            return {"parents": 0, "children": 0}

        child_texts = [c.text for c in all_children]

        print(f"Dense embedding {len(all_children)} chunks...")
        dense_vectors = _embed(child_texts)

        print(f"Sparse embedding {len(all_children)} chunks...")
        sparse_vectors = _sparse_embed(child_texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vectors[i],
                    "sparse": sparse_vectors[i],
                },
                payload={
                    "text": all_children[i].text,
                    "parent_id": all_children[i].parent_id,
                    **all_children[i].metadata,
                },
            )
            for i in range(len(all_children))
        ]

        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Stored {len(points)} chunks (dense+sparse) in Qdrant")

        return {"parents": len(parents), "children": len(all_children)}

    def query(self, question: str, top_k: int = 3) -> list[dict]:
        """
        Full query pipeline:
        cache → hybrid search → parent fetch → rerank → return
        """
        # 1. Check cache
        if self.use_cache:
            cached = self.cache.get(question)
            if cached:
                return cached.get("contexts", [])

        # 2. Embed question (dense + sparse)
        dense_vec = _embed([question])[0]
        sparse_vec = _sparse_embed([question])[0]

        # 3. Hybrid search — Qdrant fuses dense + sparse results
        retrieval_limit = top_k * 4
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=retrieval_limit,
                ),
                Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=retrieval_limit,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=retrieval_limit,
            with_payload=True,
        ).points
        
        # 4. Fetch parent contexts
        seen_parents = set()
        candidates = []
        for hit in results:
            parent_id = hit.payload.get("parent_id")
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            parent = self._parent_store.get(parent_id)
            context_text = parent.text if parent else hit.payload["text"]

            candidates.append({
                "context": context_text,
                "child_text": hit.payload["text"],
                "score": round(hit.score, 4),
                "parent_id": parent_id,
                "metadata": {k: v for k, v in hit.payload.items()
                            if k not in ("text", "parent_id")},
            })

        # 5. Rerank
        if self.use_reranker and candidates:
            candidates = self._rerank(question, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        return candidates


if __name__ == "__main__":
    # Clear and reingest fresh
    from dotenv import load_dotenv
    load_dotenv()
    from qdrant_client import QdrantClient
    import os

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    client.delete_collection(COLLECTION_NAME)
    print("Cleared collection")

    sample = """
    Retrieval-Augmented Generation (RAG) is a technique that enhances
    language model responses by retrieving relevant documents from an
    external knowledge base before generating an answer.

    Vector databases store document embeddings and enable semantic search.
    When a user submits a query, it is embedded into the same vector space
    and the most similar document chunks are retrieved.

    Parent-child chunking is an advanced RAG technique where small chunks
    are used for precise retrieval, but the larger parent chunk is passed
    to the LLM to provide richer context for generation.

    Qdrant is a high-performance vector database that supports cosine
    similarity search, filtering, and hybrid sparse-dense retrieval.
    It is available as a managed cloud service with a free tier.

    Cross-encoder reranking improves retrieval quality by scoring each
    candidate passage against the query jointly, rather than independently.
    This gives much more accurate relevance scores at the cost of speed.

    Semantic caching stores query embeddings and returns cached answers
    for semantically similar queries, dramatically reducing latency and
    API costs for repeated or paraphrased questions.
    """ * 3

    retriever = HermesRetriever(use_cache=True, use_reranker=True)

    print("\n--- Ingesting ---")
    stats = retriever.ingest(sample, metadata={"source": "test_doc", "page": 1})
    print(f"Stats: {stats}")

    print("\n--- Query 1 (cache miss) ---")
    r1 = retriever.query("How does semantic caching work?", top_k=3)
    for i, r in enumerate(r1):
        score_info = f"reranker: {r.get('reranker_score', 'N/A')}"
        print(f"  [{i+1}] ({score_info}) {r['context'][:80]}...")

    print("\n--- Query 2 (should cache hit) ---")
    r2 = retriever.query("Explain semantic caching in RAG systems", top_k=3)
    print(f"  Cache hit: {len(r2) > 0 and r2 == r1}")

    if hasattr(retriever, 'cache'):
        print(f"\nCache stats: {retriever.cache.stats()}")
