"""
Retriever v2 — hybrid dense+sparse search, parent expansion, and reranking.

What this file does:
    ``HermesRetriever`` is the core RAG engine: chunks and ingests text into
    Qdrant, runs hybrid retrieval at query time, expands child hits to parent
    context, and optionally reranks with a cross-encoder.

Where it sits in the HERMES pipeline:
    - **Ingestion**: loaders call ``ingest()`` after extracting text
    - **Query**: ``research_node`` calls ``query()``; cache uses the same
      retriever instance via ``factory.get_retriever()``

What calls this:
    - ``src/rag/factory.py`` — singleton accessor
    - ``src/ingestion/*_loader.py`` — direct use in ``__main__`` blocks
    - Indirectly all agent nodes through the factory

What this calls:
    - Ollama ``nomic-embed-text`` — dense embeddings
    - fastembed BM25 — sparse vectors for hybrid search
    - Qdrant — vector storage and RRF fusion query
    - ``src.rag.chunker`` — parent-child splitting on ingest
    - ``src.rag.reranker`` — cross-encoder rescoring
    - ``src.rag.cache`` — attached when ``use_cache=True`` (lookup happens in graph)

Query flow:
  1. Embed question (dense + sparse)
  2. Hybrid search Qdrant (dense + sparse vectors, RRF fusion)
  3. Fetch parent contexts for retrieved children
  4. Rerank with cross-encoder
  5. Return ranked contexts with metadata for citations

Note: semantic cache **reads** happen in ``cache_check_node``, not in ``query()``.
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
    """
    Embed a batch of texts with Ollama's dense embedding API.

    Parameters:
        texts: Strings to embed (one HTTP request per text).

    Returns:
        List of float vectors, each length ``VECTOR_DIM``.
    """
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
    """
    Generate sparse BM25-style vectors for keyword-aware hybrid search.

    Parameters:
        texts: Strings to encode with fastembed's Qdrant/bm25 model.

    Returns:
        Qdrant ``SparseVector`` objects (indices + values) per input text.
    """
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
    """
    End-to-end retrieval service: ingest documents and query the knowledge base.

    Attributes:
        client: Qdrant client for upsert and hybrid search.
        chunker: Splits raw text into parent/child chunks.
        use_cache: When True, attaches a ``SemanticCache`` (used by graph layer).
        use_reranker: When True, cross-encoder reranks hybrid search candidates.
        cache: Redis semantic cache instance (if enabled).
    """

    def __init__(self, use_cache: bool = True, use_reranker: bool = True):
        """
        Initialize Qdrant connection, chunker, and optional cache/reranker.

        Parameters:
            use_cache: Attach ``SemanticCache`` for graph-level cache checks.
            use_reranker: Load cross-encoder reranking on ``query()``.
        """
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.chunker = HierarchicalChunker()
        # In-process map from parent_id → full parent text (fast path after ingest).
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
        """
        Create the Qdrant collection if missing.

        Configures named dense (cosine, 768-d) and sparse vector slots so hybrid
        queries can prefetch both signal types and fuse with RRF.
        """
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
        """
        Chunk text, embed children, and upsert points into Qdrant.

        Only **child** chunks are indexed (precise retrieval). Parent text is
        stored in the payload and in ``_parent_store`` for context expansion.

        Parameters:
            text: Raw document text (one page, URL body, or transcript segment).
            metadata: Source fields merged into each point payload (source, url, etc.).

        Returns:
            Dict with counts ``{"parents": N, "children": M}``; zeros if empty input.
        """
        parents = self.chunker.chunk(text, metadata)

        all_children = []
        parent_text_by_id: dict[str, str] = {}
        for parent in parents:
            self._parent_store[parent.id] = parent
            parent_text_by_id[parent.id] = parent.text
            all_children.extend(parent.children)

        if not all_children:
            return {"parents": 0, "children": 0}

        child_texts = [c.text for c in all_children]

        print(f"Dense embedding {len(all_children)} chunks...")
        dense_vectors = _embed(child_texts)

        print(f"Sparse embedding {len(all_children)} chunks...")
        sparse_vectors = _sparse_embed(child_texts)

        # Each Qdrant point = one child chunk with both vector types in payload.
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
                    # Persist the parent context so expansion works across
                    # processes and survives restarts (not just in-memory).
                    "parent_text": parent_text_by_id.get(
                        all_children[i].parent_id, all_children[i].text
                    ),
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
        Run the retrieval pipeline: hybrid search → parent fetch → rerank.

        Semantic caching lives at the LangGraph layer (cache_check node), not
        here, so the retriever has a single, honest responsibility.

        Parameters:
            question: User query string.
            top_k: Number of final passages to return after reranking.

        Returns:
            List of dicts with ``context``, ``score``, ``metadata``, and optional
            ``reranker_score`` / ``child_text`` / ``parent_id``.
        """
        # 1. Embed question (dense semantic + sparse lexical signals).
        dense_vec = _embed([question])[0]
        sparse_vec = _sparse_embed([question])[0]

        # 2. Over-fetch candidates — reranker will trim to top_k.
        retrieval_limit = top_k * 4
        # 3. Hybrid search — Qdrant runs two prefetches then fuses with RRF.
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

        # 4. Deduplicate by parent — one expanded parent context per hit family.
        seen_parents = set()
        candidates = []
        for hit in results:
            parent_id = hit.payload.get("parent_id")
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            # Prefer persisted parent_text; fall back to memory store or child text.
            parent = self._parent_store.get(parent_id)
            context_text = (
                hit.payload.get("parent_text")
                or (parent.text if parent else None)
                or hit.payload["text"]
            )

            candidates.append({
                "context": context_text,
                "child_text": hit.payload["text"],
                "score": round(hit.score, 4),
                "parent_id": parent_id,
                "metadata": {k: v for k, v in hit.payload.items()
                            if k not in ("text", "parent_id", "parent_text")},
            })

        # 5. Cross-encoder rerank for sharper relevance ordering.
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

    print("\n--- Query 2 (paraphrase, full retrieval — caching is at graph layer) ---")
    r2 = retriever.query("Explain semantic caching in RAG systems", top_k=3)
    for i, r in enumerate(r2):
        score_info = f"reranker: {r.get('reranker_score', 'N/A')}"
        print(f"  [{i+1}] ({score_info}) {r['context'][:80]}...")
