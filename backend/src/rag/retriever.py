"""
Retriever — embeds child chunks into Qdrant, fetches parent context at query time.

Flow:
  ingest:  text → chunker → embed children → store in Qdrant
                                           → store parents in memory dict
  query:   question → embed → search Qdrant (children) → fetch parents → return context
"""

import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue
)
from src.rag.chunker import HierarchicalChunker, ParentChunk

load_dotenv()

COLLECTION_NAME = "hermes_docs"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_DIM = 768  # nomic-embed-text output dimension


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed texts using Ollama nomic-embed-text."""
    import httpx, json
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


class HermesRetriever:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.chunker = HierarchicalChunker()
        # In-memory parent store: parent_id → ParentChunk
        # Phase 2 will move this to Redis
        self._parent_store: dict[str, ParentChunk] = {}
        self._ensure_collection()

    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_DIM,
                    distance=Distance.COSINE,
                ),
            )
            print(f"Created collection: {COLLECTION_NAME}")
        else:
            print(f"Collection exists: {COLLECTION_NAME}")

    def ingest(self, text: str, metadata: dict = {}) -> dict:
        """
        Chunk text, embed children, store in Qdrant.
        Returns ingestion stats.
        """
        parents = self.chunker.chunk(text, metadata)

        # Collect all children for batch embedding
        all_children = []
        for parent in parents:
            self._parent_store[parent.id] = parent
            all_children.extend(parent.children)

        if not all_children:
            return {"parents": 0, "children": 0}

        # Batch embed all child texts
        print(f"Embedding {len(all_children)} child chunks...")
        child_texts = [c.text for c in all_children]
        vectors = _embed(child_texts)

        # Upsert into Qdrant
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i],
                payload={
                    "text": all_children[i].text,
                    "parent_id": all_children[i].parent_id,
                    **all_children[i].metadata,
                },
            )
            for i in range(len(all_children))
        ]

        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Stored {len(points)} child chunks in Qdrant")

        return {
            "parents": len(parents),
            "children": len(all_children),
        }

    def query(self, question: str, top_k: int = 3) -> list[dict]:
        """
        Embed question, search Qdrant for similar child chunks,
        return their parent contexts with citations.
        """
        # Embed the question
        q_vector = _embed([question])[0]

        # Search Qdrant for top-k child chunks
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vector,
            limit=top_k,
            with_payload=True,
        ).points

        # Deduplicate by parent_id and fetch parent context
        seen_parents = set()
        contexts = []

        for hit in results:
            parent_id = hit.payload.get("parent_id")
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            parent = self._parent_store.get(parent_id)
            context_text = parent.text if parent else hit.payload["text"]

            contexts.append({
                "context": context_text,
                "child_text": hit.payload["text"],
                "score": round(hit.score, 4),
                "parent_id": parent_id,
                "metadata": {k: v for k, v in hit.payload.items()
                            if k not in ("text", "parent_id")},
            })

        return contexts


if __name__ == "__main__":
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
    """ * 4

    retriever = HermesRetriever()

    print("\n--- Ingesting sample text ---")
    stats = retriever.ingest(sample, metadata={"source": "test_doc", "page": 1})
    print(f"Ingested: {stats}")

    print("\n--- Querying ---")
    results = retriever.query("What is parent-child chunking?", top_k=3)
    for i, r in enumerate(results):
        print(f"\nResult {i+1} (score: {r['score']}):")
        print(f"  Child: {r['child_text'][:80]}...")
        print(f"  Parent context ({len(r['context'])} chars): {r['context'][:120]}...")
