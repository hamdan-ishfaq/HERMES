"""
Retriever v2 — hybrid dense+sparse search, parent expansion, reranking, graph index.
"""

import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from src.rag.chunk_strategies import build_chunker
from src.rag.chunker import ParentChunk
from src.rag.embeddings import dense_embed, sparse_embed, vector_dimension
from src.rag.graph_index import index_parent_chunk

load_dotenv()

COLLECTION_NAME = "hermes_docs"


class HermesRetriever:
    def __init__(self, use_cache: bool = True, use_reranker: bool = True):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.chunker = build_chunker()
        self._parent_store: dict[str, ParentChunk] = {}
        self.use_cache = use_cache
        self.use_reranker = use_reranker
        self._vector_dim = vector_dimension()

        if use_cache:
            from src.rag.cache import SemanticCache

            self.cache = SemanticCache()

        if use_reranker:
            from src.rag.reranker import rerank

            self._rerank = rerank

        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME in existing:
            info = self.client.get_collection(COLLECTION_NAME)
            current_dim = info.config.params.vectors["dense"].size
            if current_dim != self._vector_dim:
                print(
                    f"Collection dim mismatch ({current_dim} vs {self._vector_dim}) — "
                    "delete collection and re-ingest when switching EMBED_MODEL"
                )
            else:
                print(f"Collection exists: {COLLECTION_NAME}")
                return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=self._vector_dim, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
        )
        print(f"Created collection: {COLLECTION_NAME} (dim={self._vector_dim})")

    def ingest(self, text: str, metadata: dict | None = None) -> dict:
        metadata = metadata or {}
        parents = self.chunker.chunk(text, metadata)

        all_children = []
        parent_text_by_id: dict[str, str] = {}
        graph_triples = 0
        user_id = metadata.get("user_id")
        source = metadata.get("source") or metadata.get("url")

        for parent in parents:
            self._parent_store[parent.id] = parent
            parent_text_by_id[parent.id] = parent.text
            all_children.extend(parent.children)
            graph_triples += index_parent_chunk(
                parent.id, parent.text, source=source, user_id=user_id
            )

        if not all_children:
            return {"parents": 0, "children": 0, "graph_triples": 0}

        child_texts = [c.text for c in all_children]
        print(f"Dense embedding {len(all_children)} chunks...")
        dense_vectors = dense_embed(child_texts)
        print(f"Sparse embedding {len(all_children)} chunks...")
        sparse_vectors = sparse_embed(child_texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": dense_vectors[i], "sparse": sparse_vectors[i]},
                payload={
                    "text": all_children[i].text,
                    "parent_id": all_children[i].parent_id,
                    "parent_text": parent_text_by_id.get(
                        all_children[i].parent_id, all_children[i].text
                    ),
                    **all_children[i].metadata,
                },
            )
            for i in range(len(all_children))
        ]

        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Stored {len(points)} chunks; graph triples indexed: {graph_triples}")
        return {
            "parents": len(parents),
            "children": len(all_children),
            "graph_triples": graph_triples,
        }

    def query(
        self,
        question: str,
        top_k: int = 3,
        user_id: str | None = None,
    ) -> list[dict]:
        dense_vec = dense_embed([question])[0]
        sparse_vec = sparse_embed([question])[0]

        candidate_pool = int(os.getenv("RETRIEVAL_CANDIDATES", "50"))
        retrieval_limit = max(top_k * 4, candidate_pool)

        query_filter = None
        if user_id is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=str(user_id)),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=retrieval_limit,
                    filter=query_filter,
                ),
                Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=retrieval_limit,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=retrieval_limit,
            with_payload=True,
        ).points

        seen_parents: set[str] = set()
        candidates = []
        for hit in results:
            parent_id = hit.payload.get("parent_id")
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

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
                "metadata": {
                    k: v
                    for k, v in hit.payload.items()
                    if k not in ("text", "parent_id", "parent_text")
                },
            })

        if self.use_reranker and candidates:
            candidates = self._rerank(question, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        return candidates

    def fetch_parent_text(self, parent_id: str) -> str | None:
        parent = self._parent_store.get(parent_id)
        if parent:
            return parent.text
        points, _ = self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_id",
                        match=MatchValue(value=parent_id),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if points:
            return points[0].payload.get("parent_text") or points[0].payload.get("text")
        return None
