"""
GraphRAG-lite — entity triple extraction at ingest, 1-hop expansion at query.

Stores nodes/edges in Postgres (sync SQLAlchemy) for entity-linked retrieval.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.llm.providers import get_completion

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://hermes:hermes_pass@localhost:5432/hermes_db",
).replace("+asyncpg", "")


class GraphBase(DeclarativeBase):
    pass


class EntityNode(GraphBase):
    __tablename__ = "entity_nodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(512), nullable=False, index=True)
    parent_id = Column(String(64), nullable=False, index=True)
    source = Column(Text, nullable=True)
    user_id = Column(String(64), nullable=True, index=True)


class EntityEdge(GraphBase):
    __tablename__ = "entity_edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    subject = Column(String(512), nullable=False, index=True)
    relation = Column(String(256), nullable=False)
    obj = Column(String(512), nullable=False, index=True)
    parent_id = Column(String(64), nullable=False, index=True)
    source = Column(Text, nullable=True)
    user_id = Column(String(64), nullable=True, index=True)


_sync_engine = None
_SessionLocal = None


def _engine():
    global _sync_engine, _SessionLocal
    if _sync_engine is None:
        _sync_engine = create_engine(DATABASE_URL, echo=False)
        GraphBase.metadata.create_all(_sync_engine)
        _SessionLocal = sessionmaker(bind=_sync_engine)
    return _sync_engine


@contextmanager
def graph_session():
    _engine()
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def graph_enabled() -> bool:
    return os.getenv("HERMES_GRAPH_RAG", "0") != "0"


def _heuristic_triples(parent_text: str, max_triples: int = 8) -> list[tuple[str, str, str]]:
    """Fallback entity links when LLM extraction fails — co-occurring domain terms."""
    KEY_TERMS = [
        "retrieval-augmented generation", "rag", "hybrid search", "bm25",
        "dense", "sparse", "cross-encoder", "bi-encoder", "rerank",
        "chunking", "parent-child", "rrf", "reciprocal rank fusion",
        "vector database", "qdrant", "embedding", "semantic cache",
        "beir", "context window", "faithfulness",
    ]
    lower = parent_text.lower()
    found = [t for t in KEY_TERMS if t in lower]
    triples: list[tuple[str, str, str]] = []
    for i, a in enumerate(found):
        for b in found[i + 1 :]:
            triples.append((a, "co_occurs_with", b))
            if len(triples) >= max_triples:
                return triples
    # Also link title-cased words as soft entities
    proper = re.findall(r"\b([A-Z][a-zA-Z0-9-]{2,}(?:\s+[A-Z][a-zA-Z0-9-]{2,}){0,2})\b", parent_text)
    for i, a in enumerate(proper[:6]):
        for b in proper[i + 1 : 6]:
            triples.append((a.lower(), "related_to", b.lower()))
            if len(triples) >= max_triples:
                return triples
    return triples


def extract_triples(parent_text: str, max_triples: int = 8) -> list[tuple[str, str, str]]:
    """Extract triples — heuristic-first when ``HERMES_GRAPH_FAST=1`` (default)."""
    if os.getenv("HERMES_GRAPH_FAST", "1") != "0":
        return _heuristic_triples(parent_text, max_triples=max_triples)

    snippet = parent_text[:2000]
    prompt = [
        {
            "role": "system",
            "content": (
                "Extract factual entity-relationship triples from the text. "
                "Return a JSON array of objects with keys subject, relation, object. "
                "Max 8 triples. Only facts explicitly in the text."
            ),
        },
        {"role": "user", "content": snippet},
    ]
    try:
        raw = get_completion(prompt, complexity="classify")
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            items = json.loads(raw[start:end])
            triples = []
            for item in items:
                s = str(item.get("subject", "")).strip()
                r = str(item.get("relation", "")).strip()
                o = str(item.get("object", "")).strip()
                if s and r and o:
                    triples.append((s, r, o))
                if len(triples) >= max_triples:
                    break
            if triples:
                return triples
    except Exception as e:
        print(f"[GraphIndex] LLM triples failed ({e}) — using heuristic")
    return _heuristic_triples(parent_text, max_triples=max_triples)

def index_parent_chunk(
    parent_id: str,
    parent_text: str,
    source: str | None = None,
    user_id: str | None = None,
) -> int:
    """Extract and store triples for one parent chunk. Returns triple count."""
    if not graph_enabled():
        return 0

    triples = extract_triples(parent_text)
    if not triples:
        return 0

    with graph_session() as session:
        for subject, relation, obj in triples:
            for name in (subject, obj):
                session.add(
                    EntityNode(
                        name=name.lower(),
                        parent_id=parent_id,
                        source=source,
                        user_id=str(user_id) if user_id else None,
                    )
                )
            session.add(
                EntityEdge(
                    subject=subject.lower(),
                    relation=relation.lower(),
                    obj=obj.lower(),
                    parent_id=parent_id,
                    source=source,
                    user_id=str(user_id) if user_id else None,
                )
            )
    return len(triples)


def _tokenize_entities(question: str) -> list[str]:
    """Simple entity tokens from question (proper nouns + key terms)."""
    stop = {
        "what", "how", "why", "when", "where", "who", "does", "do", "is", "are",
        "the", "a", "an", "in", "of", "for", "to", "and", "or", "used", "use",
    }
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{1,}", question)
    out = []
    for t in tokens:
        low = t.lower()
        if low not in stop and len(low) > 2:
            out.append(low)
    return out[:12]


def expand_graph(
    question: str,
    user_id: str | None = None,
    limit: int = 5,
) -> list[str]:
    """
    Return parent_ids linked to entities mentioned in the question (1-hop).
    """
    if not graph_enabled():
        return []

    entities = _tokenize_entities(question)
    if not entities:
        return []

    parent_ids: set[str] = set()
    with graph_session() as session:
        for ent in entities:
            q = select(EntityNode.parent_id).where(EntityNode.name.contains(ent))
            if user_id:
                q = q.where(EntityNode.user_id == str(user_id))
            for row in session.execute(q.limit(limit)).scalars():
                parent_ids.add(row)

            eq = select(EntityEdge.parent_id).where(
                (EntityEdge.subject.contains(ent)) | (EntityEdge.obj.contains(ent))
            )
            if user_id:
                eq = eq.where(EntityEdge.user_id == str(user_id))
            for row in session.execute(eq.limit(limit)).scalars():
                parent_ids.add(row)

    return list(parent_ids)[:limit]


def fetch_graph_contexts(
    question: str,
    retriever,
    user_id: str | None = None,
) -> list[dict]:
    """Build context dicts from graph-linked parent_ids."""
    parent_ids = expand_graph(question, user_id=user_id)
    contexts = []
    for pid in parent_ids:
        text = retriever.fetch_parent_text(pid)
        if not text:
            continue
        contexts.append({
            "context": text,
            "child_text": text[:200],
            "score": 0.5,
            "reranker_score": 0.5,
            "parent_id": pid,
            "metadata": {"source": "graph", "graph_linked": True},
        })
    return contexts
