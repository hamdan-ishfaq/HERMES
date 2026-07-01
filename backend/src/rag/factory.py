"""
Unified retriever factory — single shared ``HermesRetriever`` per process.

What this file does:
    Lazily constructs one ``HermesRetriever`` instance and returns it to any
    caller that needs ingestion or query-time retrieval. Provides ``reset_retriever``
    for tests and eval runs that need a clean singleton.

Where it sits in the HERMES pipeline:
    Glue between agents, ingestion loaders, and evaluation. Every code path
  that touches Qdrant should use ``get_retriever()`` so they share the same
    in-memory parent store and cache configuration.

What calls this:
    - ``src/agents/cache_check.py``, ``research.py``, ``synthesis.py``
    - ``src/routers/ingest.py``
    - ``src/evaluation/ragas_eval.py``

What this calls:
    - ``src.rag.retriever.HermesRetriever`` — hybrid search implementation

Previously each module created its own singleton, so documents ingested via one
instance were invisible to the in-memory parent store of the instance used at
query time — parent-context expansion silently never fired in the running server.
"""

from src.rag.retriever import HermesRetriever

_retriever: HermesRetriever | None = None


def get_retriever() -> HermesRetriever:
    """
    Return the process-wide ``HermesRetriever``, creating it on first use.

    Default flags: semantic cache on, cross-encoder reranker on.

    Returns:
        Shared ``HermesRetriever`` instance.
    """
    global _retriever
    if _retriever is None:
        _retriever = HermesRetriever(use_cache=True, use_reranker=True)
    return _retriever


def reset_retriever() -> None:
    """
    Drop the cached retriever instance.

    Next ``get_retriever()`` call builds a fresh object — used by tests and
    RAGAS evaluation after clearing Qdrant/Redis.
    """
    global _retriever
    _retriever = None
