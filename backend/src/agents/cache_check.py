"""
Cache-check node — first step in the LangGraph research pipeline.

What this file does:
    Looks up the user's question in the Redis semantic cache before any LLM
    classification or vector retrieval runs. On a hit, copies the cached
    answer and citations into state and routes directly to END.

Where it sits in the HERMES pipeline:
    Immediately after START in ``agents/graph.py``. This is the honest
    latency win for repeated or paraphrased questions — the supervisor,
    retriever, and generator are skipped entirely on a cache hit.

What calls this:
    - ``src/agents/graph.py`` — registered as the ``cache_check`` node

What this calls:
    - ``src.rag.factory.get_retriever`` — accesses ``retriever.cache`` (Redis)
"""

from src.agents.state import ResearchState
from src.rag.factory import get_retriever


def cache_check_node(state: ResearchState) -> ResearchState:
    """
    LangGraph node — semantic cache lookup and short-circuit routing.

    Embeds the query (inside ``SemanticCache.get``) and compares it to prior
    cached questions. If similarity exceeds the threshold and a stored answer
    exists, populates ``final_answer`` and sets ``next_agent`` to ``"END"``.
    Otherwise routes to the supervisor for normal processing.

    Parameters:
        state: Must contain at least ``query``; other fields are passed through.

    Returns:
        Updated ``ResearchState`` with ``cache_hit``, answer fields, and
        ``next_agent`` set to either ``"END"`` or ``"supervisor"``.
    """
    print(f"\n[Cache] Checking semantic cache for: {state['query'][:60]}...")

    retriever = get_retriever()
    # Cache can be disabled on the retriever (e.g. some tests); treat as miss.
    cached = retriever.cache.get(state["query"]) if retriever.use_cache else None

    if cached and "answer" in cached:
        # Hit — replay stored pipeline outputs without re-running downstream nodes.
        print("[Cache] HIT — bypassing supervisor, retrieval, and generation")
        return {
            **state,
            "retrieved_contexts": cached.get("contexts", []),
            "citations": cached.get("citations", []),
            "draft_answer": cached["answer"],
            "final_answer": cached["answer"],
            "cache_hit": True,
            "next_agent": "END",
        }

    print("[Cache] MISS — routing to supervisor")
    return {**state, "cache_hit": False, "next_agent": "supervisor"}
