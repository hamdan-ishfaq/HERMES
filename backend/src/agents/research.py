"""
Research agent — retrieves relevant contexts and generates a draft answer.

What this file does:
    Queries the shared ``HermesRetriever`` for top passages, filters weak
    matches by reranker score, builds citations, calls an LLM with the
    assembled context, and optionally caches the result for simple queries.

Where it sits in the HERMES pipeline:
    Central RAG + generation step after the supervisor. For ``simple``
    queries this node is the last step; for ``multi_hop`` / ``synthesis`` it
    passes a draft to ``synthesis_agent``.

What calls this:
    - ``src/agents/graph.py`` — ``research_agent`` node

What this calls:
    - ``src.rag.factory.get_retriever`` — hybrid search + reranking
    - ``src.llm.providers.get_completion`` — draft answer generation
"""

import os

from src.agents.state import ResearchState, Citation
from src.rag.factory import get_retriever
from src.llm.providers import get_completion, MODEL_MAP

# Minimum cross-encoder relevance score for a context to be kept.
# ms-marco-MiniLM scores are logits; ~0.0 is borderline, higher is better.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.35"))


def research_node(state: ResearchState) -> ResearchState:
    """
    LangGraph node — retrieve passages, generate draft answer, set routing.

    Steps:
        1. Hybrid retrieval from Qdrant (via ``HermesRetriever.query``)
        2. Filter candidates below ``MIN_RERANK_SCORE``
        3. Build ``Citation`` list for the frontend
        4. LLM completion with context (model tier from ``query_complexity``)
        5. Cache on END path; route to synthesis or END

    Parameters:
        state: Must include ``query`` and ``query_complexity``.

    Returns:
        Updated state with contexts, citations, draft/final answer, and
        ``next_agent`` of ``"END"`` or ``"synthesis_agent"``.
    """
    print(f"\n[Research] Retrieving for: {state['query'][:60]}...")

    retriever = get_retriever()

    # Cache is checked upstream in cache_check_node; here we always retrieve.
    contexts = retriever.query(state["query"], top_k=5)

    # Drop weakly-relevant contexts using the cross-encoder reranker score.
    contexts = [
        c for c in contexts
        if c.get("reranker_score", c.get("score", 0)) >= MIN_RERANK_SCORE
    ]

    if not contexts:
        # Early exit — nothing in the knowledge base passed the relevance bar.
        return {
            **state,
            "retrieved_contexts": [],
            "draft_answer": "No relevant information found in the knowledge base.",
            "final_answer": "No relevant information found in the knowledge base.",
            "citations": [],
            "next_agent": "END",
            "error": "no_context",
        }

    # Format citations — carry source-type metadata through to the UI so web
    # and YouTube results render as links with titles/timestamps.
    citations: list[Citation] = [
        {
            "source": c["metadata"].get("source", "unknown"),
            "title": c["metadata"].get("title"),
            "url": c["metadata"].get("url"),
            "page_num": c["metadata"].get("page_num"),
            "timestamp": c["metadata"].get("timestamp"),
            "context": c["context"],
            "score": c.get("reranker_score", c.get("score", 0)),
        }
        for c in contexts
    ]

    # Single string block fed to the LLM — sources listed per chunk for grounding.
    context_str = "\n\n---\n\n".join([
        f"[Source: {c['source']} | Score: {c['score']:.2f}]\n{c['context']}"
        for c in citations
    ])

    # Optional conversational history (if LangGraph checkpoint provides messages).
    history_msgs = state.get("messages", [])
    history_str = ""

    # Exclude current query and grab the last 4 messages for short-term memory.
    past_msgs = [m for m in history_msgs if (getattr(m, "content", m.get("content", "")) if isinstance(m, dict) or hasattr(m, "content") else "") != state["query"]][-4:]

    if past_msgs:
        history_str = "Previous Conversation:\n"
        for m in past_msgs:
            role = getattr(m, "type", m.get("role", "User") if isinstance(m, dict) else "User").capitalize()
            content = getattr(m, "content", m.get("content", "") if isinstance(m, dict) else "")
            history_str += f"{role}: {content}\n"
        history_str += "\n"

    # Map supervisor complexity label to LLM routing tier (local vs cloud).
    complexity_to_model = {
        "simple": "simple",
        "multi_hop": "complex",
        "synthesis": "complex",
    }
    model_complexity = complexity_to_model.get(state["query_complexity"], "simple")

    # System prompt keeps citations out of the prose — UI renders them separately.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Answer the question using the provided context. "
                "Use whatever relevant information is available, even if the context only "
                "partially addresses the question. Do not refuse to answer if relevant "
                "information exists in the context. "
                "Synthesize the information naturally. CRITICAL: Do NOT mention filenames, URLs, page numbers, or relevance scores in your text response. Do not use brackets like [Source: X] or citations like [1]. Just write a clean, fluid answer. The UI will handle the citations separately."
            )
        },
        {
            "role": "user",
            "content": f"{history_str}Context:\n{context_str}\n\nQuestion: {state['query']}"
        }
    ]

    draft = get_completion(messages, complexity=model_complexity)
    model_used = MODEL_MAP[model_complexity]

    print(f"[Research] Generated draft ({len(draft)} chars) using {model_used}")

    # Complex queries need a second pass; simple ones finish here.
    next_agent = (
        "synthesis_agent"
        if state["query_complexity"] in ("multi_hop", "synthesis")
        else "END"
    )

    # Only cache complete answers (simple path) — synthesis path caches later.
    if retriever.use_cache and next_agent == "END":
        retriever.cache.set(
            state["query"],
            {"contexts": contexts, "answer": draft, "citations": citations}
        )

    return {
        **state,
        "retrieved_contexts": contexts,
        "citations": citations,
        "draft_answer": draft,
        "final_answer": draft,  # overwritten by synthesis if needed
        "model_used": model_used,
        "next_agent": next_agent,
        "cache_hit": False,
    }
