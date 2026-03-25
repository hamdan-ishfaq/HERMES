"""
Research agent — retrieves relevant contexts and generates a draft answer.

For simple queries: this is the final step.
For multi_hop/synthesis: passes draft to synthesis_agent.
"""

from src.agents.state import ResearchState, Citation
from src.rag.retriever import HermesRetriever
from src.llm.providers import get_completion, MODEL_MAP

# Shared retriever instance (initialised once)
_retriever = None

def get_retriever() -> HermesRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HermesRetriever(use_cache=True, use_reranker=True)
    return _retriever


def research_node(state: ResearchState) -> ResearchState:
    """Retrieve contexts and generate draft answer."""
    print(f"\n[Research] Retrieving for: {state['query'][:60]}...")

    retriever = get_retriever()

    # Check cache first
    cached = retriever.cache.get(state["query"]) if retriever.use_cache else None
    if cached and "answer" in cached:
        print("[Research] Cache hit — returning cached answer")
        return {
            **state,
            "retrieved_contexts": cached.get("contexts", []),
            "draft_answer": cached["answer"],
            "final_answer": cached["answer"],
            "citations": cached.get("citations", []),
            "cache_hit": True,
            "next_agent": "END",
        }

    # Retrieve contexts
    contexts = retriever.query(state["query"], top_k=5)

    # Filter out irrelevant results (reranker score below 0 = no real match)
    contexts = [c for c in contexts if c.get("reranker_score", c.get("score", 0)) > -100]

    if not contexts:
        return {
            **state,
            "retrieved_contexts": [],
            "draft_answer": "No relevant information found in the knowledge base.",
            "final_answer": "No relevant information found in the knowledge base.",
            "citations": [],
            "next_agent": "END",
            "error": "no_context",
        }
    # Format citations
    citations: list[Citation] = [
        {
            "source": c["metadata"].get("source", "unknown"),
            "page_num": c["metadata"].get("page_num"),
            "context": c["context"],
            "score": c.get("reranker_score", c.get("score", 0)),
        }
        for c in contexts
    ]

    # Build context string for LLM
    context_str = "\n\n---\n\n".join([
        f"[Source: {c['source']} | Score: {c['score']:.2f}]\n{c['context']}"
        for c in citations
    ])

    # Choose model based on complexity
    complexity_to_model = {
        "simple": "simple",
        "multi_hop": "complex",
        "synthesis": "complex",
    }
    model_complexity = complexity_to_model.get(state["query_complexity"], "simple")

    # Generate draft answer
    messages = [
        {
            "role": "system",
            "content": (
                    "You are a research assistant. Answer the question using the provided context. "
                    "Be precise and cite which source supports each claim. "
                    "Use whatever relevant information is available, even if the context only "
                    "partially addresses the question. Do not refuse to answer if relevant "
                    "information exists in the context."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuestion: {state['query']}"
        }
    ]

    draft = get_completion(messages, complexity=model_complexity)
    model_used = MODEL_MAP[model_complexity]

    print(f"[Research] Generated draft ({len(draft)} chars) using {model_used}")

    # Determine next step
    next_agent = (
        "synthesis_agent"
        if state["query_complexity"] in ("multi_hop", "synthesis")
        else "END"
    )

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
