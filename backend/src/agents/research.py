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

What this calls:
    - ``src.rag.factory.get_retriever`` — hybrid search + reranking
    - ``src.rag.query_expand`` — multi-query paraphrase retrieval
    - ``src.rag.context_pack`` — dedup + top-N cap before generation
    - ``src.llm.providers.get_completion`` — draft answer generation
"""

import os

from src.agents.state import ResearchState, Citation
from src.rag.factory import get_retriever
from src.rag.context_pack import pack_contexts
from src.rag.query_expand import expand_queries, merge_contexts_by_parent
from src.llm.providers import get_completion, model_map, resolve_complexity
from src.tools.hybrid_search import hybrid_search
from src.tools.graph_search import graph_search
from src.rag.graph_index import graph_enabled

# Minimum cross-encoder relevance score for a context to be kept.
# ms-marco-MiniLM scores are logits; ~0.0 is borderline, higher is better.

NO_CONTEXT_MSG = "No relevant information found in the knowledge base."


def _min_rerank_score() -> float:
    return float(os.getenv("MIN_RERANK_SCORE", "0.0"))


def _grade_score_threshold() -> float:
    return float(os.getenv("GRADE_SCORE_THRESHOLD", "0.25"))


def _crag_lite_enabled() -> bool:
    return os.getenv("HERMES_CRAG_LITE", "1") != "0"


_SOFT_EMPTY_FLOOR = -2.0
_RETRIEVAL_TOP_K = 10


def _score_of(ctx: dict) -> float:
    return float(ctx.get("reranker_score", ctx.get("score", 0)) or 0)


def _filter_by_rerank(contexts: list[dict]) -> list[dict]:
    """Hard threshold with soft floor: never return empty if top hit ≥ −2.0."""
    threshold = _min_rerank_score()
    filtered = [c for c in contexts if _score_of(c) >= threshold]
    if filtered:
        return filtered
    if not contexts:
        return []
    top = max(contexts, key=_score_of)
    if _score_of(top) >= _SOFT_EMPTY_FLOOR:
        return [top]
    return []


def _retrieve_contexts(
    search_q: str,
    user_id: str | None,
    tool_trace: list[dict],
) -> list[dict]:
    """Run multi-query or single hybrid_search; filter and pack contexts."""
    queries = expand_queries(search_q)
    if len(queries) == 1:
        raw = hybrid_search(search_q, top_k=_RETRIEVAL_TOP_K, user_id=user_id)
        tool_trace.append({
            "name": "hybrid_search",
            "args": {
                "query": search_q,
                "top_k": _RETRIEVAL_TOP_K,
                "user_id": user_id,
            },
            "ok": True,
            "summary": f"{len(raw)} contexts",
        })
        if graph_enabled():
            graph_hits = graph_search(search_q, user_id=user_id)
            if graph_hits:
                tool_trace.append({
                    "name": "graph_search",
                    "args": {"query": search_q, "user_id": user_id},
                    "ok": True,
                    "summary": f"{len(graph_hits)} graph contexts",
                })
                raw = merge_contexts_by_parent([raw, graph_hits])
    else:
        per_query: list[list[dict]] = []
        for q in queries:
            hits = hybrid_search(q, top_k=_RETRIEVAL_TOP_K, user_id=user_id)
            per_query.append(hits)
            tool_trace.append({
                "name": "hybrid_search",
                "args": {
                    "query": q,
                    "top_k": _RETRIEVAL_TOP_K,
                    "user_id": user_id,
                },
                "ok": True,
                "summary": f"{len(hits)} contexts (multi-query)",
            })
        raw = merge_contexts_by_parent(per_query)

    if graph_enabled():
        graph_hits = graph_search(search_q, user_id=user_id)
        if graph_hits:
            tool_trace.append({
                "name": "graph_search",
                "args": {"query": search_q, "user_id": user_id},
                "ok": True,
                "summary": f"{len(graph_hits)} graph contexts",
            })
            raw = merge_contexts_by_parent([raw, graph_hits])

    filtered = _filter_by_rerank(raw)
    return pack_contexts(filtered)


def _llm_grade_relevant(question: str, contexts: list[dict]) -> bool:
    """Binary relevance grade using the cheap classify model."""
    snippets = "\n---\n".join(
        c.get("context", "")[:400] for c in contexts[:3]
    )
    prompt = [
        {
            "role": "system",
            "content": (
                "You grade retrieval quality. Answer only 'yes' or 'no'. "
                "Say 'yes' if the passages contain information that could "
                "help answer the question; otherwise say 'no'."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nPassages:\n{snippets}\n\nRelevant?",
        },
    ]
    try:
        verdict = get_completion(prompt, complexity="classify").strip().lower()
        return verdict.startswith("y")
    except Exception as e:
        print(f"[CRAG-lite] grade failed ({e}) — trusting reranker scores")
        top = max(c.get("reranker_score", c.get("score", 0)) for c in contexts)
        return top >= _grade_score_threshold()


def _passes_relevance_gate(question: str, contexts: list[dict]) -> bool:
    if not contexts:
        return False
    top = max(c.get("reranker_score", c.get("score", 0)) for c in contexts)
    if top >= _grade_score_threshold():
        return True
    if not _crag_lite_enabled():
        return True
    return _llm_grade_relevant(question, contexts)


def _rewrite_for_retrieval(question: str) -> str:
    """One-shot query rewrite when CRAG-lite grade fails."""
    prompt = [
        {
            "role": "system",
            "content": (
                "Rewrite the question into a clearer standalone search query "
                "for a knowledge base. Output ONLY the rewritten query."
            ),
        },
        {"role": "user", "content": question},
    ]
    try:
        rewritten = get_completion(prompt, complexity="classify").strip().strip('"')
        return rewritten if rewritten and len(rewritten) >= 5 else question
    except Exception as e:
        print(f"[CRAG-lite] rewrite failed ({e})")
        return question


def _no_context_result(state: ResearchState, tool_trace: list[dict]) -> ResearchState:
    return {
        **state,
        "retrieved_contexts": [],
        "draft_answer": NO_CONTEXT_MSG,
        "final_answer": NO_CONTEXT_MSG,
        "citations": [],
        "tool_trace": tool_trace,
        "next_agent": "END",
        "error": "no_context",
    }


def research_node(state: ResearchState) -> ResearchState:
    """
    LangGraph node — retrieve passages via hybrid_search tool, generate draft.
    """
    print(f"\n[Research] Retrieving for: {state['query'][:60]}...")

    retriever = get_retriever()
    search_q = state.get("rewritten_query") or state["query"]
    user_id = state.get("user_id")
    tool_trace: list[dict] = list(state.get("tool_trace") or [])

    contexts = _retrieve_contexts(search_q, user_id, tool_trace)

    if not contexts:
        return _no_context_result(state, tool_trace)

    if _crag_lite_enabled() and not _passes_relevance_gate(state["query"], contexts):
        print("[CRAG-lite] low relevance — rewrite + one retry")
        retry_q = _rewrite_for_retrieval(search_q)
        retry_contexts = _retrieve_contexts(retry_q, user_id, tool_trace)
        if retry_contexts and _passes_relevance_gate(state["query"], retry_contexts):
            contexts = retry_contexts
        else:
            return _no_context_result(state, tool_trace)

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

    context_str = "\n\n---\n\n".join([
        f"[Source: {c['source']} | Score: {c['score']:.2f}]\n{c['context']}"
        for c in citations
    ])

    history_msgs = state.get("messages", [])
    history_str = ""
    past_msgs = [
        m for m in history_msgs
        if (getattr(m, "content", m.get("content", "") if isinstance(m, dict) else "")
            != state["query"])
    ][-4:]

    if past_msgs:
        history_str = "Previous Conversation:\n"
        for m in past_msgs:
            role = getattr(
                m, "type", m.get("role", "User") if isinstance(m, dict) else "User"
            ).capitalize()
            content = getattr(
                m, "content", m.get("content", "") if isinstance(m, dict) else ""
            )
            history_str += f"{role}: {content}\n"
        history_str += "\n"

    complexity_to_model = {
        "simple": "simple",
        "multi_hop": "complex",
        "synthesis": "complex",
    }
    model_complexity = complexity_to_model.get(state["query_complexity"], "simple")
    resolved = resolve_complexity(model_complexity)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Answer the question using ONLY the "
                "provided context. If the context is empty or irrelevant, say you "
                "found no relevant information. Never invent URLs, titles, or sources "
                "that are not in the context. "
                "Start with a direct one-sentence answer, then add supporting detail. "
                "CRITICAL: Do NOT mention filenames, URLs, page numbers, or relevance "
                "scores in your text response. Do not use brackets like [Source: X] or "
                "citations like [1]. Just write a clean, fluid answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{history_str}Context:\n{context_str}\n\n"
                f"Question: {state['query']}\n\n"
                "Answer in the first sentence directly, then explain:"
            ),
        },
    ]

    draft = get_completion(messages, complexity=resolved)
    model_used = model_map()[resolved]

    print(f"[Research] Generated draft ({len(draft)} chars) using {model_used}")

    next_agent = (
        "synthesis_agent"
        if state["query_complexity"] in ("multi_hop", "synthesis")
        else "END"
    )

    if retriever.use_cache and next_agent == "END":
        retriever.cache.set(
            state["query"],
            {"contexts": contexts, "answer": draft, "citations": citations},
            user_id=user_id,
        )

    return {
        **state,
        "retrieved_contexts": contexts,
        "citations": citations,
        "draft_answer": draft,
        "final_answer": draft,
        "model_used": model_used,
        "next_agent": next_agent,
        "cache_hit": False,
        "tool_trace": tool_trace,
    }
