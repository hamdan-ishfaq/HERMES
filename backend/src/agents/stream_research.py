"""
Streaming research helper — real token SSE events (HE-5).

Yields (event_name, data_dict) tuples for the research stream endpoint.
"""

from __future__ import annotations

from src.agents.cache_check import cache_check_node
from src.agents.query_rewrite import query_rewrite_node
from src.agents.research import (
    _passes_relevance_gate,
    _retrieve_contexts,
    _rewrite_for_retrieval,
    _crag_lite_enabled,
)
from src.agents.state import Citation, ResearchState
from src.agents.supervisor import classify_complexity
from src.llm.providers import get_completion, get_completion_stream, model_map, resolve_complexity
from src.rag.factory import get_retriever


def _build_gen_messages(
    query: str,
    citations: list[Citation],
    history_msgs: list,
) -> list[dict]:
    context_str = "\n\n---\n\n".join(
        [
            f"[Source: {c['source']} | Score: {c['score']:.2f}]\n{c['context']}"
            for c in citations
        ]
    )
    history_str = ""
    past_msgs = [
        m
        for m in history_msgs
        if (m.get("content", "") if isinstance(m, dict) else "") != query
    ][-4:]
    if past_msgs:
        history_str = "Previous Conversation:\n"
        for m in past_msgs:
            role = (m.get("role", "User") if isinstance(m, dict) else "User").capitalize()
            content = m.get("content", "") if isinstance(m, dict) else ""
            history_str += f"{role}: {content}\n"
        history_str += "\n"
    return [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Answer the question using ONLY the provided context. "
                "If the context is empty or irrelevant, say you found no relevant information. "
                "Never invent URLs, titles, or sources that are not in the context. "
                "Synthesize the information naturally. CRITICAL: Do NOT mention filenames, URLs, "
                "page numbers, or relevance scores in your text response. Do not use brackets "
                "like [Source: X] or citations like [1]. Just write a clean, fluid answer. "
                "The UI will handle the citations separately."
            ),
        },
        {
            "role": "user",
            "content": f"{history_str}Context:\n{context_str}\n\nQuestion: {query}",
        },
    ]


def iter_research_events(
    query: str,
    messages: list[dict] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
):
    """
    Generator of SSE event payloads for a research request.

    Events: status | token | done | error
    """
    messages = messages or []
    state: ResearchState = {
        "query": query,
        "query_complexity": "",
        "next_agent": "",
        "retrieved_contexts": [],
        "citations": [],
        "draft_answer": "",
        "final_answer": "",
        "messages": messages,
        "rewritten_query": "",
        "tool_trace": [],
        "user_id": user_id,
        "cache_hit": False,
        "model_used": "",
        "error": None,
    }

    try:
        state = cache_check_node(state)
        if state.get("cache_hit"):
            yield ("status", {"stage": "cache"})
            answer = state.get("final_answer") or ""
            if answer:
                yield ("token", {"text": answer})
            yield (
                "done",
                {
                    "answer": answer,
                    "citations": state.get("citations") or [],
                    "tool_trace": state.get("tool_trace") or [],
                    "cache_hit": True,
                    "session_id": session_id,
                    "model_used": state.get("model_used") or "cache",
                    "rewritten_query": query,
                },
            )
            return

        yield ("status", {"stage": "retrieve"})
        complexity = classify_complexity(query)
        state["query_complexity"] = complexity
        state = query_rewrite_node(state)

        search_q = state.get("rewritten_query") or query
        tool_trace: list[dict] = []
        contexts = _retrieve_contexts(search_q, user_id, tool_trace)

        if not contexts:
            answer = "No relevant information found in the knowledge base."
            yield ("token", {"text": answer})
            yield (
                "done",
                {
                    "answer": answer,
                    "citations": [],
                    "tool_trace": tool_trace,
                    "cache_hit": False,
                    "session_id": session_id,
                    "model_used": "",
                    "rewritten_query": search_q,
                },
            )
            return

        if _crag_lite_enabled() and not _passes_relevance_gate(query, contexts):
            retry_q = _rewrite_for_retrieval(search_q)
            retry_contexts = _retrieve_contexts(retry_q, user_id, tool_trace)
            if retry_contexts and _passes_relevance_gate(query, retry_contexts):
                contexts = retry_contexts
            else:
                answer = "No relevant information found in the knowledge base."
                yield ("token", {"text": answer})
                yield (
                    "done",
                    {
                        "answer": answer,
                        "citations": [],
                        "tool_trace": tool_trace,
                        "cache_hit": False,
                        "session_id": session_id,
                        "model_used": "",
                        "rewritten_query": retry_q,
                    },
                )
                return

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

        gen_messages = _build_gen_messages(query, citations, messages)
        needs_synthesis = complexity in ("multi_hop", "synthesis")
        model_complexity = "complex" if needs_synthesis else "simple"
        resolved = resolve_complexity(model_complexity)

        yield ("status", {"stage": "generate"})

        if needs_synthesis:
            # Draft first (non-stream), then stream the synthesis pass.
            draft = get_completion(gen_messages, complexity="simple")
            synth_messages = [
                {
                    "role": "system",
                    "content": (
                        "Refine the draft answer for clarity and completeness using only "
                        "the provided context. Do not invent sources. Write clean prose "
                        "without citation brackets."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\nDraft:\n{draft}\n\n"
                        f"Context:\n"
                        + "\n\n".join(c["context"] for c in citations)
                    ),
                },
            ]
            pieces: list[str] = []
            for text in get_completion_stream(synth_messages, complexity="complex"):
                pieces.append(text)
                yield ("token", {"text": text})
            answer = "".join(pieces)
            model_used = model_map()["complex"]
        else:
            pieces = []
            for text in get_completion_stream(gen_messages, complexity=resolved):
                pieces.append(text)
                yield ("token", {"text": text})
            answer = "".join(pieces)
            model_used = model_map()[resolved]
            retriever = get_retriever()
            if retriever.use_cache:
                retriever.cache.set(
                    query,
                    {"contexts": contexts, "answer": answer, "citations": citations},
                    user_id=user_id,
                )

        yield (
            "done",
            {
                "answer": answer,
                "citations": citations,
                "tool_trace": tool_trace,
                "cache_hit": False,
                "session_id": session_id,
                "model_used": model_used,
                "rewritten_query": search_q,
            },
        )
    except Exception as e:
        yield ("error", {"detail": str(e)})
