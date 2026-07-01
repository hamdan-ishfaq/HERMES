"""
Synthesis agent — refines draft answers for multi-hop and cross-doc queries.

What this file does:
    Takes the draft answer and citation list from ``research_node`` and runs a
    second, stronger LLM pass to improve completeness, coherence, and inline
    citations. Stores the refined answer in the semantic cache when enabled.

Where it sits in the HERMES pipeline:
    Optional final node for ``multi_hop`` and ``synthesis`` complexity only.
    Simple queries never reach this node.

What calls this:
    - ``src/agents/graph.py`` — ``synthesis_agent`` node

What this calls:
    - ``src.llm.providers.get_completion`` with ``complexity="complex"``
    - ``src.rag.factory.get_retriever`` — cache write after refinement

Takes the draft from research_agent and:
  - Checks for contradictions across sources
  - Ensures all parts of multi-hop questions are answered
  - Produces a final coherent answer with explicit citations
"""

from src.agents.state import ResearchState
from src.llm.providers import get_completion


def synthesis_node(state: ResearchState) -> ResearchState:
    """
    LangGraph node — refine draft answer for complex query types.

    If no draft is present, passes through to END unchanged. Otherwise
    builds a senior-analyst prompt with question, draft, and citation
    summaries, then replaces ``final_answer`` with the model output.

    Parameters:
        state: Expects ``draft_answer``, ``query``, ``query_complexity``,
               and optionally ``citations``.

    Returns:
        State with updated ``final_answer`` and ``next_agent`` ``"END"``.
    """
    print(f"\n[Synthesis] Refining answer for {state['query_complexity']} query...")

    if not state.get("draft_answer"):
        return {**state, "next_agent": "END"}

    # Short preview of each citation helps the model verify coverage.
    citation_summary = "\n".join([
        f"- [{c['source']} p.{c['page_num']}]: {c['context'][:100]}..."
        for c in state.get("citations", [])
    ])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior research analyst. You will receive:\n"
                "1. A research question\n"
                "2. A draft answer\n"
                "3. The source citations\n\n"
                "Your job:\n"
                "- Verify the draft is complete and accurate given the sources\n"
                "- For multi-hop questions: ensure ALL parts are answered\n"
                "- For synthesis questions: explicitly note agreements/contradictions\n"
                "- Add inline citations like [Source: filename p.N]\n"
                "- Be concise but complete\n"
                "Output ONLY the final refined answer."
            )
        },
        {
            "role": "user",
            "content": (
                f"Question: {state['query']}\n\n"
                f"Draft answer:\n{state['draft_answer']}\n\n"
                f"Available citations:\n{citation_summary}"
            )
        }
    ]

    final_answer = get_completion(messages, complexity="complex")
    print(f"[Synthesis] Final answer ready ({len(final_answer)} chars)")

    from src.rag.factory import get_retriever
    retriever = get_retriever()
    # Cache the polished answer so paraphrased repeats skip the whole graph.
    if retriever.use_cache:
        retriever.cache.set(
            state["query"],
            {
                "contexts": state.get("retrieved_contexts", []),
                "answer": final_answer,
                "citations": state.get("citations", [])
            }
        )

    return {
        **state,
        "final_answer": final_answer,
        "next_agent": "END",
    }
