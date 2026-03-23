"""
Synthesis agent — refines draft answers for multi-hop and cross-doc queries.

Takes the draft from research_agent and:
  - Checks for contradictions across sources
  - Ensures all parts of multi-hop questions are answered
  - Produces a final coherent answer with explicit citations
"""

from src.agents.state import ResearchState
from src.llm.providers import get_completion


def synthesis_node(state: ResearchState) -> ResearchState:
    """Refine and synthesise the draft answer."""
    print(f"\n[Synthesis] Refining answer for {state['query_complexity']} query...")

    if not state.get("draft_answer"):
        return {**state, "next_agent": "END"}

    # Build citation summary
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

    return {
        **state,
        "final_answer": final_answer,
        "next_agent": "END",
    }
