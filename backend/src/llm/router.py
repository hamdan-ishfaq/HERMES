from src.llm.providers import get_completion

def classify_query(query: str) -> str:
    """
    Classify query complexity to route to correct model.
    Returns: 'simple' | 'complex' | 'long_doc'
    """
    prompt = [
        {
            "role": "system",
            "content": (
                "Classify the user query into exactly one category:\n"
                "- simple: single fact, definition, or direct lookup\n"
                "- complex: multi-step reasoning, comparison, analysis\n"
                "- long_doc: requires reading very long documents\n"
                "Reply with ONLY the category word. No explanation."
            )
        },
        {"role": "user", "content": query}
    ]
    result = get_completion(prompt, complexity="classify").strip().lower()

    # Sanitise — model sometimes adds punctuation
    for category in ["simple", "complex", "long_doc"]:
        if category in result:
            return category
    return "simple"  # safe default


if __name__ == "__main__":
    tests = [
        "What is RAG?",
        "Compare transformer attention mechanisms across GPT-4 and Gemini architectures",
        "Summarise this 200-page annual report",
    ]
    for q in tests:
        print(f"  [{classify_query(q)}] {q}")
