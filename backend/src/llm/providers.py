"""
LLM provider layer — unified text generation via LiteLLM.

What this file does:
    Maps logical complexity keys (``simple``, ``complex``, ``classify``) to concrete
    model IDs and calls ``litellm.completion``. Configures automatic fallback when
    a hosted API rate-limits or errors.

Where it sits in the HERMES pipeline:
    Called by ``supervisor.py`` (classify), ``research.py`` (draft answer),
    and ``synthesis.py`` (refined answer). This is the only module that talks
    to external/hosted LLMs.

What calls this:
    - ``src/agents/supervisor.classify_complexity``
    - ``src/agents/research.research_node``
    - ``src/agents/synthesis.synthesis_node``

What this calls:
    - LiteLLM → Ollama (local), Groq, or Gemini depending on ``MODEL_MAP`` key
"""

import os
import litellm
from dotenv import load_dotenv

load_dotenv()

litellm.suppress_debug_info = True

# Fallback chain — LiteLLM retries the next model on rate limit or API error.
litellm.fallbacks = [
    # Complex queries: Groq 70b → Gemini → local 8b
    {"groq/llama-3.3-70b-versatile": [
        "gemini/gemini-2.0-flash-exp",
        "ollama/llama3.1:8b",
    ]},
    # Simple queries: local 8b → Groq instant (reversed — local first)
    {"ollama/llama3.1:8b": [
        "groq/llama-3.1-8b-instant",
    ]},
]

MODEL_MAP = {
    "simple":   "ollama/llama3.1:8b",          # local GPU first, free always
    "complex":  "groq/llama-3.3-70b-versatile", # 70b quality, Groq free tier
    "long_doc": "gemini/gemini-2.0-flash-exp",  # 1M context
    "offline":  "ollama/llama3.1:8b",           # explicit offline mode
    "classify": "ollama/llama3.2:3b",           # local 3b, fast, never hits API
}

def get_completion(messages: list[dict], complexity: str = "simple") -> str:
    """
    Send a chat completion request for the given complexity tier.

    Parameters:
        messages: OpenAI-style list of ``{role, content}`` dicts.
        complexity: Key into ``MODEL_MAP`` — ``simple``, ``complex``, ``classify``, etc.

    Returns:
        Assistant message text (first choice).
    """
    model = MODEL_MAP.get(complexity, MODEL_MAP["simple"])
    # Ollama models need an explicit api_base; hosted models use env API keys.
    response = litellm.completion(
        model=model,
        messages=messages,
        api_base=os.getenv("OLLAMA_API_BASE") if model.startswith("ollama") else None,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    result = get_completion(
        [{"role": "user", "content": "Say READY in one word"}],
        complexity="simple"
    )
    print(f"Simple (Ollama 8b): {result}")

    result = get_completion(
        [{"role": "user", "content": "Say READY in one word"}],
        complexity="complex"
    )
    print(f"Complex (Groq 70b): {result}")

    result = get_completion(
        [{"role": "user", "content": "Say READY in one word"}],
        complexity="classify"
    )
    print(f"Classify (Ollama 3b): {result}")