import os
import litellm
from dotenv import load_dotenv

load_dotenv()

# Suppress verbose logging
litellm.suppress_debug_info = True

# Fallback chain — Groq → Gemini → Ollama local
litellm.fallbacks = [
    {"groq/llama-3.3-70b-versatile": [
        "gemini/gemini-2.0-flash-exp",
        "ollama/llama3.1:8b"
    ]},
    {"groq/llama-3.2-3b-preview": [
        "ollama/llama3.2:3b"
    ]},
]

# Updated model map — llama-3.2-3b-preview decommissioned, use llama-3.2-1b-preview
MODEL_MAP = {
    "simple":   "groq/llama-3.2-1b-preview",   # fastest free option
    "complex":  "groq/llama-3.3-70b-versatile", # best free quality
    "long_doc": "gemini/gemini-2.0-flash-exp",  # 1M context
    "offline":  "ollama/llama3.1:8b",           # local GPU
    "classify": "ollama/llama3.2:3b",           # local, fast
}

def get_completion(messages: list[dict], complexity: str = "simple") -> str:
    """Single entry point for all LLM calls. Returns response text."""
    model = MODEL_MAP.get(complexity, MODEL_MAP["simple"])

    response = litellm.completion(
        model=model,
        messages=messages,
        api_base=os.getenv("OLLAMA_API_BASE") if model.startswith("ollama") else None,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Quick smoke test
    result = get_completion(
        messages=[{"role": "user", "content": "Say READY in one word"}],
        complexity="complex"
    )
    print(f"Complex (Groq 70b): {result}")

    result = get_completion(
        messages=[{"role": "user", "content": "Say READY in one word"}],
        complexity="offline"
    )
    print(f"Offline (Ollama 8b): {result}")
