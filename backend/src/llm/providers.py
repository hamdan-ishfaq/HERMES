"""
LLM provider layer — unified text generation via LiteLLM.

Supports ``LLM_PROVIDER=ollama`` (local GPU) or ``openrouter`` (hosted, fast eval).
"""

from __future__ import annotations

import os

import litellm
from dotenv import load_dotenv

# Shell env wins over .env so RAGAS experiment levers are not clobbered.
load_dotenv(override=False)

litellm.suppress_debug_info = True

MODEL_MAP_OLLAMA = {
    "simple": "ollama/llama3.1:8b",
    "complex": "groq/llama-3.3-70b-versatile",
    "long_doc": "gemini/gemini-2.0-flash-exp",
    "offline": "ollama/llama3.1:8b",
    "classify": "ollama/llama3.2:3b",
}

litellm.fallbacks = [
    {"groq/llama-3.3-70b-versatile": [
        "gemini/gemini-2.0-flash-exp",
        "ollama/llama3.1:8b",
    ]},
    {"ollama/llama3.1:8b": ["groq/llama-3.1-8b-instant"]},
]


def model_map() -> dict[str, str]:
    """Resolve models from env on each call (avoids stale import-time config)."""
    if os.getenv("LLM_PROVIDER", "ollama").strip().lower() == "openrouter":
        return {
            "simple": os.getenv(
                "OPENROUTER_MODEL_SIMPLE",
                "openrouter/meta-llama/llama-3.2-3b-instruct",
            ),
            "complex": os.getenv(
                "OPENROUTER_MODEL_COMPLEX",
                "openrouter/google/gemini-2.5-flash-lite",
            ),
            "long_doc": os.getenv(
                "OPENROUTER_MODEL_SIMPLE",
                "openrouter/meta-llama/llama-3.2-3b-instruct",
            ),
            "offline": "ollama/llama3.1:8b",
            "classify": os.getenv(
                "OPENROUTER_MODEL_CLASSIFY",
                "openrouter/meta-llama/llama-3.2-3b-instruct",
            ),
        }
    return MODEL_MAP_OLLAMA


def _completion_kwargs(model: str, *, stream: bool = False) -> dict:
    kwargs: dict = {"model": model, "stream": stream}
    if model.startswith("ollama"):
        kwargs["api_base"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    return kwargs


def resolve_complexity(complexity: str) -> str:
    """Map simple tier to complex when ``HERMES_SIMPLE_MODEL=complex``."""
    if complexity == "simple" and os.getenv("HERMES_SIMPLE_MODEL", "").strip().lower() == "complex":
        return "complex"
    return complexity


def get_completion(messages: list[dict], complexity: str = "simple") -> str:
    """Send a chat completion for the given complexity tier."""
    models = model_map()
    # Keep classify/supervisor on the cheap model even when simple→complex is set.
    if complexity == "classify":
        resolved = "classify"
    else:
        resolved = resolve_complexity(complexity)
    model = models.get(resolved, models["simple"])
    if os.getenv("LLM_PROVIDER", "ollama").strip().lower() == "openrouter":
        litellm.fallbacks = []
    response = litellm.completion(messages=messages, **_completion_kwargs(model))
    return response.choices[0].message.content


def get_completion_stream(messages: list[dict], complexity: str = "simple"):
    """Yield text deltas from a streaming LiteLLM completion."""
    models = model_map()
    if complexity == "classify":
        resolved = "classify"
    else:
        resolved = resolve_complexity(complexity)
    model = models.get(resolved, models["simple"])
    if os.getenv("LLM_PROVIDER", "ollama").strip().lower() == "openrouter":
        litellm.fallbacks = []
    stream = litellm.completion(
        messages=messages,
        **_completion_kwargs(model, stream=True),
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None) or ""
        if text:
            yield text
