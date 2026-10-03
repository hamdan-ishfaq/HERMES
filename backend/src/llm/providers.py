"""
LLM provider layer — unified text generation via LiteLLM.

``LLM_PROVIDER=openrouter`` selects OpenRouter-hosted models. Any other value
(including the default, ``ollama``) selects the Groq-hosted tier map below;
only the ``offline`` tier resolves to a local Ollama server. Per-tier models
are overridable via env.
"""

from __future__ import annotations

import os

import litellm

litellm.suppress_debug_info = True

# Groq retired llama-3.3-70b-versatile and llama-3.1-8b-instant; both now return
# "model does not exist". gpt-oss-120b is the large tier, gpt-oss-20b the cheap one.
# Every tier is overridable by env so a provider retiring a model does not
# require a code change.
MODEL_MAP_OLLAMA = {
    "simple": os.getenv("GROQ_MODEL_SIMPLE", "groq/openai/gpt-oss-20b"),
    "complex": os.getenv("GROQ_MODEL_COMPLEX", "groq/openai/gpt-oss-120b"),
    "long_doc": os.getenv("GROQ_MODEL_LONG_DOC", "groq/openai/gpt-oss-120b"),
    "offline": os.getenv("GROQ_MODEL_OFFLINE", "groq/openai/gpt-oss-20b"),
    "classify": os.getenv("GROQ_MODEL_CLASSIFY", "groq/openai/gpt-oss-20b"),
}

# The offline tier is documented as the no-egress path. If OLLAMA_API_BASE is set
# it resolves there instead of a hosted provider, so the caller keeps local
# inference local. Otherwise it falls back to the hosted default above.
if os.getenv("OLLAMA_API_BASE"):
    MODEL_MAP_OLLAMA["offline"] = os.getenv("OLLAMA_MODEL_OFFLINE", "ollama/llama3.1:8b")

litellm.fallbacks = [
    {MODEL_MAP_OLLAMA["complex"]: [MODEL_MAP_OLLAMA["simple"]]},
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
    kwargs: dict = {
        "model": model,
        "stream": stream,
        # Hosted free tiers rate-limit aggressively (Groq's is 30 req/min), and
        # without this a 429 surfaced to the caller as an opaque HTTP 500.
        # LiteLLM retries 429/5xx with exponential backoff.
        #
        # Only pass parameters LiteLLM consumes itself. Anything else is
        # forwarded into the provider's request body and rejected as an
        # unsupported property -- `retry_interval`, for example, made every
        # Groq call fail with a 400. test_llm_providers.py pins this key set.
        "num_retries": int(os.getenv("LLM_MAX_RETRIES", "3")),
    }
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
