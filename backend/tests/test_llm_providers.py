"""Unit tests for LLM model resolution and the offline-tier egress rule."""

import importlib

import pytest

from src.llm import providers


@pytest.fixture
def fresh(monkeypatch):
    """Re-import providers so module-level env reads are re-evaluated."""

    def _reload(**env):
        for k in ("GROQ_MODEL_SIMPLE", "GROQ_MODEL_COMPLEX", "GROQ_MODEL_LONG_DOC",
                  "GROQ_MODEL_OFFLINE", "GROQ_MODEL_CLASSIFY", "OLLAMA_API_BASE",
                  "OLLAMA_MODEL_OFFLINE", "LLM_PROVIDER",
                  "OPENROUTER_MODEL_SIMPLE", "OPENROUTER_MODEL_COMPLEX"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return importlib.reload(providers)

    return _reload


def test_complex_tier_is_not_a_retired_groq_model(fresh):
    p = fresh()
    for tier, model in p.MODEL_MAP_OLLAMA.items():
        assert "llama-3.3-70b-versatile" not in model, tier
        assert "llama-3.1-8b-instant" not in model, tier


def test_every_tier_is_env_overridable(fresh):
    overrides = {
        "GROQ_MODEL_SIMPLE": "groq/sentinel/simple",
        "GROQ_MODEL_COMPLEX": "groq/sentinel/complex",
        "GROQ_MODEL_LONG_DOC": "groq/sentinel/long_doc",
        "GROQ_MODEL_OFFLINE": "groq/sentinel/offline",
        "GROQ_MODEL_CLASSIFY": "groq/sentinel/classify",
    }
    p = fresh(**overrides)
    for tier, env_key in (
        ("simple", "GROQ_MODEL_SIMPLE"),
        ("complex", "GROQ_MODEL_COMPLEX"),
        ("long_doc", "GROQ_MODEL_LONG_DOC"),
        ("classify", "GROQ_MODEL_CLASSIFY"),
    ):
        assert p.model_map()[tier] == overrides[env_key], tier
    # offline is overridden to ollama whenever OLLAMA_API_BASE is set, so assert
    # its own env var directly on the module dict.
    assert p.MODEL_MAP_OLLAMA["offline"] == overrides["GROQ_MODEL_OFFLINE"]


def test_offline_tier_defaults_to_hosted_without_ollama(fresh):
    p = fresh()
    assert not p.MODEL_MAP_OLLAMA["offline"].startswith("ollama")


def test_offline_tier_uses_ollama_when_configured(fresh):
    p = fresh(OLLAMA_API_BASE="http://localhost:11434")
    assert p.MODEL_MAP_OLLAMA["offline"].startswith("ollama")


def test_fallback_targets_a_live_tier(fresh):
    p = fresh()
    fallback_map = p.litellm.fallbacks[0]
    (src_model, targets), = fallback_map.items()
    assert src_model in p.MODEL_MAP_OLLAMA.values()
    for t in targets:
        assert t in p.MODEL_MAP_OLLAMA.values()


def test_openrouter_branch_is_env_driven(fresh):
    p = fresh(
        LLM_PROVIDER="openrouter",
        OPENROUTER_MODEL_SIMPLE="openrouter/meta-llama/llama-3.2-3b-instruct",
        OPENROUTER_MODEL_COMPLEX="openrouter/google/gemini-2.5-flash-lite",
    )
    m = p.model_map()
    assert m["complex"].startswith("openrouter/")
    assert "gpt-oss" not in m["complex"]


def test_resolve_complexity_promotes_simple_when_configured(fresh):
    p = fresh()
    import os
    os.environ["HERMES_SIMPLE_MODEL"] = "complex"
    try:
        assert p.resolve_complexity("simple") == "complex"
        assert p.resolve_complexity("complex") == "complex"
    finally:
        del os.environ["HERMES_SIMPLE_MODEL"]


def test_resolve_complexity_leaves_simple_alone_by_default(fresh):
    p = fresh()
    assert p.resolve_complexity("simple") == "simple"


def test_completion_kwargs_adds_ollama_api_base(fresh):
    p = fresh()
    kw = p._completion_kwargs("ollama/llama3.1:8b")
    assert kw["api_base"] == "http://localhost:11434"
    assert p._completion_kwargs("groq/openai/gpt-oss-20b").get("api_base") is None