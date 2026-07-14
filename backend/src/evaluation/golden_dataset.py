"""
Golden dataset loader — versioned Q&A under backend/eval/gold_v2.json.

What this file does:
    Loads the fixed gold set used by RAGAS evaluation. Prefer the JSON file
    so experiments can version questions + gold_contexts without code edits.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_GOLD_PATH = os.path.join(_BACKEND_ROOT, "eval", "gold_v2.json")


@lru_cache(maxsize=4)
def load_golden(path: str | None = None) -> list[dict]:
    """
    Load gold items as list of {id, question, ground_truth, gold_contexts?, tags?}.

    Falls back to an empty list if the file is missing (tests may mock this).
    """
    gold_path = path or DEFAULT_GOLD_PATH
    if not os.path.exists(gold_path):
        return []
    with open(gold_path) as f:
        data = json.load(f)
    return list(data.get("items", []))


def load_kb_urls(path: str | None = None) -> list[str]:
    """Return KB URLs listed in the gold JSON (for eval ingest)."""
    gold_path = path or DEFAULT_GOLD_PATH
    if not os.path.exists(gold_path):
        return []
    with open(gold_path) as f:
        data = json.load(f)
    return list(data.get("kb_urls", []))


def load_kb_files(path: str | None = None) -> list[str]:
    """Return local KB file paths (relative to backend root) from gold JSON."""
    gold_path = path or DEFAULT_GOLD_PATH
    if not os.path.exists(gold_path):
        return []
    with open(gold_path) as f:
        data = json.load(f)
    return list(data.get("kb_files", []))


def resolve_kb_file(rel_path: str) -> str:
    """Resolve a gold kb_files entry to an absolute path under backend/."""
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(_BACKEND_ROOT, rel_path)


# Back-compat alias used by older imports / scripts.
GOLDEN_QA = load_golden()
