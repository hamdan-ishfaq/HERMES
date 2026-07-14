"""
Retrieval-only eval — no answer generation, no RAGAS judge (zero OpenRouter tokens).

Measures whether gold_contexts keywords appear in retrieved parent passages.
Typical runtime: 1–3 minutes for 20Q (local embed + rerank only).

Usage:
  cd backend
  HERMES_MULTI_QUERY=0 uv run python -m src.evaluation.retrieval_eval --n 20
  HERMES_MULTI_QUERY=1 uv run python -m src.evaluation.retrieval_eval --n 20 --exp-name mq
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv(override=False)

from src.agents.research import _retrieve_contexts
from src.evaluation.golden_dataset import load_golden
from src.rag.factory import get_retriever, reset_retriever

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
EXPERIMENTS_DIR = os.path.join(_BACKEND_ROOT, "eval", "experiments")


def _hit(contexts: list[dict], gold_terms: list[str]) -> dict:
    """Return which gold terms appear in concatenated retrieved text."""
    blob = " ".join(c.get("context", "") for c in contexts).lower()
    found = [t for t in gold_terms if t.lower() in blob]
    missing = [t for t in gold_terms if t.lower() not in blob]
    return {
        "found": found,
        "missing": missing,
        "recall_proxy": len(found) / len(gold_terms) if gold_terms else 0.0,
        "any_hit": len(found) > 0,
    }


def run_retrieval_eval(n: int = 20, exp_name: str = "retrieval_only") -> dict:
    reset_retriever()
    get_retriever(use_cache=False)

    items = load_golden()[:n]
    rows = []
    t0 = time.time()

    print(f"\n=== Retrieval-only eval ({len(items)} Q) — no generation ===")
    print(
        f"MULTI_QUERY={os.getenv('HERMES_MULTI_QUERY', '1')} "
        f"GRAPH={os.getenv('HERMES_GRAPH_RAG', '0')} "
        f"CANDIDATES={os.getenv('RETRIEVAL_CANDIDATES', '50')} "
        f"RERANK_DEVICE={os.getenv('RERANK_DEVICE', 'cpu')}"
    )

    for i, item in enumerate(items):
        q = item["question"]
        gold = item.get("gold_contexts") or []
        tool_trace: list[dict] = []
        t_q = time.time()
        contexts = _retrieve_contexts(q, user_id="eval", tool_trace=tool_trace)
        elapsed = time.time() - t_q
        score = _hit(contexts, gold)
        rows.append({
            "id": item.get("id"),
            "question": q,
            "n_contexts": len(contexts),
            "top_score": (
                max(c.get("reranker_score", c.get("score", 0)) for c in contexts)
                if contexts else 0.0
            ),
            "recall_proxy": round(score["recall_proxy"], 3),
            "any_hit": score["any_hit"],
            "missing": score["missing"],
            "sec": round(elapsed, 2),
            "tools": [t["name"] for t in tool_trace],
        })
        mark = "OK" if score["any_hit"] else "MISS"
        print(
            f"  [{i+1}/{len(items)}] {mark} "
            f"recall={score['recall_proxy']:.2f} "
            f"ctx={len(contexts)} {elapsed:.1f}s  {q[:50]}..."
        )

    total = time.time() - t0
    hit_rate = sum(1 for r in rows if r["any_hit"]) / len(rows) if rows else 0.0
    mean_recall = sum(r["recall_proxy"] for r in rows) / len(rows) if rows else 0.0
    mean_sec = sum(r["sec"] for r in rows) / len(rows) if rows else 0.0

    report = {
        "timestamp": datetime.now().isoformat(),
        "exp_name": exp_name,
        "mode": "retrieval_only",
        "questions_tested": len(rows),
        "hit_rate": round(hit_rate, 4),
        "mean_gold_term_recall": round(mean_recall, 4),
        "mean_sec_per_query": round(mean_sec, 2),
        "total_sec": round(total, 1),
        "levers": {
            "HERMES_MULTI_QUERY": os.getenv("HERMES_MULTI_QUERY", "1"),
            "HERMES_GRAPH_RAG": os.getenv("HERMES_GRAPH_RAG", "0"),
            "RETRIEVAL_CANDIDATES": os.getenv("RETRIEVAL_CANDIDATES", "50"),
            "MIN_RERANK_SCORE": os.getenv("MIN_RERANK_SCORE", "0.0"),
            "RERANK_DEVICE": os.getenv("RERANK_DEVICE", "cpu"),
            "EMBED_MODEL": os.getenv("EMBED_MODEL", "ollama"),
            "CHUNK_STRATEGY": os.getenv("CHUNK_STRATEGY", "fixed"),
            "CONTEXT_PACK_TOP_K": os.getenv("CONTEXT_PACK_TOP_K", "5"),
        },
        "misses": [r for r in rows if not r["any_hit"]],
        "rows": rows,
    }

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    out = os.path.join(EXPERIMENTS_DIR, f"{exp_name}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 50)
    print("RETRIEVAL-ONLY REPORT")
    print(f"Hit rate (any gold term):     {hit_rate:.3f}")
    print(f"Mean gold-term recall:        {mean_recall:.3f}")
    print(f"Mean sec/query:               {mean_sec:.2f}")
    print(f"Total:                        {total:.1f}s")
    print(f"Misses:                       {len(report['misses'])}/{len(rows)}")
    print(f"Wrote {out}")
    print("=" * 50)
    return report


def main():
    p = argparse.ArgumentParser(description="Fast retrieval-only eval (no LLM gen/judge)")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--exp-name", default="retrieval_only")
    args = p.parse_args()
    run_retrieval_eval(n=args.n, exp_name=args.exp_name)


if __name__ == "__main__":
    main()
