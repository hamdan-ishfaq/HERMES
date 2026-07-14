"""
RAGAS evaluation for Hermes — local Ollama judge, shared production retriever.

Usage:
  uv run python -m src.evaluation.ragas_eval
  uv run python -m src.evaluation.ragas_eval --n 20 --exp-name baseline
  uv run python -m src.evaluation.ragas_eval --n 20 --exp-name rerank_0.20 --no-fresh-kb
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime

import redis
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv(override=False)

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

from src.agents.graph import run_research
from src.evaluation.golden_dataset import (
    load_golden,
    load_kb_files,
    load_kb_urls,
    resolve_kb_file,
)
from src.ingestion.url_loader import ingest_url
from src.rag.factory import get_retriever, reset_retriever

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
REPORT_PATH = os.path.join(_BACKEND_ROOT, "eval_report.json")
DETAILS_PATH = os.path.join(_BACKEND_ROOT, "eval_details.json")
EXPERIMENTS_DIR = os.path.join(_BACKEND_ROOT, "eval", "experiments")

HERMES_REDIS_KEYS = ("hermes:cache:index",)


def _flush_hermes_cache() -> None:
    """Delete Hermes semantic-cache keys only — never FLUSHDB."""
    client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    for key in HERMES_REDIS_KEYS:
        client.delete(key)
    # Per-user index namespaces (HE-4)
    for key in client.scan_iter(match="hermes:cache:index:*", count=200):
        client.delete(key)
    for key in client.scan_iter(match="hermes:query:*", count=100):
        client.delete(key)
    print("Hermes Redis cache keys cleared (no FLUSHDB)")


def _build_dataset(n_questions: int) -> Dataset:
    questions = load_golden()[:n_questions]
    if not questions:
        raise RuntimeError("No golden items found — check backend/eval/gold_v2.json")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    build_workers = int(os.getenv("RAGAS_BUILD_WORKERS", "8"))

    def _one(item: dict, idx: int) -> tuple[int, dict]:
        q = item["question"]
        print(f"  [{idx + 1}/{len(questions)}] {q[:60]}...")
        _flush_hermes_cache()
        state = run_research(q, user_id="eval")
        answer = state.get("final_answer") or state.get("draft_answer") or ""
        raw_contexts = state.get("retrieved_contexts", [])
        contexts = [
            c.get("context", "")
            for c in raw_contexts
            if isinstance(c, dict) and "context" in c
        ]
        if not contexts:
            contexts = ["No relevant context was found in the database."]
        return idx, {
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        }

    print(f"Building answers with {build_workers} parallel workers...")
    results: list[dict | None] = [None] * len(questions)
    with ThreadPoolExecutor(max_workers=build_workers) as pool:
        futs = {
            pool.submit(_one, item, i): i for i, item in enumerate(questions)
        }
        for fut in as_completed(futs):
            idx, row = fut.result()
            results[idx] = row

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for row in results:
        assert row is not None
        rows["question"].append(row["question"])
        rows["answer"].append(row["answer"])
        rows["contexts"].append(row["contexts"])
        rows["ground_truth"].append(row["ground_truth"])

    return Dataset.from_dict(rows)


def _build_ragas_llm():
    """RAGAS judge — OpenRouter when configured, else local Ollama."""
    load_dotenv(override=False)
    judge = os.getenv(
        "HERMES_RAGAS_JUDGE",
        "openrouter/google/gemini-2.5-flash-lite"
        if os.getenv("LLM_PROVIDER") == "openrouter"
        else "llama3.1:8b",
    )
    use_openrouter = (
        judge.startswith("openrouter/")
        or (os.getenv("OPENROUTER_API_KEY") and os.getenv("LLM_PROVIDER") == "openrouter")
    )
    if use_openrouter:
        if not judge.startswith("openrouter/"):
            judge = f"openrouter/{judge}"
        from langchain_community.chat_models import ChatLiteLLM

        print(f"RAGAS judge: {judge} (OpenRouter)")
        return LangchainLLMWrapper(ChatLiteLLM(model=judge, temperature=0))

    ollama_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    print(f"RAGAS judge: {judge} (Ollama)")
    return LangchainLLMWrapper(
        ChatOllama(model=judge, base_url=ollama_base, temperature=0)
    )


def _current_levers() -> dict:
    from src.rag.chunk_strategies import get_chunk_config

    cfg = get_chunk_config()
    return {
        "MIN_RERANK_SCORE": float(os.getenv("MIN_RERANK_SCORE", "0.0")),
        "RETRIEVAL_CANDIDATES": int(os.getenv("RETRIEVAL_CANDIDATES", "50")),
        "HERMES_MULTI_QUERY": os.getenv("HERMES_MULTI_QUERY", "1"),
        "HERMES_CRAG_LITE": os.getenv("HERMES_CRAG_LITE", "1"),
        "HERMES_GRAPH_RAG": os.getenv("HERMES_GRAPH_RAG", "0"),
        "CONTEXT_PACK_TOP_K": int(os.getenv("CONTEXT_PACK_TOP_K", "5")),
        "HERMES_SIMPLE_MODEL": os.getenv("HERMES_SIMPLE_MODEL", ""),
        "CHUNK_STRATEGY": os.getenv("CHUNK_STRATEGY", "fixed"),
        "CHILD_CHUNK_SIZE": cfg.child_chunk_size,
        "CHILD_CHUNK_OVERLAP": cfg.child_chunk_overlap,
        "EMBED_MODEL": os.getenv("EMBED_MODEL", "ollama"),
        "RERANK_MODEL": os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "RAGAS_MAX_WORKERS": int(os.getenv("RAGAS_MAX_WORKERS", "16")),
        "RAGAS_BUILD_WORKERS": int(os.getenv("RAGAS_BUILD_WORKERS", "8")),
        "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "ollama"),
        "OLLAMA_API_BASE": os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
        "judge": os.getenv("HERMES_RAGAS_JUDGE", "llama3.1:8b"),
    }


def run_evaluation(
    n_questions: int = 20,
    exp_name: str = "baseline",
    fresh_kb: bool = True,
) -> dict:
    print("\n" + "=" * 50)
    print("=== Hermes RAGAS Evaluation ===")
    print("=" * 50)

    # Avoid Ollama embed contention during eval (LLM may be on OpenRouter).
    reset_retriever()

    print("\nStep 1: Clearing Hermes cache...")
    _flush_hermes_cache()

    if fresh_kb:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        try:
            client.delete_collection("hermes_docs")
            print("Qdrant collection cleared")
        except Exception:
            pass
        reset_retriever()

        print("\nStep 2: Ingesting knowledge base...")
        retriever = get_retriever(use_cache=False)
        eval_meta = {"user_id": "eval", "workspace_id": "eval"}
        for url in load_kb_urls():
            # Stamp eval workspace so ACL-filtered queries can hit the same points.
            ingest_url(url, retriever, extra_metadata=eval_meta)
        for rel in load_kb_files():
            path = resolve_kb_file(rel)
            print(f"\nIngesting local file: {path}")
            with open(path, encoding="utf-8") as f:
                text = f.read()
            retriever.ingest(
                text=text,
                metadata={
                    "source": path,
                    "url": path,
                    "title": os.path.basename(path),
                    "page_num": 1,
                    "type": "file",
                    **eval_meta,
                },
            )
    else:
        print("\nStep 2: Skipping KB rebuild (--no-fresh-kb)")
        get_retriever(use_cache=False)

    print("\nStep 3: Running evaluation...")
    ollama_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

    local_llm = _build_ragas_llm()
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base)
    )

    faithfulness = Faithfulness()
    answer_relevancy = AnswerRelevancy()
    context_precision = ContextPrecision()
    context_recall = ContextRecall()
    faithfulness.llm = local_llm
    context_precision.llm = local_llm
    context_recall.llm = local_llm
    answer_relevancy.llm = local_llm
    answer_relevancy.embeddings = embeddings

    gold = load_golden()
    n_questions = min(n_questions, len(gold))
    print(f"\nBuilding eval dataset ({n_questions} questions)...")
    dataset = _build_dataset(n_questions)

    print("\nScoring with RAGAS...")
    # High concurrency for OpenRouter judge — no intentional sleeps.
    # Override with RAGAS_MAX_WORKERS (default 16).
    judge_workers = int(os.getenv("RAGAS_MAX_WORKERS", "12"))
    run_config = RunConfig(
        max_workers=judge_workers,
        timeout=int(os.getenv("RAGAS_TIMEOUT", "600")),
        max_retries=int(os.getenv("RAGAS_MAX_RETRIES", "3")),
    )
    print(f"RAGAS judge workers={judge_workers} retries={run_config.max_retries}")
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        run_config=run_config,
    )

    try:
        df = scores.to_pandas()
        df.to_json(DETAILS_PATH, orient="records", indent=2)
        print(f"\nDetailed rows saved to {DETAILS_PATH}")
    except Exception as e:
        print(f"Could not save details: {e}")
        df = None

    def _safe(val, col: str | None = None):
        """Average non-null metric values; avoid NaN when some rows fail."""
        if df is not None and col and col in df.columns:
            nums = []
            for x in df[col].tolist():
                try:
                    fx = float(x)
                except (TypeError, ValueError):
                    continue
                if fx == fx:  # not NaN
                    nums.append(fx)
            if nums:
                return round(sum(nums) / len(nums), 4)
        if isinstance(val, list):
            nums = [float(x) for x in val if x == x]
            return round(sum(nums) / len(nums), 4) if nums else None
        try:
            fx = float(val)
            return round(fx, 4) if fx == fx else None
        except (TypeError, ValueError):
            return None

    report = {
        "timestamp": datetime.now().isoformat(),
        "exp_name": exp_name,
        "questions_tested": n_questions,
        "levers": _current_levers(),
        "faithfulness": _safe(scores["faithfulness"], "faithfulness"),
        "answer_relevancy": _safe(scores["answer_relevancy"], "answer_relevancy"),
        "context_precision": _safe(scores["context_precision"], "context_precision"),
        "context_recall": _safe(scores["context_recall"], "context_recall"),
    }

    print("\n" + "=" * 50)
    print("HERMES RAGAS EVALUATION REPORT")
    print("=" * 50)
    for label, key, target in [
        ("Faithfulness:", "faithfulness", 0.83),
        ("Answer Relevancy:", "answer_relevancy", 0.80),
        ("Context Precision:", "context_precision", 0.80),
        ("Context Recall:", "context_recall", 0.80),
    ]:
        val = report[key]
        if val is None:
            print(f"{label:<22} nan")
        else:
            mark = "OK" if val >= target else f"below {target}"
            print(f"{label:<22} {val:.4f}  ({mark})")
    print("=" * 50)

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    exp_path = os.path.join(EXPERIMENTS_DIR, f"{exp_name}.json")
    with open(exp_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {REPORT_PATH} and {exp_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run Hermes RAGAS evaluation")
    parser.add_argument("--n", type=int, default=20, help="Number of gold questions")
    parser.add_argument("--exp-name", default="baseline", help="Experiment name")
    parser.add_argument(
        "--no-fresh-kb",
        action="store_true",
        help="Do not wipe/re-ingest Qdrant before scoring",
    )
    args = parser.parse_args()
    run_evaluation(
        n_questions=args.n,
        exp_name=args.exp_name,
        fresh_kb=not args.no_fresh_kb,
    )


if __name__ == "__main__":
    main()
