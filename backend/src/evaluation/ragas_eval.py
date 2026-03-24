"""
RAGAS evaluation for Hermes — Phase 5
100% Local Scoring Architecture: Ollama (llama3.1:8b) for all metrics.
Embeddings: nomic-embed-text via Ollama
"""

import os
import json
import redis
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.run_config import RunConfig

from src.evaluation.golden_dataset import GOLDEN_QA
from src.agents.graph import run_research
from src.rag.retriever import HermesRetriever
from src.ingestion.url_loader import ingest_url

def _build_dataset(n_questions: int) -> Dataset:
    questions = GOLDEN_QA[:n_questions]
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    
    redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

    for i, item in enumerate(questions):
        q = item["question"]
        print(f"  [{i+1}/{n_questions}] {q[:60]}...")

        # Flush cache right before the agent runs
        redis_client.flushdb()

        state = run_research(q)
        answer = state.get("final_answer") or state.get("draft_answer") or ""

        # Pull raw chunks straight from Qdrant
        raw_contexts = state.get("retrieved_contexts", [])
        contexts = [c.get("context", "") for c in raw_contexts if isinstance(c, dict) and "context" in c]
        
        if not contexts:
            contexts = ["No relevant context was found in the database."]

        rows["question"].append(q)
        rows["answer"].append(answer)
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(item["ground_truth"])

    return Dataset.from_dict(rows)

def run_evaluation(n_questions: int = 10):
    print("\n" + "=" * 50)
    print("=== Hermes Phase 5 — RAGAS Evaluation ===")
    print("=" * 50)
    
    print("\nStep 1: Clearing all stale state...")
    r = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
    r.flushdb()
    print("Redis flushed")
    
    from qdrant_client import QdrantClient
    client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
    try:
        client.delete_collection('hermes_docs')
        print("Qdrant collection cleared")
    except Exception:
        pass
        
    print("\nStep 2: Ingesting rich knowledge base...")
    retriever = HermesRetriever(use_cache=False, use_reranker=True)
    urls = [
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "https://qdrant.tech/articles/hybrid-search/",
        "https://www.sbert.net/docs/cross_encoder/usage/usage.html"
    ]
    for url in urls:
        ingest_url(url, retriever)
    
    print("\nStep 3: Running evaluation...")
    ollama_base = os.getenv("OLLAMA_API_BASE", "http://172.25.16.1:11434")

    # 100% Local Setup to bypass API limits
    local_llm = LangchainLLMWrapper(
        ChatOllama(model="llama3.1:8b", base_url=ollama_base, temperature=0)
    )

    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base)
    )

    faithfulness      = Faithfulness()
    answer_relevancy  = AnswerRelevancy()
    context_precision = ContextPrecision()
    context_recall    = ContextRecall()

    faithfulness.llm            = local_llm
    context_precision.llm       = local_llm
    context_recall.llm          = local_llm
    answer_relevancy.llm        = local_llm
    answer_relevancy.embeddings = embeddings

    print(f"\nBuilding eval dataset ({n_questions} questions)...")
    dataset = _build_dataset(n_questions)

    print("\nScoring with RAGAS (100% Local Ollama 8b)...")
    print("Note: This will take roughly 15-20 minutes on your RTX 4050. Do not press Ctrl+C.")
    
    # max_workers=1 protects the local GPU from timeouts. max_retries=5 protects against bad JSON.
    run_config = RunConfig(max_workers=1, timeout=600, max_retries=5)

    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        run_config=run_config
    )

    try:
        df = scores.to_pandas()
        df.to_json("eval_details.json", orient="records", indent=2)
        print("\n✅ Detailed row-by-row data saved to 'eval_details.json'")
    except Exception as e:
        print(f"\n⚠️ Could not save detailed pandas dataframe: {e}")

    def _safe(val):
        if isinstance(val, list):
            val = sum(val) / len(val) if val else 0.0
        return round(val, 4) if val == val else None

    report = {
        "timestamp":         datetime.now().isoformat(),
        "questions_tested":  n_questions,
        "faithfulness":      _safe(scores["faithfulness"]),
        "answer_relevancy":  _safe(scores["answer_relevancy"]),
        "context_precision": _safe(scores["context_precision"]),
        "context_recall":    _safe(scores["context_recall"]),
    }

    print("\n" + "=" * 50)
    print("HERMES RAGAS EVALUATION REPORT")
    print("=" * 50)
    print(f"Timestamp:          {report['timestamp']}")
    print(f"Questions tested:   {report['questions_tested']}")
    print("-" * 50)

    def fmt(label, val, target):
        if val is None:
            print(f"{label:<22} nan   ⚠️")
        else:
            status = "✅" if val >= target else f"❌ (target >{target})"
            print(f"{label:<22} {val:.4f}  {status}")

    fmt("Faithfulness:",      report["faithfulness"],      0.83)
    fmt("Answer Relevancy:",  report["answer_relevancy"],  0.80)
    fmt("Context Precision:", report["context_precision"], 0.80)
    fmt("Context Recall:",    report["context_recall"],    0.80)
    print("=" * 50)

    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report

if __name__ == "__main__":
    run_evaluation(n_questions=10)
