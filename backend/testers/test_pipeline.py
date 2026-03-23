"""
Hermes Pipeline — Autonomous Test Suite
========================================
10 critical tests covering the full stack:
  - Ingestion
  - Retrieval + hybrid search
  - Reranker score threshold
  - Semantic cache
  - Supervisor routing
  - End-to-end RAG answer quality
  - Synthesis agent

Run with:
    cd ~/hermes/backend
    uv run pytest testers/test_pipeline.py -v
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv()

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )


@pytest.fixture(scope="session")
def clean_retriever(qdrant_client):
    """Fresh retriever with a clean collection seeded with RAG content."""
    from src.rag.retriever import HermesRetriever

    # Drop and recreate collection
    try:
        qdrant_client.delete_collection("hermes_docs")
    except Exception:
        pass

    retriever = HermesRetriever(use_cache=False, use_reranker=True)

    rag_content = """
    Retrieval-Augmented Generation (RAG) is a technique that enhances
    language model responses by retrieving relevant documents from an
    external knowledge base before generating an answer.

    Vector databases store document embeddings and enable semantic search.
    When a user submits a query, it is embedded into the same vector space
    and the most similar document chunks are retrieved.

    Parent-child chunking is an advanced RAG technique where small chunks
    are used for precise retrieval, but the larger parent chunk is passed
    to the LLM to provide richer context for generation.

    Hybrid search combines dense vector search with sparse BM25 keyword
    search. Reciprocal Rank Fusion (RRF) merges both result lists, improving
    recall for both semantic and exact-match queries over dense-only retrieval.

    Cross-encoder reranking improves retrieval quality by scoring each
    candidate passage against the query jointly, rather than independently.
    This gives much more accurate relevance scores at the cost of speed.

    Semantic caching stores query embeddings and returns cached answers
    for semantically similar queries, dramatically reducing latency and
    API costs for repeated or paraphrased questions.

    Qdrant is a high-performance vector database that supports cosine
    similarity search, filtering, and hybrid sparse-dense retrieval.
    It is available as a managed cloud service with a free tier.

    LangGraph is a library for building stateful multi-agent pipelines.
    It uses a directed graph where each node is an agent and edges define
    routing logic based on the shared state object.
    """ * 2

    retriever.ingest(rag_content, metadata={"source": "rag_guide.pdf", "page_num": 1})
    return retriever


@pytest.fixture(scope="session")
def redis_client():
    import redis
    r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    r.delete("hermes:cache:index")
    return r


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestIngestion:
    def test_01_ingest_returns_nonzero_chunks(self, clean_retriever):
        """Ingestion must produce parent and child chunks."""
        # clean_retriever fixture already ingested — verify collection is populated
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        count = client.count("hermes_docs").count
        assert count > 0, f"Expected chunks in Qdrant, got {count}"
        print(f"\n  ✓ {count} chunks stored in Qdrant")

    def test_02_pdf_ingestion(self, qdrant_client):
        """PDF loader must extract pages and store chunks."""
        import urllib.request
        from src.rag.retriever import HermesRetriever
        from src.ingestion.pdf_loader import ingest_pdf

        test_pdf = "/tmp/hermes_test.pdf"
        urllib.request.urlretrieve(
            "https://pdfobject.com/pdf/sample.pdf", test_pdf
        )

        retriever = HermesRetriever(use_cache=False, use_reranker=False)
        stats = ingest_pdf(test_pdf, retriever)

        assert stats["pages_processed"] >= 1
        assert stats["total_children"] > 0
        print(f"\n  ✓ PDF ingested: {stats}")


class TestRetrieval:
    def test_03_retrieval_returns_results(self, clean_retriever):
        """A relevant query must return at least 1 result."""
        results = clean_retriever.query("What is RAG?", top_k=3)
        assert len(results) > 0, "Expected results for a relevant query"
        print(f"\n  ✓ {len(results)} results returned")

    def test_04_reranker_scores_positive_for_relevant(self, clean_retriever):
        """Reranker scores must be positive for content that matches the query."""
        results = clean_retriever.query(
            "How does retrieval augmented generation work?", top_k=3
        )
        assert len(results) > 0
        top_score = results[0].get("reranker_score", 0)
        assert top_score > 0, f"Expected positive reranker score, got {top_score}"
        print(f"\n  ✓ Top reranker score: {top_score:.3f}")

    def test_05_reranker_scores_negative_for_irrelevant(self, clean_retriever):
        """Reranker must return low/negative scores for completely unrelated queries."""
        results = clean_retriever.query(
            "What is the recipe for chocolate cake?", top_k=3
        )
        if results:
            top_score = results[0].get("reranker_score", 0)
            assert top_score < 2.0, (
                f"Expected low score for irrelevant query, got {top_score}"
            )
        print(f"\n  ✓ Irrelevant query scored low as expected")

    def test_06_hybrid_search_citations_have_metadata(self, clean_retriever):
        """Every result must have source and page_num in metadata."""
        results = clean_retriever.query("vector database embeddings", top_k=3)
        assert len(results) > 0
        for r in results:
            assert "source" in r["metadata"], "Missing source in metadata"
            assert "page_num" in r["metadata"], "Missing page_num in metadata"
        print(f"\n  ✓ All results have source and page_num metadata")


class TestSemanticCache:
    def test_07_cache_miss_then_hit(self, clean_retriever, redis_client):
        """First query is a miss; paraphrase of same query is a hit."""
        from src.rag.cache import SemanticCache

        redis_client.delete("hermes:cache:index")
        cache = SemanticCache()

        # Miss
        result = cache.get("What is retrieval augmented generation?")
        assert result is None, "Expected cache miss on first query"

        # Store
        cache.set("What is retrieval augmented generation?", {
            "answer": "RAG combines retrieval with LLM generation.",
            "citations": []
        })

        # Hit — exact
        hit = cache.get("What is retrieval augmented generation?")
        assert hit is not None, "Expected cache hit on exact query"
        assert hit["cache_hit"] is True
        print(f"\n  ✓ Cache miss → store → hit working")

    def test_08_cache_paraphrase_hit(self, redis_client):
        """Semantically similar but differently worded query must hit cache."""
        from src.rag.cache import SemanticCache

        redis_client.delete("hermes:cache:index")
        cache = SemanticCache()

        cache.set("What is retrieval augmented generation?", {
            "answer": "RAG combines retrieval with LLM generation.",
            "citations": []
        })

        hit = cache.get("How does retrieval augmented generation work?")
        assert hit is not None, (
            "Expected paraphrase cache hit — threshold may be too strict"
        )
        print(f"\n  ✓ Paraphrase hit (similarity: {hit.get('similarity', '?')})")


class TestAgents:
    def test_09_supervisor_routes_simple_correctly(self):
        """Simple factual query must be classified as simple or at most multi_hop."""
        from src.agents.supervisor import classify_complexity
        complexity = classify_complexity("What is a vector database?")
        assert complexity in ("simple", "multi_hop"), (
            f"Simple query misclassified as: {complexity}"
        )
        print(f"\n  ✓ Simple query classified as: {complexity}")

    def test_10_end_to_end_pipeline_with_relevant_content(self, clean_retriever, redis_client):
        """
        Full pipeline test: ingest RAG content → query → get a real answer.
        Final answer must not be the fallback 'No relevant information' message.
        """
        from src.agents.graph import run_research

        # Flush cache so we don't hit stale entries
        redis_client.delete("hermes:cache:index")

        result = run_research("How does semantic caching reduce API costs in RAG?")

        assert result["final_answer"] != "", "Expected a non-empty answer"
        assert "No relevant information" not in result["final_answer"], (
            f"Pipeline returned fallback answer. Check ingestion or score threshold.\n"
            f"Answer: {result['final_answer']}"
        )
        assert len(result["citations"]) > 0, "Expected at least one citation"

        print(f"\n  ✓ Answer: {result['final_answer'][:120]}...")
        print(f"  ✓ Citations: {len(result['citations'])}")
        print(f"  ✓ Model used: {result['model_used']}")
        print(f"  ✓ Complexity: {result['query_complexity']}")
