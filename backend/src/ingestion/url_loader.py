"""
URL Loader — scrapes web pages and ingests into Hermes retriever.

Uses trafilatura for clean article extraction (removes nav, ads, footers).
Falls back to raw HTML parsing if trafilatura gets nothing.
"""

import trafilatura
from src.rag.retriever import HermesRetriever


def fetch_url(url: str) -> str | None:
    """Download and extract main content from a URL."""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    return text


def ingest_url(url: str, retriever: HermesRetriever) -> dict:
    """Full pipeline: URL → extract → chunk → embed → store."""
    print(f"\n{'='*50}")
    print(f"Ingesting URL: {url}")
    print(f"{'='*50}")

    text = fetch_url(url)

    if not text or len(text.strip()) < 100:
        print(f"❌ Could not extract content from {url}")
        return {"url": url, "status": "failed", "pages_processed": 0, "total_children": 0}

    print(f"Extracted {len(text)} chars")

    stats = retriever.ingest(
        text=text,
        metadata={
            "source": url,
            "page_num": 1,
            "type": "url",
        }
    )

    result = {
        "url": url,
        "status": "ok",
        "chars_extracted": len(text),
        "total_parents": stats["parents"],
        "total_children": stats["children"],
    }
    print(f"Done: {result}")
    return result


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    retriever = HermesRetriever(use_cache=False, use_reranker=True)

    test_url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    stats = ingest_url(test_url, retriever)

    print("\n--- Test Query ---")
    results = retriever.query("What is retrieval augmented generation?", top_k=2)
    for i, r in enumerate(results):
        print(f"\nResult {i+1} (score: {r.get('reranker_score', r['score']):.3f}):")
        print(f"  Source: {r['metadata'].get('source')}")
        print(f"  {r['context'][:150]}...")
