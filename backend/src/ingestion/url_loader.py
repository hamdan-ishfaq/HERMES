"""
URL Loader — scrapes web pages and ingests into Hermes retriever.

Uses trafilatura for clean article extraction (removes nav, ads, footers).
Falls back to raw HTML parsing if trafilatura gets nothing.
"""

import trafilatura
from src.rag.retriever import HermesRetriever


def fetch_url(url: str) -> tuple[str | None, str | None]:
    """Download and extract main content + title from a URL."""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None, None

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )

    title = None
    try:
        meta = trafilatura.extract_metadata(downloaded)
        if meta:
            title = meta.title
    except Exception:
        title = None

    return text, title


def ingest_url(url: str, retriever: HermesRetriever, extra_metadata: dict | None = None) -> dict:
    """Full pipeline: URL → extract → chunk → embed → store."""
    print(f"\n{'='*50}")
    print(f"Ingesting URL: {url}")
    print(f"{'='*50}")

    text, title = fetch_url(url)

    if not text or len(text.strip()) < 100:
        print(f"❌ Could not extract content from {url}")
        return {
            "url": url,
            "status": "failed",
            "error": f"Could not extract readable content from {url}",
            "pages_processed": 0,
            "total_children": 0,
        }

    print(f"Extracted {len(text)} chars")

    metadata = {
        "source": url,
        "url": url,
        "title": title,
        "page_num": 1,
        "type": "url",
        **(extra_metadata or {}),
    }
    stats = retriever.ingest(text=text, metadata=metadata)

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
