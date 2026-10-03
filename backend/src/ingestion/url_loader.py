"""
URL Loader — scrapes web pages and ingests into Hermes retriever.

Fetches HTML with httpx and hands it to trafilatura for clean article
extraction (removes nav, ads, footers).

Redirects are followed manually and every hop is re-validated against
``url_guard.validate_public_url``, because a public URL that 302s to
http://127.0.0.1/ would otherwise walk straight past an entry-point check.
"""

import httpx
import trafilatura
from urllib.parse import urljoin

from src.ingestion.url_guard import validate_public_url
from src.rag.retriever import HermesRetriever

FETCH_TIMEOUT = 15.0
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 3
REDIRECT_STATUSES = (301, 302, 303, 307, 308)


def _download_html(url: str) -> str | None:
    """
    Fetch ``url`` and return its HTML, or None if it cannot be retrieved.

    Returns None when the response is a redirect without a Location header or
    the body exceeds MAX_RESPONSE_BYTES.
    """
    # RESIDUAL RISK (DNS rebinding): validate_public_url resolves the hostname
    # here, but httpx resolves it again independently when it opens the socket.
    # An attacker controlling DNS with a very low TTL can answer with a public
    # address for the check and a private one for the connect. Closing that
    # fully requires resolving once and pinning the IP for the connection
    # (e.g. a custom transport that dials the validated address with a
    # matching Host header / SNI). This narrows the window; it does not close it.
    current = url
    with httpx.Client(follow_redirects=False, timeout=FETCH_TIMEOUT) as client:
        for _ in range(MAX_REDIRECTS + 1):
            validate_public_url(current)
            with client.stream("GET", current) as resp:
                if resp.status_code in REDIRECT_STATUSES:
                    location = resp.headers.get("location")
                    if not location:
                        return None
                    current = urljoin(current, location)
                    continue

                resp.raise_for_status()

                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > MAX_RESPONSE_BYTES:
                        print(f"Response exceeded {MAX_RESPONSE_BYTES} bytes; discarding.")
                        return None
                    chunks.append(chunk)

                return b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")

    raise ValueError(f"Exceeded {MAX_REDIRECTS} redirects while fetching {url}.")


def fetch_url(url: str) -> tuple[str | None, str | None]:
    """Download and extract main content + title from a URL."""
    downloaded = _download_html(url)
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
