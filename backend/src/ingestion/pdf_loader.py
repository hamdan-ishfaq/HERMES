"""
PDF Loader — extracts text from PDFs and ingests into Hermes retriever.

Uses pypdf for clean text-based PDFs.
Falls back to unstructured for scanned/complex layouts.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import pypdf
from src.rag.retriever import HermesRetriever

load_dotenv()


def extract_text_pypdf(pdf_path: str) -> list[dict]:
    """
    Extract text page by page using pypdf.
    Returns list of {text, page_num, source}.
    """
    pages = []
    reader = pypdf.PdfReader(pdf_path)
    source = Path(pdf_path).name

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "text": text.strip(),
                "page_num": i + 1,
                "source": source,
                "total_pages": len(reader.pages),
            })

    return pages


def extract_text_unstructured(pdf_path: str) -> list[dict]:
    """
    Fallback extractor using unstructured.
    Handles scanned PDFs, tables, complex layouts.
    """
    from unstructured.partition.pdf import partition_pdf

    source = Path(pdf_path).name
    elements = partition_pdf(pdf_path)

    # Group elements by page
    pages_dict: dict[int, list[str]] = {}
    for el in elements:
        page_num = el.metadata.page_number or 1
        pages_dict.setdefault(page_num, []).append(str(el))

    pages = []
    for page_num, texts in sorted(pages_dict.items()):
        combined = "\n".join(texts).strip()
        if combined:
            pages.append({
                "text": combined,
                "page_num": page_num,
                "source": source,
                "total_pages": max(pages_dict.keys()),
            })

    return pages


def load_pdf(pdf_path: str, use_unstructured: bool = False) -> list[dict]:
    """
    Load a PDF and return pages.
    Tries pypdf first, falls back to unstructured if needed.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if use_unstructured:
        print(f"Using unstructured for: {pdf_path}")
        return extract_text_unstructured(pdf_path)

    pages = extract_text_pypdf(pdf_path)

    # If pypdf got very little text, it's probably scanned — use unstructured
    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars < 500:
        print(f"pypdf got {total_chars} chars — switching to unstructured")
        return extract_text_unstructured(pdf_path)

    print(f"pypdf extracted {len(pages)} pages, {total_chars} chars")
    return pages


def ingest_pdf(
    pdf_path: str,
    retriever: HermesRetriever,
    extra_metadata: dict | None = None,
) -> dict:
    """
    Full pipeline: PDF → extract → chunk → embed → store in Qdrant.
    Returns ingestion stats.
    """
    print(f"\n{'='*50}")
    print(f"Ingesting: {pdf_path}")
    print(f"{'='*50}")

    pages = load_pdf(pdf_path)

    total_parents = 0
    total_children = 0

    for page in pages:
        stats = retriever.ingest(
            text=page["text"],
            metadata={
                "source": page["source"],
                "page_num": page["page_num"],
                "total_pages": page["total_pages"],
                "type": "pdf",
                **(extra_metadata or {}),
            }
        )
        total_parents += stats["parents"]
        total_children += stats["children"]

    result = {
        "source": Path(pdf_path).name,
        "pages_processed": len(pages),
        "total_parents": total_parents,
        "total_children": total_children,
    }

    print(f"\nDone: {result}")
    return result


if __name__ == "__main__":
    import sys

    # If a PDF path is passed as argument, use it
    # Otherwise create a quick test PDF
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Create a minimal test PDF using pypdf
        import pypdf
        from pypdf import PdfWriter

        writer = PdfWriter()
        # We'll use an existing simple approach - download a small test PDF
        import urllib.request
        test_pdf = "/tmp/test_hermes.pdf"
        print("Downloading test PDF...")
        urllib.request.urlretrieve(
            "https://www.w3.org/WAI/WCAG21/Techniques/pdf/sample.pdf",
            test_pdf
        )
        pdf_path = test_pdf

    retriever = HermesRetriever()

    # Ingest the PDF
    stats = ingest_pdf(pdf_path, retriever)

    # Ask a question about it
    print(f"\n--- Test Query ---")
    results = retriever.query("What is this document about?", top_k=3)

    for i, r in enumerate(results):
        print(f"\nResult {i+1} (score: {r['score']}):")
        print(f"  Source: {r['metadata'].get('source')} p.{r['metadata'].get('page_num')}")
        print(f"  Context: {r['context'][:150]}...")
