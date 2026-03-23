"""
Hierarchical parent-child chunker.

Strategy:
  - Parent chunks: ~1000 chars — sent to LLM for context
  - Child chunks:  ~200 chars  — used for precise retrieval

Each child stores its parent_id so we can fetch the full
parent context after retrieval.
"""

import uuid
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChildChunk:
    id: str
    text: str
    parent_id: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ParentChunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    children: list[ChildChunk] = field(default_factory=list)


class HierarchicalChunker:
    def __init__(
        self,
        parent_chunk_size: int = 1000,
        parent_chunk_overlap: int = 100,
        child_chunk_size: int = 200,
        child_chunk_overlap: int = 20,
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
        )

    def chunk(self, text: str, metadata: dict = {}) -> list[ParentChunk]:
        """
        Split text into parent chunks, then split each parent
        into child chunks. Returns list of ParentChunk objects,
        each containing their child chunks.
        """
        parent_texts = self.parent_splitter.split_text(text)
        parents = []

        for parent_text in parent_texts:
            parent_id = str(uuid.uuid4())
            parent = ParentChunk(
                id=parent_id,
                text=parent_text,
                metadata=metadata.copy(),
            )

            child_texts = self.child_splitter.split_text(parent_text)
            for child_text in child_texts:
                child = ChildChunk(
                    id=str(uuid.uuid4()),
                    text=child_text,
                    parent_id=parent_id,
                    metadata={**metadata, "parent_id": parent_id},
                )
                parent.children.append(child)

            parents.append(parent)

        return parents


if __name__ == "__main__":
    sample = """
    Retrieval-Augmented Generation (RAG) is a technique that combines
    information retrieval with text generation. Instead of relying solely
    on the knowledge encoded in model parameters, RAG retrieves relevant
    documents from an external knowledge base and uses them as context
    for generating answers.

    The key advantage of RAG is that it allows language models to access
    up-to-date information without retraining. The retrieval component
    searches a vector database for semantically similar documents, while
    the generation component synthesises a coherent answer from the
    retrieved context.

    Parent-child chunking improves RAG by separating retrieval precision
    from generation context. Small child chunks are indexed for precise
    retrieval, while their larger parent chunks are sent to the LLM to
    provide sufficient context for accurate answer generation.
    """ * 5  # repeat to get multiple chunks

    chunker = HierarchicalChunker()
    parents = chunker.chunk(sample, metadata={"source": "test"})

    print(f"Parents: {len(parents)}")
    total_children = sum(len(p.children) for p in parents)
    print(f"Children: {total_children}")
    print(f"\nFirst parent ({len(parents[0].text)} chars):")
    print(f"  {parents[0].text[:100]}...")
    print(f"\nFirst child ({len(parents[0].children[0].text)} chars):")
    print(f"  {parents[0].children[0].text[:100]}...")
    print(f"\nChild→Parent link: {parents[0].children[0].parent_id} == {parents[0].id}")
    print(f"Link valid: {parents[0].children[0].parent_id == parents[0].id}")
