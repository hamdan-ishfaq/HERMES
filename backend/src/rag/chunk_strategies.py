"""
Multi-strategy chunking — fixed parent/child sizes and semantic boundaries.

Select via ``CHUNK_STRATEGY`` env: fixed | fixed_large | fixed_small | semantic
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.chunker import ChildChunk, HierarchicalChunker, ParentChunk


@dataclass
class ChunkConfig:
    parent_chunk_size: int
    parent_chunk_overlap: int
    child_chunk_size: int
    child_chunk_overlap: int
    semantic_max_chars: int = 512


STRATEGY_CONFIGS: dict[str, ChunkConfig] = {
    "fixed": ChunkConfig(1000, 100, 200, 40),
    "fixed_large": ChunkConfig(1200, 120, 150, 50),
    "fixed_small": ChunkConfig(800, 80, 256, 64),
    "semantic": ChunkConfig(1500, 150, 400, 80, semantic_max_chars=512),
}


def get_chunk_config() -> ChunkConfig:
    strategy = os.getenv("CHUNK_STRATEGY", "fixed").strip().lower()
    if strategy in STRATEGY_CONFIGS:
        return STRATEGY_CONFIGS[strategy]
    return ChunkConfig(
        parent_chunk_size=int(os.getenv("PARENT_CHUNK_SIZE", "1000")),
        parent_chunk_overlap=int(os.getenv("PARENT_CHUNK_OVERLAP", "100")),
        child_chunk_size=int(os.getenv("CHILD_CHUNK_SIZE", "200")),
        child_chunk_overlap=int(os.getenv("CHILD_CHUNK_OVERLAP", "40")),
    )


def build_chunker() -> HierarchicalChunker | "SemanticChunker":
    strategy = os.getenv("CHUNK_STRATEGY", "fixed").strip().lower()
    cfg = get_chunk_config()
    if strategy == "semantic":
        return SemanticChunker(cfg)
    return HierarchicalChunker(
        parent_chunk_size=cfg.parent_chunk_size,
        parent_chunk_overlap=cfg.parent_chunk_overlap,
        child_chunk_size=cfg.child_chunk_size,
        child_chunk_overlap=cfg.child_chunk_overlap,
    )


class SemanticChunker:
    """Paragraph-aware parent chunks with smaller searchable children."""

    def __init__(self, cfg: ChunkConfig):
        self.cfg = cfg
        self._para_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.semantic_max_chars,
            chunk_overlap=cfg.parent_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.child_chunk_size,
            chunk_overlap=cfg.child_chunk_overlap,
            separators=["\n", ". ", ", ", " ", ""],
        )

    def chunk(self, text: str, metadata: dict | None = None) -> list[ParentChunk]:
        metadata = metadata or {}
        text = re.sub(r"\n{3,}", "\n\n", text.strip())
        parent_texts = self._para_splitter.split_text(text)
        parents: list[ParentChunk] = []

        for parent_text in parent_texts:
            if len(parent_text.strip()) < 40:
                continue
            parent_id = str(uuid.uuid4())
            parent = ParentChunk(id=parent_id, text=parent_text, metadata=metadata.copy())
            child_texts = self._child_splitter.split_text(parent_text)
            for child_text in child_texts:
                if len(child_text.strip()) < 20:
                    continue
                parent.children.append(
                    ChildChunk(
                        id=str(uuid.uuid4()),
                        text=child_text,
                        parent_id=parent_id,
                        metadata={**metadata, "parent_id": parent_id},
                    )
                )
            if parent.children:
                parents.append(parent)
        return parents
