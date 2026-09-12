from __future__ import annotations

from typing import Any, Dict, List

from cognitive_kitchen.ingestion.schemas import DocumentChunk


class CombinedSemanticRecursiveChunker:
    """Compose recursive structure-based splits with semantic refinement.

    This keeps the recursive splitter as the structural first pass while letting
    the semantic splitter break apart clusters that are still semantically mixed.
    """

    def __init__(self, recursive_splitter: Any, semantic_chunker: Any):
        self.recursive = recursive_splitter
        self.semantic = semantic_chunker

    def chunk_text(self, text: str, doc_id: str, metadata: Dict[str, Any] | None = None) -> List[DocumentChunk]:
        initial_chunks = self.recursive.chunk_text(text, doc_id, metadata)
        refined_chunks: List[DocumentChunk] = []

        for chunk in initial_chunks:
            if not chunk.text.strip():
                continue

            subchunks = self.semantic.chunk_text(
                chunk.text,
                doc_id=f"{doc_id}_{chunk.chunk_id}",
                metadata={**(chunk.metadata or {}), "parent_chunk_id": chunk.chunk_id, "strategy": "semantic_plus_recursive"},
            )

            if not subchunks:
                refined_chunks.append(chunk)
                continue

            refined_chunks.extend(subchunks)

        return refined_chunks
