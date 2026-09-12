from typing import List, Dict, Any
from cognitive_kitchen.ingestion.schemas import DocumentChunk


class NaiveChunker:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, doc_id: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        base_metadata = metadata or {}
        chunks: List[DocumentChunk] = []
        
        cleaned_text = text.strip()
        if not cleaned_text:
            return chunks

        step = self.chunk_size - self.chunk_overlap
        start = 0
        chunk_idx = 0

        while start < len(cleaned_text):
            end = start + self.chunk_size
            slice_text = cleaned_text[start:end]
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = chunk_idx
            chunk_metadata["char_start"] = start
            chunk_metadata["char_end"] = min(end, len(cleaned_text))

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}#chunk_{chunk_idx}",
                    text=slice_text,
                    metadata=chunk_metadata,
                )
            )

            start += step
            chunk_idx += 1

        return chunks