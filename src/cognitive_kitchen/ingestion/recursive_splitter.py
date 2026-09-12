from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from cognitive_kitchen.ingestion.schemas import DocumentChunk

class RecursiveRecipeChunker:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 40):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Hierarchical split order: Paragraphs -> Newlines -> Words -> Characters
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True,
        )

    def chunk_text(
        self, text: str, doc_id: str, metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        base_metadata = metadata or {}
        chunks: List[DocumentChunk] = []

        cleaned_text = text.strip()
        if not cleaned_text:
            return chunks

        # Perform the recursive splitting
        raw_splits = self.splitter.split_text(cleaned_text)

        for idx, split in enumerate(raw_splits):
            chunk_meta = base_metadata.copy()
            chunk_meta["chunk_index"] = idx
            chunk_meta["strategy"] = "recursive_character"

            # Coerce the output into our canonical Pydantic model
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}#rec_{idx}",
                    text=split.strip(),
                    metadata=chunk_meta,
                )
            )

        return chunks