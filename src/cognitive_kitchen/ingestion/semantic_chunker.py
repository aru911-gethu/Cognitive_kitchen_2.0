import re
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from cognitive_kitchen.ingestion.schemas import DocumentChunk

class LocalSemanticRecipeChunker:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        distance_threshold: float = 0.55,
        min_chunk_length: int = 50,
    ):
        # We load the exact same lightweight model used in our vector store
        self.model = SentenceTransformer(model_name)
        self.distance_threshold = distance_threshold
        self.min_chunk_length = min_chunk_length

    def _split_into_sentences(self, text: str) -> List[str]:
        # Splits text strictly by punctuation marks, handling messy PDF outputs
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        return sentences

    def _cosine_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculates distance between two vectors. Higher = more different."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 1.0
        return float(1.0 - (dot_product / (norm_v1 * norm_v2)))

    def chunk_text(
        self, text: str, doc_id: str, metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        base_metadata = metadata or {}
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []
        if len(sentences) == 1:
            return [DocumentChunk(chunk_id=f"{doc_id}#sem_0", text=sentences[0], metadata=base_metadata)]

        # Generate embeddings for every single sentence in one batch
        embeddings = self.model.encode(sentences)

        chunks = []
        current_chunk = [sentences[0]]
        current_length = len(sentences[0])

        for i in range(len(sentences) - 1):
            dist = self._cosine_distance(embeddings[i], embeddings[i + 1])
            
            # If the semantic distance spikes above our threshold, we've hit a new topic/recipe
            if dist >= self.distance_threshold and current_length >= self.min_chunk_length:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}#sem_{len(chunks)}",
                        text=chunk_text,
                        metadata={**base_metadata, "chunk_index": len(chunks), "strategy": "semantic"}
                    )
                )
                current_chunk = []
                current_length = 0

            current_chunk.append(sentences[i + 1])
            current_length += len(sentences[i + 1])

        # Append whatever is left in the buffer
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}#sem_{len(chunks)}",
                    text=chunk_text,
                    metadata={**base_metadata, "chunk_index": len(chunks), "strategy": "semantic"}
                )
            )

        return chunks