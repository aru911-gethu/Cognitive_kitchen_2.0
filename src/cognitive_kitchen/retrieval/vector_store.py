from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from cognitive_kitchen.ingestion.schemas import DocumentChunk

VECTOR_STORE_DIR = Path("data/vector_store")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def chunk_to_document(chunk: DocumentChunk) -> Document:
    metadata = dict(chunk.metadata)
    metadata["chunk_id"] = chunk.chunk_id
    return Document(page_content=chunk.text, metadata=metadata)


def build_faiss_index(
    chunks: List[DocumentChunk],
    index_name: str = "naive_index",
) -> FAISS:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    persist_path = VECTOR_STORE_DIR / index_name

    documents = [chunk_to_document(c) for c in chunks]
    embeddings = get_embedding_model()

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )
    vector_store.save_local(str(persist_path))
    return vector_store


def load_faiss_index(index_name: str = "naive_index") -> FAISS:
    persist_path = VECTOR_STORE_DIR / index_name
    embeddings = get_embedding_model()

    return FAISS.load_local(
        folder_path=str(persist_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )