import sys
import asyncio
import tempfile
import json
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# Windows asyncio fix for subprocesses (Playwright)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from cognitive_kitchen.ingestion.pdf_parser import PDFRecipeParser
from cognitive_kitchen.ingestion.category_scraper import CategoryMenuScraper
from cognitive_kitchen.ingestion.naive_chunker import NaiveChunker
from cognitive_kitchen.ingestion.recursive_splitter import RecursiveRecipeChunker
from cognitive_kitchen.ingestion.semantic_chunker import LocalSemanticRecipeChunker
from cognitive_kitchen.ingestion.combined_chunker import CombinedSemanticRecursiveChunker
from cognitive_kitchen.ingestion.schemas import DocumentChunk
from cognitive_kitchen.evaluation.chunking_evaluator import evaluate_all_strategies


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensures event loop policy persists during the app lifecycle."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    yield


app = FastAPI(
    title="CognitiveKitchen API",
    description="Ingestion, scraping, chunking, and AI chef assistant endpoints.",
    version="0.1.0",
    lifespan=lifespan,
)


class ScrapeRequest(BaseModel):
    category_url: str
    link_selector: str = "span a"
    max_items: int = 2


class ChunkRequest(BaseModel):
    text: str
    chunk_size: int = 260
    chunk_overlap: int = 30
    doc_id: str = "doc_manual_001"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


@app.get("/health")
def health_check():
    """Simple ping to verify server status."""
    return {"status": "healthy", "service": "CognitiveKitchen API"}


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """Accepts a cookbook PDF, parses layout blocks, and saves raw extraction to disk."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    parser = PDFRecipeParser(tmp_path)
    blocks = parser.extract_blocks()
    full_text = parser.get_full_text()

    # Define safe_stem properly to avoid NameError crash
    safe_stem = Path(file.filename).stem.replace(" ", "_").lower()
    storage_dir = Path("data/raw_pdfs")
    storage_dir.mkdir(parents=True, exist_ok=True)
    save_path = storage_dir / f"raw_pdf_{safe_stem}_{int(time.time())}.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_file": file.filename,
                "total_blocks": len(blocks),
                "total_characters": len(full_text),
                "blocks": blocks,
                "full_text": full_text,
            },
            f,
            indent=2,
        )

    return {
        "filename": file.filename,
        "total_blocks": len(blocks),
        "total_characters": len(full_text),
        "saved_to": str(save_path),
        "blocks": blocks,
        "full_text": full_text,
    }


@app.post("/ingest/url")
async def ingest_url(payload: ScrapeRequest):
    """Scrapes recipe pages asynchronously via worker thread to avoid blocking server."""
    if not payload.category_url or not payload.category_url.startswith("http"):
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid URL starting with http:// or https://",
        )

    scraper = CategoryMenuScraper(headless=True)
    try:
        # Offload synchronous Playwright calls so FastAPI event loop doesn't hang
        scraped_items = await asyncio.to_thread(
            scraper.scrape_category_pipeline,
            category_url=payload.category_url,
            link_selector=payload.link_selector,
            max_items=payload.max_items,
        )

        if not scraped_items:
            return {
                "source_category_url": payload.category_url,
                "extracted_count": 0,
                "saved_to": None,
                "recipe_headers": [],
            }

        recipe_headers = []
        for item in scraped_items:
            title = item.get("title", "").strip()
            if title and title.lower() != "untitled":
                recipe_headers.append(title)
            elif item.get("menu_items"):
                recipe_headers.append(item["menu_items"][0][:60])

        storage_dir = Path("data/raw_web")
        storage_dir.mkdir(parents=True, exist_ok=True)
        save_path = storage_dir / f"raw_web_{int(time.time())}.json"

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source": payload.category_url,
                    "count": len(scraped_items),
                    "headers": recipe_headers,
                    "items": scraped_items,
                },
                f,
                indent=2,
            )

        return {
            "source_category_url": payload.category_url,
            "extracted_count": len(recipe_headers),
            "total_pages": len(scraped_items),
            "saved_to": str(save_path),
            "recipe_headers": recipe_headers,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scraper error: {type(e).__name__} - {str(e)}",
        )


@app.post("/chunk/naive")
def chunk_text_naive(payload: ChunkRequest):
    """Baseline naive sliding window chunker endpoint."""
    chunker = NaiveChunker(chunk_size=payload.chunk_size, chunk_overlap=payload.chunk_overlap)
    chunks = chunker.chunk_text(text=payload.text, doc_id=payload.doc_id)
    return {
        "total_chunks": len(chunks),
        "chunks": [c.model_dump() for c in chunks],
    }


@app.post("/chunk/recursive")
def chunk_text_recursive(payload: ChunkRequest):
    """Structure-aware recursive chunking endpoint."""
    chunker = RecursiveRecipeChunker(chunk_size=payload.chunk_size, chunk_overlap=payload.chunk_overlap)
    chunks = chunker.chunk_text(text=payload.text, doc_id=payload.doc_id)
    return {
        "total_chunks": len(chunks),
        "chunks": [c.model_dump() for c in chunks],
    }


@app.post("/chunk/semantic")
def chunk_text_semantic(payload: ChunkRequest):
    """Semantic sentence-boundary chunking endpoint."""
    chunker = LocalSemanticRecipeChunker(distance_threshold=0.55, min_chunk_length=50)
    chunks = chunker.chunk_text(text=payload.text, doc_id=payload.doc_id)
    return {
        "total_chunks": len(chunks),
        "chunks": [c.model_dump() for c in chunks],
    }


@app.post("/chunk/combined")
def chunk_text_combined(payload: ChunkRequest):
    """Combined recursive + semantic chunking endpoint."""
    chunker = CombinedSemanticRecursiveChunker(
        recursive_splitter=RecursiveRecipeChunker(chunk_size=payload.chunk_size, chunk_overlap=payload.chunk_overlap),
        semantic_chunker=LocalSemanticRecipeChunker(distance_threshold=0.55, min_chunk_length=50),
    )
    chunks = chunker.chunk_text(text=payload.text, doc_id=payload.doc_id)
    return {
        "total_chunks": len(chunks),
        "chunks": [c.model_dump() for c in chunks],
    }


@app.get("/eval/chunking")
def chunking_evaluation():
    """Run the Qwen-relevant chunking comparison over the golden dataset."""
    try:
        results = evaluate_all_strategies("data/golden_datasets/golden_recipes.json")
        return {"results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")


@app.post("/chat/chef")
def chef_dialogue(payload: ChatRequest):
    """Lightweight rule-based conversational assistant endpoint."""
    user_prompt = payload.messages[-1].content if payload.messages else ""
    prompt_lower = user_prompt.lower()

    if "ingredient" in prompt_lower or "ingest" in prompt_lower:
        reply = (
            "I'm ready! You can feed me recipes two ways: upload a cookbook PDF in 'Feed the Chef' "
            "or drop a website URL. Once ingested, I'll organize your ingredients and plan meals."
        )
    elif "raita" in prompt_lower:
        reply = (
            "Careful! With naive chunking, my instructions were contaminated to pressure cook cucumber raita. "
            "Fresh raita requires raw cucumber folded into chilled yogurt—no pressure cooking!"
        )
    else:
        reply = (
            f"Chef Pierre at your service! I've noted: '{user_prompt}'. "
            "To plan meals accurately, feed me recipes through the 'Feed the Chef' panel."
        )

    return {"reply": reply}