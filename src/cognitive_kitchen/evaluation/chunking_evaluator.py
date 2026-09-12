from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from cognitive_kitchen.ingestion.naive_chunker import NaiveChunker
from cognitive_kitchen.ingestion.recursive_splitter import RecursiveRecipeChunker
from cognitive_kitchen.ingestion.semantic_chunker import LocalSemanticRecipeChunker
from cognitive_kitchen.ingestion.combined_chunker import CombinedSemanticRecursiveChunker


class ChunkingEvaluator:
    """Compare chunking strategies against the golden recipe dataset.

    This is intentionally simple and educational: we compare the strategies by
    measuring whether chunks stay within a single recipe and whether a recipe
    query retrieves the correct recipe chunks.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _load_golden_recipes(path: str | Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload["recipes"]

    @staticmethod
    def _recipe_text(recipe: Dict[str, Any]) -> str:
        title = recipe.get("title", "")
        ingredients = recipe.get("ingredients", [])
        instructions = recipe.get("instructions", [])

        lines = [title, "\nIngredients:"]
        for ingredient in ingredients:
            raw = ingredient.get("raw_text", "")
            if raw:
                lines.append(f"- {raw}")

        lines.append("\nInstructions:")
        for step in instructions:
            if isinstance(step, dict):
                text = step.get("instruction", "")
            else:
                text = str(step)
            if text:
                lines.append(f"{text}")

        return "\n".join(lines)

    @staticmethod
    def _queries_for_recipe(recipe: Dict[str, Any]) -> List[str]:
        title = recipe.get("title", "")
        ingredients = [item.get("raw_text", "") for item in recipe.get("ingredients", [])]
        query_candidates = [title]
        query_candidates.extend(ingredients[:3])

        # Keep only meaningful questions, and avoid duplicates.
        seen = set()
        results: List[str] = []
        for q in query_candidates:
            if q and q.lower() not in seen:
                seen.add(q.lower())
                results.append(q)
        return results

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        v1 = v1.astype(np.float32)
        v2 = v2.astype(np.float32)
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            return 0.0
        return float(np.dot(v1, v2) / denom)

    def _embed(self, texts: Iterable[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)

    @staticmethod
    def build_prompt_examples(recipes: List[Dict[str, Any]], limit: int = 6) -> List[str]:
        prompts: List[str] = []
        for recipe in recipes[:limit]:
            title = recipe.get("title", "")
            ingredients = recipe.get("ingredients", [])
            sample = ingredients[0].get("raw_text", "") if ingredients else ""
            if title:
                prompts.append(f"What is the recipe for {title}?")
            if sample:
                prompts.append(f"Which recipe contains '{sample}' and how is it prepared?")
        return prompts

    def evaluate_strategy(self, strategy_name: str, chunker: Any, recipes: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_chunks = []
        recipe_chunks: Dict[str, List[str]] = {}
        all_recipe_titles = [r["title"].lower() for r in recipes]

        for recipe in recipes:
            recipe_id = recipe["recipe_id"]
            text = self._recipe_text(recipe)
            chunks = chunker.chunk_text(text, doc_id=recipe_id, metadata={"recipe_id": recipe_id, "title": recipe["title"]})
            recipe_chunks[recipe_id] = [c.text for c in chunks]
            all_chunks.extend(chunks)

        contamination_count = 0
        for chunk in all_chunks:
            text = chunk.text.lower()
            matches = [title for title in all_recipe_titles if title in text]
            if len(matches) > 1:
                contamination_count += 1

        total_chunks = len(all_chunks)
        contamination_rate = contamination_count / total_chunks if total_chunks else 0.0
        isolation_score = 1.0 - contamination_rate

        retrieval_hits = 0
        top1_hits = 0
        retrieval_total = 0
        query_results: List[Dict[str, Any]] = []

        chunk_texts = [chunk.text for chunk in all_chunks]
        chunk_embeddings = self._embed(chunk_texts)

        for recipe in recipes:
            recipe_id = recipe["recipe_id"]
            queries = self._queries_for_recipe(recipe)
            for query in queries:
                retrieval_total += 1
                q_emb = self.model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
                similarities = [
                    self._cosine_similarity(q_emb, emb)
                    for emb in chunk_embeddings
                ]
                top_indices = list(np.argsort(similarities)[::-1][:3])
                top_recipe_id = all_chunks[top_indices[0]].metadata.get("recipe_id")
                retrieved_recipe_ids = {
                    all_chunks[idx].metadata.get("recipe_id")
                    for idx in top_indices
                }
                hit = recipe_id in retrieved_recipe_ids
                top1_hit = top_recipe_id == recipe_id
                retrieval_hits += int(hit)
                top1_hits += int(top1_hit)
                query_results.append(
                    {
                        "query": query,
                        "recipe_id": recipe_id,
                        "hit": hit,
                        "top1_hit": top1_hit,
                        "retrieved_recipe_ids": sorted(retrieved_recipe_ids),
                    }
                )

        recall = retrieval_hits / retrieval_total if retrieval_total else 0.0
        top1_rate = top1_hits / retrieval_total if retrieval_total else 0.0
        avg_chunk_count = sum(len(v) for v in recipe_chunks.values()) / max(1, len(recipe_chunks))
        avg_chunk_length = np.mean([len(text) for text in chunk_texts]) if chunk_texts else 0.0

        return {
            "strategy": strategy_name,
            "total_chunks": total_chunks,
            "avg_chunk_count_per_recipe": round(avg_chunk_count, 2),
            "avg_chunk_length": round(float(avg_chunk_length), 2),
            "contamination_rate": round(contamination_rate, 4),
            "isolation_score": round(isolation_score, 4),
            "retrieval_recall": round(recall, 4),
            "retrieval_top1_rate": round(top1_rate, 4),
            "query_results": query_results,
        }


def evaluate_all_strategies(golden_path: str | Path) -> List[Dict[str, Any]]:
    recipes = ChunkingEvaluator._load_golden_recipes(golden_path)
    evaluator = ChunkingEvaluator()

    strategies = {
        "naive": NaiveChunker(chunk_size=260, chunk_overlap=30),
        "recursive": RecursiveRecipeChunker(chunk_size=260, chunk_overlap=30),
        "semantic_plus_recursive": CombinedSemanticRecursiveChunker(
            RecursiveRecipeChunker(chunk_size=260, chunk_overlap=30),
            LocalSemanticRecipeChunker(distance_threshold=0.55, min_chunk_length=50),
        ),
    }

    results = [
        evaluator.evaluate_strategy(name, chunker, recipes)
        for name, chunker in strategies.items()
    ]
    return results
