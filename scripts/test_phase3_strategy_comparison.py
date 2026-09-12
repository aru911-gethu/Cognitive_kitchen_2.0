from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from cognitive_kitchen.retrieval.hybrid_retriever import HybridRecipeRetriever


@dataclass
class EvalCase:
    name: str
    query: str
    expected: str


CASES: List[EvalCase] = [
    EvalCase(
        name="exact_ingredient_match",
        query="Which recipe contains curd and chickpeas?",
        expected="Curd Chickpea Salad",
    ),
    EvalCase(
        name="paraphrase_yogurt_cucumber",
        query="Find the recipe with yogurt and cucumber that is served cold.",
        expected="Mint Yogurt Raita",
    ),
    EvalCase(
        name="cold_chickpea_recipe",
        query="What recipe is served cold and uses chickpeas?",
        expected="Curd Chickpea Salad",
    ),
    EvalCase(
        name="mint_and_yogurt",
        query="Which vegetarian recipe with mint and yogurt is not a curry?",
        expected="Mint Yogurt Raita",
    ),
]


SAMPLE_DOCS = [
    Document(
        page_content="Recipe: Curd Chickpea Salad. Mix chilled yogurt with chickpeas, cucumber, mint, and a pinch of salt. Serve cold.",
        metadata={"recipe_id": "rc_001", "title": "Curd Chickpea Salad"},
    ),
    Document(
        page_content="Recipe: Veggie Coconut Curry. Saute onion, garlic, and green beans; simmer with coconut milk. Add spinach at the end.",
        metadata={"recipe_id": "rc_002", "title": "Veggie Coconut Curry"},
    ),
    Document(
        page_content="Recipe: Tomato Rice Bowl. Cook rice, top with tomato, onion, and basil, then serve warm.",
        metadata={"recipe_id": "rc_003", "title": "Tomato Rice Bowl"},
    ),
    Document(
        page_content="Recipe: Mint Yogurt Raita. Whisk yogurt with cucumber, mint, and a little lemon juice. Serve chilled.",
        metadata={"recipe_id": "rc_004", "title": "Mint Yogurt Raita"},
    ),
    Document(
        page_content="Recipe: Chickpea Spinach Stew. Simmer chickpeas with spinach and garlic. It is served warm.",
        metadata={"recipe_id": "rc_005", "title": "Chickpea Spinach Stew"},
    ),
]


def simple_semantic_top3(query: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [d.page_content for d in SAMPLE_DOCS]
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    sims = (embeddings @ q)
    ranked = sorted(range(len(SAMPLE_DOCS)), key=lambda i: sims[i], reverse=True)
    return [SAMPLE_DOCS[i].metadata.get("title") for i in ranked[:3]]


def baseline_naive_top3(query: str):
    q = query.lower()
    scored = []
    for doc in SAMPLE_DOCS:
        content = doc.page_content.lower()
        score = sum(1 for token in ["curd", "chickpeas", "yogurt", "cucumber", "mint", "vegetarian", "curry", "cold"] if token in q and token in content)
        scored.append((score, doc.metadata.get("title")))
    return [name for _, name in sorted(scored, reverse=True)[:3]]


def compare():
    print("Phase 3 strategy comparison")
    print("=" * 90)
    for case in CASES:
        print(f"CASE: {case.name}")
        print(f"QUERY: {case.query}")
        print(f"EXPECTED: {case.expected}")
        print(f"NAIVE: {baseline_naive_top3(case.query)}")
        print(f"SEMANTIC: {simple_semantic_top3(case.query)}")
        hybrid = HybridRecipeRetriever(SAMPLE_DOCS, k=5)
        hybrid_top = [d.metadata.get("title") for d in hybrid.retrieve(case.query, k=3)]
        print(f"HYBRID: {hybrid_top}")
        print(f"TOP1_MATCH: {hybrid_top[0] == case.expected}")
        print("-" * 90)


if __name__ == "__main__":
    compare()
