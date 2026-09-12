from pathlib import Path

from langchain_core.documents import Document

from cognitive_kitchen.retrieval.hybrid_retriever import HybridRecipeRetriever


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
]


def main():
    retriever = HybridRecipeRetriever(SAMPLE_DOCS, k=3)

    queries = [
        "Which recipe contains curd and chickpeas?",
        "Find the salad with yogurt and cucumber.",
        "What recipe is served cold and uses chickpeas?",
    ]

    for query in queries:
        print(f"\nQUERY: {query}")
        docs = retriever.retrieve(query, k=3)
        for idx, doc in enumerate(docs, start=1):
            print(f"  {idx}. {doc.metadata.get('title')} :: {doc.page_content[:120]}")


if __name__ == "__main__":
    main()
