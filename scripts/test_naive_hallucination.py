from dotenv import load_dotenv

load_dotenv()

from cognitive_kitchen.ingestion.naive_chunker import NaiveChunker
from cognitive_kitchen.retrieval.vector_store import build_faiss_index
from cognitive_kitchen.retrieval.chain import build_chef_chain

# Real adjacent recipe text from the ingested dataset
INGESTED_CORPUS = (
    "RECIPE: Himachali Chana Madra\n"
    "Ingredients: 2 cups Curd beaten till smooth, 1 bowl White Chickpeas boiled, 2 tbsp Ghee, 1 tsp cumin seeds, turmeric, garam masala.\n"
    "Instructions: In a bowl mix beaten curd with turmeric, coriander, and garam masala. Heat ghee in a pan and temper cumin seeds and whole spices. "
    "Add curd mixture into pan and cook for 15-20 mins until ghee separates. Add boiled chickpeas and simmer.\n\n"
    "RECIPE: Chennai Special Murukku Open Sandwich\n"
    "Ingredients: 4 murukku, 1 onion chopped, 1 tomato chopped, 1 boiled potato chopped, 1/4 cup green chutney, 1/4 cup nylon sev, 1 cheese cube grated.\n"
    "Instructions: Take a plate and arrange murukku. Spread green chutney over murukku. Add onion, tomato and potato. Sprinkle chaat masala. "
    "Top with grated cheese and nylon sev. Serve immediately."
)


def run_experiment():
    print("==================================================")
    print("🧪 EXPERIMENT: Naive Chunking Context Bleed")
    print("==================================================")

    # 1. Run Naive Chunking
    chunker = NaiveChunker(chunk_size=320, chunk_overlap=40)
    chunks = chunker.chunk_text(INGESTED_CORPUS, doc_id="doc_chennai_ingest")
    print(f"\n1. Generated {len(chunks)} Naive Chunks.")

    # 2. Build FAISS index
    index_name = "test_naive_experiment"
    print(f"\n2. Building FAISS index '{index_name}' with MiniLM-L6-v2...")
    build_faiss_index(chunks=chunks, index_name=index_name)
    print("   Index built and saved to data/vector_store/")

    # 3. Build Chef Pierre Chain
    print("\n3. Initializing Chef Pierre LCEL Chain (OpenAI gpt-4o-mini)...")
    chain, retriever = build_chef_chain(index_name=index_name, k=2)

    query = "How do I prepare and assemble the Murukku Open Sandwich?"
    print(f"\nUser Query: '{query}'")

    # 4. Inspect Retrieved Chunks (Root Cause)
    retrieved_docs = retriever.invoke(query)
    print("\n--- RETRIEVED CHUNKS (Top-2 Similarity) ---")
    for i, doc in enumerate(retrieved_docs, start=1):
        cid = doc.metadata.get("chunk_id", "unknown")
        print(f"\n[Chunk #{i} | ID: {cid}]")
        print(doc.page_content)

    # 5. Run Generation
    print("\n--- CHEF PIERRE'S GENERATED RESPONSE ---")
    response = chain.invoke(query)
    print(response)
    print("==================================================")


if __name__ == "__main__":
    run_experiment()