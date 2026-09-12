from cognitive_kitchen.ingestion.semantic_chunker import LocalSemanticRecipeChunker

# A messy paragraph with no \n\n structure, simulating a bad PDF extraction.
SAMPLE_TEXT = (
    "To make tomato soup, you need fresh tomatoes, water, and a pinch of salt. "
    "Boil all the ingredients together in a large pot for twenty minutes until soft. "
    "For a classic grilled cheese sandwich, grab two slices of sourdough bread and cheddar cheese. "
    "Butter the outside of the bread and grill it on a skillet until golden brown and melted."
)

def run_test():
    print("Loading SentenceTransformer model into memory (this takes a few seconds)...")
    
    # We set a threshold of 0.55. Any distance higher than this triggers a split.
    chunker = LocalSemanticRecipeChunker(distance_threshold=0.55, min_chunk_length=50)
    chunks = chunker.chunk_text(SAMPLE_TEXT, doc_id="sem_test")

    print(f"\nTotal Chunks Generated: {len(chunks)}\n")
    for c in chunks:
        print(f"--- Chunk ID: {c.chunk_id} ---")
        print(c.text)
        print("-------------------------------\n")

if __name__ == "__main__":
    run_test()