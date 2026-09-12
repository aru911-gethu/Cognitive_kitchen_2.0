from cognitive_kitchen.ingestion.recursive_splitter import RecursiveRecipeChunker

# A sample text with double newlines separating recipes, and single newlines separating lines.
SAMPLE_TEXT = (
    "RECIPE: Tomato Soup\n"
    "Ingredients: Tomatoes, Water, Salt.\n"
    "Instructions: Boil everything.\n\n"
    "RECIPE: Grilled Cheese\n"
    "Ingredients: Bread, Cheese, Butter.\n"
    "Instructions: Grill until golden."
)

def run_test():
    chunker = RecursiveRecipeChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_text(SAMPLE_TEXT, doc_id="test_doc")

    print(f"Total Chunks Generated: {len(chunks)}\n")
    for c in chunks:
        print(f"--- Chunk ID: {c.chunk_id} ---")
        print(c.text)
        print("-------------------------------\n")

if __name__ == "__main__":
    run_test()