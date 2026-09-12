from cognitive_kitchen.ingestion.naive_chunker import NaiveChunker


def run_phase1a_breakage_demo():
    raw_cookbook_page = (
        "RECIPE: Quick Jeera Rice\n"
        "Ingredients: 1 cup basmati rice, 1 tsp cumin seeds, 2 cups water, 1 tbsp ghee.\n"
        "Instructions: Wash the rice. Heat ghee in a pressure cooker. Add cumin seeds.\n"
        "Add drained rice and water. Close the lid tightly and pressure cook for 4 whistles.\n"
        "\n"
        "RECIPE: Fresh Cucumber Raita\n"
        "Ingredients: 1 cup chilled yogurt, 1 grated cucumber, 1/4 tsp roasted cumin powder, salt to taste.\n"
        "Instructions: Whisk the yogurt in a bowl until smooth. Fold in grated cucumber and salt.\n"
        "Garnish with roasted cumin powder. Serve immediately chilled."
    )

    chunker = NaiveChunker(chunk_size=260, chunk_overlap=30)
    chunks = chunker.chunk_text(
        text=raw_cookbook_page,
        doc_id="cookbook_page_102",
        metadata={"source": "Recipe-Book.pdf", "page": 102},
    )

    print("=" * 70)
    print("PHASE 1a: NAIVE TOKEN/CHARACTER CHUNKING BREAKAGE DEMO")
    print("=" * 70)
    print(f"Total Chunks Produced: {len(chunks)}\n")

    for c in chunks:
        print(f"--- [Chunk ID: {c.chunk_id}] (Chars {c.metadata['char_start']} to {c.metadata['char_end']}) ---")
        print(c.text)
        print()

    blended_chunk = None
    for c in chunks:
        if "pressure cook for 4 whistles" in c.text and "Cucumber Raita" in c.text:
            blended_chunk = c
            break

    assert blended_chunk is not None
    print("=" * 70)
    print("REPRESENTATIVE FAILURE DETECTED:")
    print("=" * 70)
    print(f"Chunk '{blended_chunk.chunk_id}' stitched boiling rice instructions with Cucumber Raita!")
    print(f"A retriever fetching this chunk causes the LLM to output:")
    print("'Pressure cook the cucumber raita for 4 whistles.'")
    print("=" * 70)


if __name__ == "__main__":
    run_phase1a_breakage_demo()