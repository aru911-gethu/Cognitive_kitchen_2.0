# Cognitive Kitchen: Phase 3 Retrieval Architecture Story

I’ve been building a recipe-focused RAG system to explore a simple but important question:

Can better retrieval improve the quality of real cooking recommendations, or does the real bottleneck appear later in the pipeline?

I started with a naive baseline and deliberately kept the model fixed while changing the retrieval setup. That allowed me to isolate the actual variable: how the recipes were chunked and retrieved.

I compared several retrieval strategies:
- naive chunking
- recursive chunking
- recursive + semantic chunking
- hybrid retrieval with reranking

The first issue I hit was not model quality — it was recipe contamination.

The retrieved chunks were mixing ingredients and instructions from different dishes, so the system could surface a relevant recipe family but still generate a partially incorrect answer.

For a prompt like:

“What can I make with chickpeas and yoghurt?”

the system retrieved a strong candidate from Himachali Chana Madra, but the final answer still drifted and blended recipe details. This exposed a key problem: the system could retrieve a good recipe candidate, but it still struggled with natural-language instructions and real-world recipe constraints.

In other words, retrieval success and answer fidelity are not the same thing.

The retriever could find a plausible recipe, but the generator did not stay firmly grounded in that recipe’s exact ingredients, method, and intent. It introduced generic cooking steps, mixed recipe patterns, and lost track of the actual constraint set in the user prompt.

This is the exact gap that made the next iteration necessary.

I then benchmarked the retrieval strategies against the same set of recipe questions.

The results were measurable and meaningful:
- Naive baseline: 78.1% of queries returned the correct recipe in the shortlist
- Recursive chunking: 84.4%
- Hybrid / reranked retrieval: 87.5%

Top-1 ranking also improved materially:
- Naive: 59.4%
- Recursive: 65.6%
- Hybrid / reranked: 81.2%

These improvements matter because they show that better retrieval directly impacts answer quality. If the correct recipe is not in the retrieved context, the model has almost no chance of being correct.

But the key insight is that retrieval quality alone was not enough.

Even when the correct recipe was in the shortlist, the system still struggled with more natural, multi-constraint prompts such as:
- “cold, vegetarian, and not a curry”
- “yogurt and cucumber, served chilled”
- “ingredients include chickpeas but no heavy gravy”

The observed behavior was very specific:
- the correct recipe often appeared in the top results
- but it was not always ranked first
- and sometimes the system still chose a recipe that matched only one keyword instead of the full constraint set

For example:
- For “yogurt and cucumber, served chilled,” the system often retrieved the right recipe but not in the correct rank
- For “served cold and uses chickpeas,” chickpea-heavy alternatives outranked the actual match because the model focused too strongly on one ingredient cue
- For “vegetarian, mint and yogurt, not a curry,” a curry recipe still appeared too high because the negative constraint was not respected

This is the main gap I found.

The system was getting better at finding candidate recipes, but not yet good enough at respecting full user intent.

That revealed the next missing layer:
- query decomposition
- negative constraint handling
- recipe attribute filtering
- structured metadata reasoning
- knowledge-graph or schema-based recipe grounding

In other words, the system needs to understand not just semantic similarity, but also the actual user constraints: ingredients, temperature, vegetarian status, dish type, and preparation style.

This project reinforced a core lesson in AI systems:

better retrieval helps a lot, but robust answers require more than semantic matching. They require structured reasoning over user intent and recipe attributes.

The next phase is about moving from “good retrieval” to “reliable recipe decision-making.”

That means building a stronger reasoning layer that can parse the user question, extract constraints, filter candidate recipes by metadata, and then generate a grounded answer that remains faithful to the actual recipe.

That is the next step in the project, and it is the point where retrieval stops being a shallow improvement and becomes a truly useful cooking assistant.

#AI #RAG #LLM #MachineLearning #RecipeAI #RetrievalAugmentedGeneration #Python #AIProduct #KnowledgeGraphs #PortfolioProject
