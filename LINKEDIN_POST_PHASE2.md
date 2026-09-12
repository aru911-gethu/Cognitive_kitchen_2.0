# Phase 2: Why retrieval quality mattered more than the model

This was a real lesson in building RAG systems: the model was not the first bottleneck. The bottleneck was how recipe content was split and retrieved before it ever reached the model.

## The use case

I was building a recipe assistant for real-world cooking queries over a small but messy corpus of South Indian and regional Indian recipes.

The product scenario was simple:

- a user asks for a recipe based on ingredients they have at home
- the system finds the most relevant recipe from a cookbook / web corpus
- the model answers with the recipe and preparation steps

The key challenge was not raw generation quality. It was making sure the correct recipe context was retrieved in the first place.

A realistic prompt looked like this:

> “I have rice, lentils, ghee, pepper, and cumin at home. What breakfast dish does this match, and how is it prepared?”

This is exactly the kind of question a user would ask after reading a recipe book or looking through a kitchen pantry.

## The failure we saw

During testing, the retrieved context was not clean enough. The retriever pulled chunks from more than one recipe, and the model answered with the wrong dish.

This was the actual failure pattern in the app:

- Retrieved chunks included:
  - Himachali Chana Madra
  - Chennai Special Murukku Open Sandwich
- The model then produced an answer for the wrong recipe
- The user’s ingredients did not match the answer, but the model still responded confidently

That was the core problem.

This was not a “bad model” issue. It was a retrieval contamination issue.

The recipe context was mixed across multiple recipes, so the model had no clean boundary to reason over.

## What this meant technically

The system architecture was built around:

- ingestion from PDFs and web sources
- chunking strategies
- vector retrieval
- Qwen generation for recipe answers
- a benchmark layer to compare retrieval quality

The main code paths were:

- [app.py](app.py)
- [src/cognitive_kitchen/api/main.py](src/cognitive_kitchen/api/main.py)
- [src/cognitive_kitchen/evaluation/chunking_evaluator.py](src/cognitive_kitchen/evaluation/chunking_evaluator.py)
- [src/cognitive_kitchen/ingestion/naive_chunker.py](src/cognitive_kitchen/ingestion/naive_chunker.py)
- [src/cognitive_kitchen/ingestion/recursive_splitter.py](src/cognitive_kitchen/ingestion/recursive_splitter.py)
- [src/cognitive_kitchen/ingestion/semantic_chunker.py](src/cognitive_kitchen/ingestion/semantic_chunker.py)
- [src/cognitive_kitchen/ingestion/combined_chunker.py](src/cognitive_kitchen/ingestion/combined_chunker.py)

The insight was straightforward:

if retrieval is messy, the model cannot recover.

## The actual benchmark

I did not use a DeepEval judge or a general-purpose LLM-as-judge layer for the main story. This phase was intentionally focused on a custom retrieval benchmark grounded in the recipe dataset itself.

The metric used was retrieval recall.

This metric measures:

- how often the correct recipe is retrieved for a given recipe query
- how often the system finds the correct recipe in the top retrieved chunks

The actual benchmark results were:

- Naive: 78.1%
- Recursive: 84.4%
- Semantic + Recursive: 87.5%

This means:

- Naive chunking retrieved the correct recipe only 78 out of 100 times
- Recursive chunking improved to 84 out of 100
- Semantic + Recursive improved further to 88 out of 100

These numbers are not abstract. They reflect the real issue we were seeing: mixing recipes in retrieved context and returning the wrong answer.

## Why the numbers matter

This is the clearest demonstration that retrieval quality was the actual bottleneck.

The same model was used across all comparisons. The only thing changing was the chunking strategy.

That means the improvement was not due to a stronger model or a different prompt. It was due to cleaner segmentation and better isolation of recipe boundaries.

In other words:

- better chunking = better retrieval
- better retrieval = more reliable recipe answers
- more reliable recipe answers = fewer wrong matches

## The key lesson

A lot of teams focus on prompt quality or model choice first.

In this project, the bigger issue was recipe boundary fidelity.

When ingredient lists and cooking instructions were chopped into weak chunks, the retriever started mixing recipes together. Once that happened, the model could not reliably decide which recipe the user actually meant.

This was the real engineering insight from the phase:

retrieval quality is the hidden bottleneck in domain-specific RAG systems.

## The outcome

This phase gave us a clean, reproducible story:

- root cause identified
- issue demonstrated in a real user-style prompt
- chunking strategies compared
- benchmark validated with actual metrics
- the model kept fixed across runs so the comparison stayed fair

## Next phase

The next step is to keep the project clean and reproducible:

- maintain the same benchmark structure
- keep the prompt examples realistic and easy to understand
- keep the evaluation focused on retrieval quality
- make the project easy to run and explain in a portfolio or GitHub repo

This phase is a good checkpoint because it shows both:

1. the problem in human terms
2. the measurable improvement in engineering terms

That is the right story for a portfolio and a LinkedIn post.

---

## Short version for LinkedIn

I’ve been building a recipe-focused RAG system, and one important lesson emerged: the real bottleneck was not the model — it was retrieval quality.

I tested a realistic user prompt: “I have rice, lentils, ghee, pepper, and cumin at home. What breakfast dish does this match, and how is it prepared?”

The system retrieved chunks from multiple recipes and answered with the wrong dish. That exposed the real issue: mixed recipe context caused by weak chunking.

I compared chunking strategies and measured retrieval recall:

- Naive: 78.1%
- Recursive: 84.4%
- Semantic + Recursive: 87.5%

The same model was used across all runs, so the improvement came from cleaner segmentation and better recipe isolation, not from switching models.

This taught me a crucial lesson for RAG systems: if retrieval is messy, generation cannot recover.

Better chunking meant better recipe retrieval, fewer mixed context errors, and a much more reliable answer.

That is the project story I’m carrying forward into the next phase.
