# Phase 2 checkpoint

## Goal
This branch captures the chunking and retrieval-quality checkpoint.

## Problem identified
The naive chunker was mixing recipe content and retrieving the wrong recipe context for ingredient-led queries.

## Benchmark results
- Naive: 78.1%
- Recursive: 84.4%
- Semantic + Recursive: 87.5%

## Conclusion
The retrieval bottleneck was chunk quality, not the model itself. The same model performed better once recipe boundaries were preserved.
