from pathlib import Path

from cognitive_kitchen.evaluation.chunking_evaluator import evaluate_all_strategies


def main():
    golden_path = Path("data/golden_datasets/golden_recipes.json")
    results = evaluate_all_strategies(golden_path)

    print("Chunking Strategy Comparison")
    print("-" * 80)
    for result in results:
        print(f"Strategy: {result['strategy']}")
        print(f"  Total chunks: {result['total_chunks']}")
        print(f"  Avg chunks per recipe: {result['avg_chunk_count_per_recipe']}")
        print(f"  Avg chunk length: {result['avg_chunk_length']}")
        print(f"  Isolation score: {result['isolation_score']}")
        print(f"  Contamination rate: {result['contamination_rate']}")
        print(f"  Retrieval recall: {result['retrieval_recall']}")
        print("")


if __name__ == "__main__":
    main()
