from cognitive_kitchen.ingestion.schemas import Ingredient, Recipe, DocumentChunk


def test_schema_coercion():
    raw_recipe_data = {
        "recipe_id": "south-indian-sambar-001",
        "title": "Classic Sambar",
        "cook_time_minutes": "25",
        "prep_time_minutes": 10,
        "ingredients": [
            {
                "name": "toor dal",
                "quantity": "100.0",
                "unit": "grams",
                "is_core": True,
            },
            {
                "name": "tomato",
                "quantity": 2,
                "unit": "pieces",
                "is_core": True,
            },
        ],
        "instructions": [
            "Pressure cook toor dal with turmeric until soft.",
            "Cook vegetables in tamarind water and sambar powder.",
            "Combine dal and vegetable mixture; temper with mustard seeds and curry leaves.",
        ],
        "dietary_tags": ["vegetarian", "high-fiber"],
    }

    recipe = Recipe(**raw_recipe_data)

    assert isinstance(recipe.cook_time_minutes, int)
    assert recipe.cook_time_minutes == 25
    assert isinstance(recipe.ingredients[0].quantity, float)
    assert recipe.ingredients[0].quantity == 100.0
    assert len(recipe.ingredients) == 2

    chunk = DocumentChunk(
        chunk_id="cookbook_vol1#001",
        text="Pressure cook toor dal with turmeric until soft.",
        metadata={"recipe_id": recipe.recipe_id, "page": 12},
    )

    assert chunk.metadata["page"] == 12

    print("All assertions passed.")
    print(f"Validated: {recipe.title}")


if __name__ == "__main__":
    test_schema_coercion()