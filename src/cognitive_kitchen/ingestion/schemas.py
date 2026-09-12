from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Ingredient(BaseModel):
    name: str = Field(
        ...,
        description="Standardized name of the ingredient (e.g., 'roma tomato', 'toor dal')",
    )
    quantity: float = Field(
        ...,
        gt=0,
        description="Numeric amount needed. Must be strictly positive.",
    )
    unit: str = Field(
        ...,
        description="Standard unit of measurement (e.g., 'grams', 'pieces', 'tbsp')",
    )
    is_core: bool = Field(
        default=True,
        description="If True, the recipe mechanically fails without this item.",
    )


class Recipe(BaseModel):
    recipe_id: str = Field(
        ...,
        description="Unique deterministic identifier (slug or hash).",
    )
    title: str = Field(
        ...,
        description="Official title of the recipe.",
    )
    cuisine: Optional[str] = Field(
        default=None,
        description="Culinary style (e.g., 'South Indian', 'Italian').",
    )
    prep_time_minutes: int = Field(
        default=0,
        ge=0,
        description="Preparation time in minutes.",
    )
    cook_time_minutes: int = Field(
        default=0,
        ge=0,
        description="Cooking time in minutes.",
    )
    ingredients: List[Ingredient] = Field(
        default_factory=list,
        description="List of ingredients with quantities and units.",
    )
    instructions: List[str] = Field(
        default_factory=list,
        description="Sequential step-by-step preparation instructions.",
    )
    dietary_tags: List[str] = Field(
        default_factory=list,
        description="Tags like 'vegan', 'high-protein', 'gluten-free'.",
    )


class DocumentChunk(BaseModel):
    chunk_id: str = Field(
        ...,
        description="Unique ID for the chunk (e.g., doc_id#chunk_idx).",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="The raw textual slice.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual metadata (page number, source type, recipe reference).",
    )