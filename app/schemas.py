"""Request and response models. Limits come from ``app.config``."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .config import (
    MAX_B64_CHARS,
    MAX_GALLERY,
    MAX_ID_CHARS,
    MAX_ITEM_CHARS,
    MAX_ITEMS,
    MAX_TAG_CHARS,
    MAX_TAGS,
    MAX_TEXT_CHARS,
    MAX_TOP_K,
)

B64Image = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_B64_CHARS,
        description='Base64-encoded image. An optional "data:image/<type>;base64," prefix is accepted.',
    ),
]
Text = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARS)]
Item = Annotated[str, Field(min_length=1, max_length=MAX_ITEM_CHARS)]
Tag = Annotated[str, Field(min_length=1, max_length=MAX_TAG_CHARS)]
TextModel = Literal["use", "fallback"]


# ---- requests ---------------------------------------------------------------
class ClassifyRequest(BaseModel):
    image_b64: B64Image


class GalleryImage(BaseModel):
    id: Annotated[str, Field(min_length=1, max_length=MAX_ID_CHARS)]
    b64: B64Image


class VisionSearchRequest(BaseModel):
    query_b64: B64Image
    gallery: Annotated[list[GalleryImage], Field(min_length=1, max_length=MAX_GALLERY)]
    top_k: Annotated[int, Field(ge=1, le=MAX_TOP_K)] = 3


class SimilarityRequest(BaseModel):
    a: Text
    b: Text


class DuplicateRequest(BaseModel):
    items: Annotated[list[Item], Field(max_length=MAX_ITEMS)]
    threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.82


class TagSuggestRequest(BaseModel):
    text: Text
    tags: Annotated[list[Tag], Field(max_length=MAX_TAGS, description="Candidate tags to rank.")]
    top_k: Annotated[int, Field(ge=1, le=MAX_TOP_K)] = 5


# ---- responses --------------------------------------------------------------
class ClassPrediction(BaseModel):
    label: str = Field(description="ImageNet WordNet id")
    name: str
    score: float


class ClassifyResponse(BaseModel):
    top: list[ClassPrediction]


class SearchHit(BaseModel):
    id: str
    similarity: float


class VisionSearchResponse(BaseModel):
    results: list[SearchHit]


class SimilarityResponse(BaseModel):
    similarity: float
    model: TextModel


class Cluster(BaseModel):
    rep: str
    indices: list[int]


class DuplicateResponse(BaseModel):
    clusters: list[Cluster]
    model: TextModel | None = Field(description="null when no items were submitted")


class TagScore(BaseModel):
    tag: str
    score: float


class TagSuggestResponse(BaseModel):
    tags: list[TagScore]
    model: TextModel | None = Field(description="null when no tags were submitted")


class ModelStatus(BaseModel):
    classifier_loaded: bool
    image_embedder_loaded: bool
    text_backend: Literal["use", "fallback", "not_loaded"]
    text_error: str | None


class HealthResponse(BaseModel):
    ok: bool
    models: ModelStatus
