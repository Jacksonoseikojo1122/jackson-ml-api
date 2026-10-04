"""FastAPI application. Run with: ``uvicorn app.main:app``."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import schemas
from .config import Settings
from .images import ImageDecodeError, decode_image
from .models import ModelBackend, get_backend
from .text import cosine, greedy_clusters, rank_by_similarity

_bearer = HTTPBearer(auto_error=False, description="Required only when ML_API_TOKEN is set.")

Backend = Annotated[ModelBackend, Depends(get_backend)]


def require_token(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """Enforce ``Authorization: Bearer <ML_API_TOKEN>`` when a token is configured."""
    expected = request.app.state.settings.api_token
    if not expected:
        return
    supplied = credentials.credentials if credentials else ""
    if not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


router = APIRouter(dependencies=[Depends(require_token)])


@router.post("/classify-image", response_model=schemas.ClassifyResponse)
def classify_image(payload: schemas.ClassifyRequest, backend: Backend) -> schemas.ClassifyResponse:
    """Top-5 ImageNet classes for one image (MobileNetV2)."""
    try:
        image = decode_image(payload.image_b64)
    except ImageDecodeError as exc:
        raise exc.with_context("image_b64") from None
    top = backend.classify(image[None, ...])[0]
    return schemas.ClassifyResponse(
        top=[schemas.ClassPrediction(label=p.label, name=p.name, score=p.score) for p in top]
    )


@router.post("/vision-embed-search", response_model=schemas.VisionSearchResponse)
def vision_embed_search(payload: schemas.VisionSearchRequest, backend: Backend) -> schemas.VisionSearchResponse:
    """Rank gallery images by cosine similarity of MobileNetV2 embeddings to the query image."""
    try:
        images = [decode_image(payload.query_b64)]
    except ImageDecodeError as exc:
        raise exc.with_context("query_b64") from None
    for i, item in enumerate(payload.gallery):
        try:
            images.append(decode_image(item.b64))
        except ImageDecodeError as exc:
            raise exc.with_context(f"gallery[{i}] (id={item.id!r})") from None

    vectors = backend.embed_images(np.stack(images))  # one batched forward pass
    ranked = rank_by_similarity(vectors[0], vectors[1:])
    return schemas.VisionSearchResponse(
        results=[
            schemas.SearchHit(id=payload.gallery[idx].id, similarity=score)
            for idx, score in ranked[: payload.top_k]
        ]
    )


@router.post("/semantic-similarity", response_model=schemas.SimilarityResponse)
def semantic_similarity(payload: schemas.SimilarityRequest, backend: Backend) -> schemas.SimilarityResponse:
    """Cosine similarity of two texts."""
    emb = backend.embed_texts([payload.a, payload.b])
    return schemas.SimilarityResponse(similarity=cosine(emb.vectors[0], emb.vectors[1]), model=emb.model)


@router.post("/duplicate-clusters", response_model=schemas.DuplicateResponse)
def duplicate_clusters(payload: schemas.DuplicateRequest, backend: Backend) -> schemas.DuplicateResponse:
    """Greedy near-duplicate grouping of short texts."""
    if not payload.items:
        return schemas.DuplicateResponse(clusters=[], model=None)
    emb = backend.embed_texts(payload.items)
    clusters = greedy_clusters(emb.vectors, payload.threshold)
    return schemas.DuplicateResponse(
        clusters=[schemas.Cluster(rep=payload.items[c[0]], indices=c) for c in clusters],
        model=emb.model,
    )


@router.post("/tag-suggest", response_model=schemas.TagSuggestResponse)
def tag_suggest(payload: schemas.TagSuggestRequest, backend: Backend) -> schemas.TagSuggestResponse:
    """Rank the *provided* candidate tags by similarity to the text; return the top_k.

    This does not generate new tags.
    """
    if not payload.tags:
        return schemas.TagSuggestResponse(tags=[], model=None)
    emb = backend.embed_texts([payload.text, *payload.tags])
    ranked = rank_by_similarity(emb.vectors[0], emb.vectors[1:])
    return schemas.TagSuggestResponse(
        tags=[schemas.TagScore(tag=payload.tags[idx], score=score) for idx, score in ranked[: payload.top_k]],
        model=emb.model,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.preload_models:
            backend = app.dependency_overrides.get(get_backend, get_backend)()
            backend.preload()
        yield

    app = FastAPI(
        title="Jackson ML API",
        version="1.0.0",
        description="Image classification, image similarity search and text similarity over HTTP.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):  # type: ignore[no-untyped-def]
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_request_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"request body exceeds {settings.max_request_bytes} bytes"},
            )
        return await call_next(request)

    # Added last so it is the outermost middleware: even 413s carry CORS headers.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.exception_handler(ImageDecodeError)
    async def image_error_handler(_: Request, exc: ImageDecodeError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.get("/health", response_model=schemas.HealthResponse)
    def health(backend: Backend) -> schemas.HealthResponse:
        """Liveness plus which models are loaded. Never triggers a model load."""
        return schemas.HealthResponse(ok=True, models=schemas.ModelStatus(**backend.status()))

    app.include_router(router)
    return app


app = create_app()
