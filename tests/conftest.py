"""Shared fixtures. The real TensorFlow backend is replaced by ``FakeBackend``."""

from __future__ import annotations

import base64
import io
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.models import Prediction, TextEmbeddings, get_backend
from app.text import fallback_embed

TOKEN = "s3cret-token"


class FakeBackend:
    """Deterministic stand-in for TensorFlowBackend.

    Image embeddings are the mean RGB colour, so same-hue images are similar.
    Text embeddings use the real lexical fallback.
    """

    def __init__(self) -> None:
        self.embed_image_calls: list[int] = []
        self.classify_calls: list[tuple[int, ...]] = []

    def classify(self, images: np.ndarray) -> list[list[Prediction]]:
        self.classify_calls.append(images.shape)
        return [[Prediction("n02123045", "tabby", 0.9), Prediction("n02124075", "Egyptian_cat", 0.05)]] * len(
            images
        )

    def embed_images(self, images: np.ndarray) -> np.ndarray:
        self.embed_image_calls.append(len(images))
        return images.mean(axis=(1, 2))

    def embed_texts(self, texts: Sequence[str]) -> TextEmbeddings:
        return TextEmbeddings(fallback_embed(texts), "fallback")

    def preload(self) -> None:
        pass

    def status(self) -> dict[str, Any]:
        return {
            "classifier_loaded": False,
            "image_embedder_loaded": False,
            "text_backend": "fallback",
            "text_error": "OSError: offline",
        }


def png_b64(color: tuple[int, int, int] = (255, 0, 0), size: tuple[int, int] = (8, 8)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def make_client(backend: FakeBackend) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def _make(**settings: Any) -> TestClient:
        app = create_app(Settings(**settings))
        app.dependency_overrides[get_backend] = lambda: backend
        client = TestClient(app)
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.close()


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()
