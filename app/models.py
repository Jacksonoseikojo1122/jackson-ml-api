"""Model layer.

``TensorFlowBackend`` loads every model lazily, on first use, behind a lock.
TensorFlow itself is only imported inside those loaders, so importing this
module (and therefore ``app.main``) is fast and does not require TensorFlow.
Tests replace the backend through FastAPI's dependency overrides.
"""

from __future__ import annotations

import logging
import os
import pathlib
import threading
from collections.abc import Sequence
from typing import Any, NamedTuple, Protocol

import numpy as np

from .text import fallback_embed

logger = logging.getLogger(__name__)

USE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
TOP_K_CLASSES = 5


class Prediction(NamedTuple):
    label: str  # ImageNet WordNet id, e.g. "n02123045"
    name: str  # human-readable class, e.g. "tabby"
    score: float


class TextEmbeddings(NamedTuple):
    vectors: np.ndarray  # (n, d)
    model: str  # "use" or "fallback"


class ModelBackend(Protocol):
    def classify(self, images: np.ndarray) -> list[list[Prediction]]:
        """images: float32 (n, 224, 224, 3) RGB in 0-255. Returns top-5 per image."""

    def embed_images(self, images: np.ndarray) -> np.ndarray:
        """images: float32 (n, 224, 224, 3) RGB in 0-255. Returns (n, d) embeddings."""

    def embed_texts(self, texts: Sequence[str]) -> TextEmbeddings: ...

    def status(self) -> dict[str, Any]: ...

    def preload(self) -> None: ...


class TensorFlowBackend:
    """MobileNetV2 (ImageNet) for vision, Universal Sentence Encoder for text."""

    def __init__(self, use_url: str = USE_URL) -> None:
        self._use_url = use_url
        self._lock = threading.Lock()
        self._classifier: Any = None
        self._embedder: Any = None
        self._use: Any = None
        self._use_attempted = False
        self._use_error: str | None = None

    # -- loaders -----------------------------------------------------------
    @staticmethod
    def _mobilenet() -> Any:
        from tensorflow import keras  # noqa: PLC0415 - deliberately lazy

        return keras.applications.mobilenet_v2

    def _get_classifier(self) -> Any:
        if self._classifier is None:
            with self._lock:
                if self._classifier is None:
                    logger.info("Loading MobileNetV2 classifier")
                    self._classifier = self._mobilenet().MobileNetV2(weights="imagenet")
        return self._classifier

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    logger.info("Loading MobileNetV2 feature extractor")
                    self._embedder = self._mobilenet().MobileNetV2(
                        weights="imagenet", include_top=False, pooling="avg"
                    )
        return self._embedder

    def _get_use(self) -> Any:
        """Load USE once. A failure is recorded and the fallback is used until restart."""
        if not self._use_attempted:
            with self._lock:
                if not self._use_attempted:
                    # Stable cache dir avoids half-written downloads in OS temp dirs.
                    os.environ.setdefault("TFHUB_CACHE_DIR", str(pathlib.Path.home() / ".tfhub_cache"))
                    try:
                        import tensorflow_hub as hub  # noqa: PLC0415 - deliberately lazy

                        logger.info("Loading Universal Sentence Encoder from %s", self._use_url)
                        self._use = hub.load(self._use_url)
                    except Exception as exc:  # noqa: BLE001 - any failure means "use fallback"
                        self._use_error = f"{type(exc).__name__}: {exc}"
                        logger.warning("USE unavailable, using lexical fallback: %s", self._use_error)
                    self._use_attempted = True
        return self._use

    # -- inference -----------------------------------------------------------
    def classify(self, images: np.ndarray) -> list[list[Prediction]]:
        mobilenet = self._mobilenet()
        model = self._get_classifier()
        preds = model.predict(mobilenet.preprocess_input(images), verbose=0)
        decoded = mobilenet.decode_predictions(preds, top=TOP_K_CLASSES)
        return [[Prediction(str(label), str(name), float(score)) for label, name, score in row] for row in decoded]

    def embed_images(self, images: np.ndarray) -> np.ndarray:
        model = self._get_embedder()
        x = self._mobilenet().preprocess_input(images)
        return np.asarray(model.predict(x, batch_size=32, verbose=0), dtype=np.float32)

    def embed_texts(self, texts: Sequence[str]) -> TextEmbeddings:
        use = self._get_use()
        if use is not None:
            return TextEmbeddings(np.asarray(use(list(texts)), dtype=np.float32), "use")
        return TextEmbeddings(fallback_embed(texts), "fallback")

    def preload(self) -> None:
        self._get_classifier()
        self._get_embedder()
        self._get_use()

    def status(self) -> dict[str, Any]:
        if not self._use_attempted:
            text_backend = "not_loaded"
        else:
            text_backend = "use" if self._use is not None else "fallback"
        return {
            "classifier_loaded": self._classifier is not None,
            "image_embedder_loaded": self._embedder is not None,
            "text_backend": text_backend,
            "text_error": self._use_error,
        }


_backend: ModelBackend | None = None
_backend_lock = threading.Lock()


def get_backend() -> ModelBackend:
    """FastAPI dependency returning the process-wide backend (constructed lazily)."""
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                _backend = TensorFlowBackend()
    return _backend
