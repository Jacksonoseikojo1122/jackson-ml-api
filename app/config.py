"""Runtime settings and request limits."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Request limits. These are enforced by pydantic schemas (422) or by the image
# decoder / request-size middleware (400 / 413).
# ---------------------------------------------------------------------------
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # decoded image size
MAX_IMAGE_PIXELS = 40_000_000  # width * height, guards against decompression bombs
# Base64 inflates by 4/3; leave headroom for a data-URL prefix and line breaks.
# The precise check happens on the decoded bytes (MAX_IMAGE_BYTES).
MAX_B64_CHARS = 7_200_000
MAX_REQUEST_BYTES = 64 * 1024 * 1024  # declared Content-Length of the whole request

MAX_GALLERY = 50
MAX_ITEMS = 500
MAX_TAGS = 200
MAX_TOP_K = 50
MAX_TEXT_CHARS = 10_000  # /semantic-similarity a, b and /tag-suggest text
MAX_ITEM_CHARS = 2_000  # each /duplicate-clusters item
MAX_TAG_CHARS = 100  # each /tag-suggest candidate tag
MAX_ID_CHARS = 200  # each gallery image id

IMAGE_SIZE = 224  # MobileNetV2 input resolution


def _parse_origins(raw: str | None) -> list[str]:
    if raw is None or raw.strip() in ("", "*"):
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def _truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    api_token: str | None = None
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])
    preload_models: bool = False
    max_request_bytes: int = MAX_REQUEST_BYTES

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            api_token=os.environ.get("ML_API_TOKEN") or None,
            allowed_origins=_parse_origins(os.environ.get("ALLOWED_ORIGINS")),
            preload_models=_truthy(os.environ.get("PRELOAD_MODELS")),
        )
