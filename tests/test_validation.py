from __future__ import annotations

import base64

import pytest

from app import config

from .conftest import png_b64


# ---- image payloads -----------------------------------------------------------
def test_invalid_base64_returns_400(client):
    resp = client.post("/classify-image", json={"image_b64": "not base64 !!!"})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "image_b64: image is not valid base64"


def test_valid_base64_but_not_an_image_returns_400(client):
    resp = client.post("/classify-image", json={"image_b64": base64.b64encode(b"hello world").decode()})
    assert resp.status_code == 400
    assert "not a decodable image" in resp.json()["detail"]


def test_empty_image_returns_422(client):
    assert client.post("/classify-image", json={"image_b64": ""}).status_code == 422


def test_data_url_prefix_is_stripped(client, backend):
    resp = client.post("/classify-image", json={"image_b64": "data:image/png;base64," + png_b64()})
    assert resp.status_code == 200
    assert backend.classify_calls == [(1, config.IMAGE_SIZE, config.IMAGE_SIZE, 3)]


def test_line_wrapped_base64_is_accepted(client):
    b64 = png_b64()
    wrapped = "\n".join(b64[i : i + 76] for i in range(0, len(b64), 76))
    assert client.post("/classify-image", json={"image_b64": wrapped}).status_code == 200


def test_oversized_image_returns_413(client):
    too_big = base64.b64encode(b"\0" * (config.MAX_IMAGE_BYTES + 1)).decode()
    resp = client.post("/classify-image", json={"image_b64": too_big})
    assert resp.status_code == 413


def test_bad_gallery_image_error_names_the_item(client):
    body = {"query_b64": png_b64(), "gallery": [{"id": "ok", "b64": png_b64()}, {"id": "bad", "b64": "%%%"}]}
    resp = client.post("/vision-embed-search", json=body)
    assert resp.status_code == 400
    assert resp.json()["detail"].startswith("gallery[1] (id='bad'):")


def test_request_size_limit_returns_413(make_client):
    client = make_client(max_request_bytes=1_000)
    resp = client.post("/semantic-similarity", json={"a": "x" * 2_000, "b": "y"})
    assert resp.status_code == 413


# ---- schema limits ------------------------------------------------------------
@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/vision-embed-search", {"query_b64": "AAAA", "gallery": []}),
        (
            "/vision-embed-search",
            {"query_b64": "AAAA", "gallery": [{"id": str(i), "b64": "AAAA"} for i in range(config.MAX_GALLERY + 1)]},
        ),
        ("/vision-embed-search", {"query_b64": "AAAA", "gallery": [{"b64": "AAAA"}]}),
        ("/vision-embed-search", {"query_b64": "AAAA", "gallery": [{"id": "a", "b64": "AAAA"}], "top_k": 0}),
        (
            "/vision-embed-search",
            {"query_b64": "AAAA", "gallery": [{"id": "a", "b64": "AAAA"}], "top_k": config.MAX_TOP_K + 1},
        ),
        ("/duplicate-clusters", {"items": ["x"] * (config.MAX_ITEMS + 1)}),
        ("/duplicate-clusters", {"items": ["x"], "threshold": 1.5}),
        ("/duplicate-clusters", {"items": ["x"], "threshold": -0.1}),
        ("/duplicate-clusters", {"items": [""]}),
        ("/duplicate-clusters", {"items": ["x" * (config.MAX_ITEM_CHARS + 1)]}),
        ("/tag-suggest", {"text": "t", "tags": ["x"] * (config.MAX_TAGS + 1)}),
        ("/tag-suggest", {"text": "t", "tags": ["x" * (config.MAX_TAG_CHARS + 1)]}),
        ("/tag-suggest", {"text": "t", "tags": ["x"], "top_k": 0}),
        ("/tag-suggest", {"text": "", "tags": ["x"]}),
        ("/semantic-similarity", {"a": "", "b": "x"}),
        ("/semantic-similarity", {"a": "x" * (config.MAX_TEXT_CHARS + 1), "b": "x"}),
    ],
    ids=[
        "empty-gallery",
        "gallery-too-long",
        "gallery-item-missing-id",
        "vision-top_k-zero",
        "vision-top_k-too-big",
        "too-many-items",
        "threshold-above-1",
        "threshold-below-0",
        "empty-item",
        "item-too-long",
        "too-many-tags",
        "tag-too-long",
        "tag-top_k-zero",
        "empty-tag-text",
        "empty-similarity-text",
        "similarity-text-too-long",
    ],
)
def test_schema_limits_return_422(client, path, body):
    assert client.post(path, json=body).status_code == 422


def test_limits_at_the_boundary_are_accepted(client):
    body = {"text": "t", "tags": [f"tag{i}" for i in range(config.MAX_TAGS)], "top_k": config.MAX_TOP_K}
    resp = client.post("/tag-suggest", json=body)
    assert resp.status_code == 200
    assert len(resp.json()["tags"]) == config.MAX_TOP_K

    resp = client.post("/duplicate-clusters", json={"items": ["a"], "threshold": 1.0})
    assert resp.status_code == 200
