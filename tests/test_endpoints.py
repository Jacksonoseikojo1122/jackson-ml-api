from __future__ import annotations

import sys

import pytest

from .conftest import png_b64

RED, DARK_RED, BLUE, GREEN = (255, 0, 0), (120, 0, 0), (0, 0, 255), (0, 255, 0)


def test_app_import_does_not_load_tensorflow(client):
    client.get("/health")
    client.post("/semantic-similarity", json={"a": "x", "b": "y"})
    assert "tensorflow" not in sys.modules


def test_health_reports_model_status(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "ok": True,
        "models": {
            "classifier_loaded": False,
            "image_embedder_loaded": False,
            "text_backend": "fallback",
            "text_error": "OSError: offline",
        },
    }


def test_real_backend_status_before_any_load():
    from app.models import TensorFlowBackend

    assert TensorFlowBackend().status() == {
        "classifier_loaded": False,
        "image_embedder_loaded": False,
        "text_backend": "not_loaded",
        "text_error": None,
    }
    assert "tensorflow" not in sys.modules


def test_classify_image(client):
    resp = client.post("/classify-image", json={"image_b64": png_b64()})
    assert resp.status_code == 200
    assert resp.json()["top"][0] == {"label": "n02123045", "name": "tabby", "score": 0.9}


def test_vision_search_batches_and_ranks(client, backend):
    body = {
        "query_b64": png_b64(RED),
        "gallery": [
            {"id": "blue", "b64": png_b64(BLUE)},
            {"id": "dark-red", "b64": png_b64(DARK_RED)},
            {"id": "green", "b64": png_b64(GREEN)},
        ],
        "top_k": 2,
    }
    resp = client.post("/vision-embed-search", json=body)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert [r["id"] for r in results] == ["dark-red", "blue"]  # blue/green tie at 0; input order kept
    assert results[0]["similarity"] == pytest.approx(1.0)
    # Query + 3 gallery images in ONE forward pass.
    assert backend.embed_image_calls == [4]


def test_semantic_similarity(client):
    same = client.post("/semantic-similarity", json={"a": "Hello there", "b": "hello  there"}).json()
    assert same["similarity"] == pytest.approx(1.0)
    assert same["model"] == "fallback"
    diff = client.post("/semantic-similarity", json={"a": "Hello there", "b": "quarterly report"}).json()
    assert -1.0 <= diff["similarity"] < same["similarity"]


def test_duplicate_clusters(client):
    items = ["Reset my password", "reset my password!", "Shipping to Kumasi", "shipping to kumasi"]
    resp = client.post("/duplicate-clusters", json={"items": items, "threshold": 0.8})
    assert resp.status_code == 200
    assert resp.json() == {
        "clusters": [
            {"rep": "Reset my password", "indices": [0, 1]},
            {"rep": "Shipping to Kumasi", "indices": [2, 3]},
        ],
        "model": "fallback",
    }


def test_duplicate_clusters_empty(client):
    assert client.post("/duplicate-clusters", json={"items": []}).json() == {"clusters": [], "model": None}


def test_tag_suggest_ranks_candidates(client):
    body = {"text": "python programming tutorial", "tags": ["cooking", "python", "gardening", "programming"]}
    resp = client.post("/tag-suggest", json={**body, "top_k": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "fallback"
    assert {t["tag"] for t in data["tags"]} == {"python", "programming"}
    scores = [t["score"] for t in data["tags"]]
    assert scores == sorted(scores, reverse=True)

    full = client.post("/tag-suggest", json=body).json()["tags"]
    assert len(full) == 4  # top_k defaults to 5, capped by the number of candidates
    assert {t["tag"] for t in full} == set(body["tags"])  # only provided tags are ever returned


def test_tag_suggest_no_tags(client):
    assert client.post("/tag-suggest", json={"text": "x", "tags": []}).json() == {"tags": [], "model": None}


def test_openapi_schema_renders(client):
    spec = client.get("/openapi.json").json()
    assert "/vision-embed-search" in spec["paths"]
    assert "GalleryImage" in spec["components"]["schemas"]
