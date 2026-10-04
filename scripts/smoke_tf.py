"""End-to-end smoke test against the real TensorFlow models.

The unit tests stub the model layer; this script does not. It loads MobileNetV2
(ImageNet weights, ~14 MB download) and exercises the vision endpoints through
the real app. The Universal Sentence Encoder is skipped: it is a ~1 GB download.

    pip install -r requirements.txt httpx
    python scripts/smoke_tf.py
"""

from __future__ import annotations

import base64
import io
import os
import sys

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.pop("ML_API_TOKEN", None)

from app.main import app  # noqa: E402


def png_b64(seed: int) -> str:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    a, b = png_b64(1), png_b64(2)
    with TestClient(app) as client:
        r = client.post("/classify-image", json={"image_b64": a})
        assert r.status_code == 200, r.text
        top = r.json()["top"]
        assert len(top) == 5, top
        scores = [p["score"] for p in top]
        assert scores == sorted(scores, reverse=True), scores
        assert all(0.0 <= s <= 1.0 for s in scores) and sum(scores) <= 1.0001, scores
        assert all(p["label"].startswith("n") and p["name"] for p in top), top
        print("classify-image ok:", [(p["name"], round(p["score"], 3)) for p in top])

        r = client.post(
            "/vision-embed-search",
            json={"query_b64": a, "gallery": [{"id": "other", "b64": b}, {"id": "same", "b64": a}], "top_k": 2},
        )
        assert r.status_code == 200, r.text
        results = r.json()["results"]
        assert results[0]["id"] == "same" and results[0]["similarity"] > 0.999, results
        assert results[1]["similarity"] < results[0]["similarity"], results
        print("vision-embed-search ok:", results)

        health = client.get("/health").json()
        models = health["models"]
        assert models["classifier_loaded"] and models["image_embedder_loaded"], health
        print("health ok:", health)

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
