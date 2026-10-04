"""Pure-function tests for app.text."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.text import FALLBACK_DIM, cosine, cosine_matrix, fallback_embed, greedy_clusters, rank_by_similarity

ROOT = Path(__file__).resolve().parents[1]


def test_fallback_embedding_shape_and_unit_norm():
    emb = fallback_embed(["hello world", "x", "   "])
    assert emb.shape == (3, FALLBACK_DIM)
    assert emb.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, rtol=1e-5)


def test_fallback_embedding_is_deterministic_within_process():
    texts = ["The quick brown fox", "Accra, Ghana", "naive cafe"]
    np.testing.assert_array_equal(fallback_embed(texts), fallback_embed(texts))


def test_fallback_embedding_is_deterministic_across_processes():
    """Must not depend on Python's per-process hash randomisation."""
    code = "from app.text import fallback_embed; print(fallback_embed(['Accra, Ghana'])[0].nonzero()[0].tolist())"
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2")
    }
    assert len(outputs) == 1


def test_fallback_normalises_case_and_whitespace():
    a, b = fallback_embed(["Hello   World", "hello world"])
    assert cosine(a, b) == pytest.approx(1.0)


def test_fallback_is_lexical():
    near, far = fallback_embed(["invoice number 1234", "invoice number 1235"]), fallback_embed(
        ["invoice number 1234", "quarterly weather report"]
    )
    assert cosine(*near) > 0.8
    assert cosine(*far) < 0.3


def test_cosine_bounds_and_special_cases():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = rng.normal(size=(2, 16))
        assert -1.0 <= cosine(a, b) <= 1.0
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, v) == pytest.approx(1.0)
    assert cosine(v, -v) == pytest.approx(-1.0)
    assert cosine(v, np.zeros(3)) == 0.0


def test_cosine_matrix_matches_pairwise():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(4, 8)), rng.normal(size=(3, 8))
    m = cosine_matrix(a, b)
    assert m.shape == (4, 3)
    for i in range(4):
        for j in range(3):
            assert m[i, j] == pytest.approx(cosine(a[i], b[j]), abs=1e-5)


def test_greedy_clusters():
    emb = np.array(
        [
            [1.0, 0.0],  # 0: rep of cluster A
            [0.0, 1.0],  # 1: rep of cluster B
            [0.99, 0.05],  # 2: joins A
            [0.05, 0.99],  # 3: joins B
            [0.7, 0.7],  # 4: ~0.71 to both reps -> own cluster at threshold 0.9
        ]
    )
    assert greedy_clusters(emb, 0.9) == [[0, 2], [1, 3], [4]]
    assert greedy_clusters(emb, 0.0) == [[0, 1, 2, 3, 4]]
    assert greedy_clusters(np.zeros((0, 2)), 0.5) == []


def test_greedy_clusters_is_not_transitive():
    # 0~1 and 1~2 are similar, 0~2 is not: 2 is not pulled into 0's cluster.
    angle = np.deg2rad([0, 30, 60])
    emb = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    assert greedy_clusters(emb, 0.8) == [[0, 1], [2]]


def test_rank_by_similarity_orders_descending_and_is_stable():
    query = np.array([1.0, 0.0])
    candidates = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    ranked = rank_by_similarity(query, candidates)
    assert [idx for idx, _ in ranked] == [1, 3, 2, 0]  # 1 and 3 tie at 1.0; input order kept
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    assert rank_by_similarity(query, np.zeros((0, 2))) == []
