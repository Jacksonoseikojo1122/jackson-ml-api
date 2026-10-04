"""Pure numpy helpers: fallback text embedding, cosine similarity, clustering, ranking.

Nothing in this module imports TensorFlow, so it is cheap to import and test.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

FALLBACK_DIM = 512
_EPS = 1e-8

_FNV_OFFSET = 2166136261
_FNV_PRIME = 16777619


def _fnv1a_32(text: str) -> int:
    """32-bit FNV-1a over UTF-8 bytes. Stable across processes (unlike ``hash()``)."""
    h = _FNV_OFFSET
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * _FNV_PRIME) & 0xFFFFFFFF
    return h


def fallback_embed(texts: Sequence[str], dim: int = FALLBACK_DIM, ngram: int = 3) -> np.ndarray:
    """Hashed character n-gram embedding, L2-normalised.

    This is a *lexical* representation: it captures shared spelling, not meaning.
    "car" and "automobile" score near zero. It exists so the text endpoints keep
    working (deterministically) when the Universal Sentence Encoder cannot be loaded.
    """
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        normalised = " " + " ".join(text.lower().split()) + " "
        n_grams = max(1, len(normalised) - ngram + 1)
        for start in range(n_grams):
            vecs[i, _fnv1a_32(normalised[start : start + ngram]) % dim] += 1.0
    return normalize_rows(vecs)


def normalize_rows(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(norms, _EPS)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors, clipped to [-1, 1]. Zero vectors give 0."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = max(float(np.linalg.norm(a)) * float(np.linalg.norm(b)), _EPS)
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of ``a`` (n, d) and ``b`` (m, d) -> (n, m)."""
    return np.clip(normalize_rows(a) @ normalize_rows(b).T, -1.0, 1.0)


def greedy_clusters(embeddings: np.ndarray, threshold: float) -> list[list[int]]:
    """Group near-duplicates.

    Walk items in input order; each unassigned item starts a cluster and absorbs
    every later unassigned item whose similarity *to that representative* is
    >= ``threshold``. Membership is not transitive. O(n^2) in the number of items.
    """
    n = len(embeddings)
    if n == 0:
        return []
    sims = cosine_matrix(embeddings, embeddings)
    assigned = np.zeros(n, dtype=bool)
    clusters: list[list[int]] = []
    for i in range(n):
        if assigned[i]:
            continue
        members = [i] + [j for j in range(i + 1, n) if not assigned[j] and sims[i, j] >= threshold]
        assigned[members] = True
        clusters.append(members)
    return clusters


def rank_by_similarity(query: np.ndarray, candidates: np.ndarray) -> list[tuple[int, float]]:
    """Return ``(candidate_index, similarity)`` sorted by similarity, descending.

    Ties keep input order (stable sort).
    """
    if len(candidates) == 0:
        return []
    scores = cosine_matrix(np.asarray(query)[None, :], candidates)[0]
    order = sorted(range(len(scores)), key=lambda idx: -float(scores[idx]))
    return [(idx, float(scores[idx])) for idx in order]
