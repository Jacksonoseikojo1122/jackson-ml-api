# Jackson ML API

[![CI](https://github.com/Jacksonoseikojo1122/jackson-ml-api/actions/workflows/ci.yml/badge.svg)](https://github.com/Jacksonoseikojo1122/jackson-ml-api/actions/workflows/ci.yml)

A small FastAPI service that puts pretrained TensorFlow models behind a JSON API:

- **Image classification**: MobileNetV2 (ImageNet), top-5 classes.
- **Image similarity search**: MobileNetV2 average-pooled embeddings + cosine similarity.
- **Text similarity, near-duplicate grouping and tag ranking**: Universal Sentence Encoder (TF-Hub), with a deterministic lexical fallback when USE cannot be loaded.

It is designed to sit behind a web app's backend (for example a Next.js API route) and is configured entirely through environment variables.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness and which models are loaded. Public, never triggers a model load. |
| `POST` | `/classify-image` | Top-5 ImageNet classes for one image. |
| `POST` | `/vision-embed-search` | Rank a gallery of images by visual similarity to a query image. |
| `POST` | `/semantic-similarity` | Cosine similarity of two texts. |
| `POST` | `/duplicate-clusters` | Group near-duplicate texts. |
| `POST` | `/tag-suggest` | Rank the candidate tags *you provide* by similarity to a text. It does not invent new tags. |

Interactive OpenAPI docs are served at `/docs` (Swagger UI) and `/redoc`.

Images are sent as base64 strings. A `data:image/<type>;base64,` prefix and line-wrapped base64 are both accepted. Any format Pillow can decode works (JPEG, PNG, WebP, GIF, BMP, ...); images are converted to RGB and resized to 224x224.

Text endpoints return `"model": "use"` or `"model": "fallback"` so callers can tell which embedder produced the scores.

### `GET /health`

```json
{
  "ok": true,
  "models": {
    "classifier_loaded": true,
    "image_embedder_loaded": false,
    "text_backend": "use",
    "text_error": null
  }
}
```

`text_backend` is `"not_loaded"` until the first text request (or startup, with `PRELOAD_MODELS=1`), then `"use"` or `"fallback"`. When it is `"fallback"`, `text_error` holds the reason USE failed to load.

### `POST /classify-image`

```json
{ "image_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..." }
```

```json
{
  "top": [
    { "label": "n02123045", "name": "tabby", "score": 0.61 },
    { "label": "n02123159", "name": "tiger_cat", "score": 0.22 },
    { "label": "n02124075", "name": "Egyptian_cat", "score": 0.09 },
    { "label": "n02127052", "name": "lynx", "score": 0.01 },
    { "label": "n04265275", "name": "space_heater", "score": 0.004 }
  ]
}
```

`label` is the ImageNet WordNet id; `name` is the human-readable class.

### `POST /vision-embed-search`

```json
{
  "query_b64": "iVBORw0KGgo...",
  "gallery": [
    { "id": "sku-101", "b64": "iVBORw0KGgo..." },
    { "id": "sku-102", "b64": "iVBORw0KGgo..." }
  ],
  "top_k": 3
}
```

```json
{
  "results": [
    { "id": "sku-102", "similarity": 0.83 },
    { "id": "sku-101", "similarity": 0.41 }
  ]
}
```

The query and all gallery images are embedded in a single batched forward pass. `top_k` defaults to 3.

### `POST /semantic-similarity`

```json
{ "a": "How do I reset my password?", "b": "I forgot my login password" }
```

```json
{ "similarity": 0.74, "model": "use" }
```

### `POST /duplicate-clusters`

```json
{
  "items": ["Reset my password", "reset my password!", "Shipping to Kumasi"],
  "threshold": 0.82
}
```

```json
{
  "clusters": [
    { "rep": "Reset my password", "indices": [0, 1] },
    { "rep": "Shipping to Kumasi", "indices": [2] }
  ],
  "model": "use"
}
```

Clustering is greedy: items are visited in order, each unassigned item becomes a representative (`rep`), and it absorbs every later unassigned item whose cosine similarity to that representative is at least `threshold` (default 0.82). Membership is not transitive. An empty `items` list returns `{"clusters": [], "model": null}`.

### `POST /tag-suggest`

```json
{
  "text": "Step-by-step guide to deploying a FastAPI app with Docker",
  "tags": ["devops", "python", "cooking", "travel"],
  "top_k": 2
}
```

```json
{
  "tags": [
    { "tag": "devops", "score": 0.38 },
    { "tag": "python", "score": 0.31 }
  ],
  "model": "use"
}
```

Only the supplied tags are scored and returned, sorted by score. `top_k` defaults to 5. An empty `tags` list returns `{"tags": [], "model": null}`.

The scores in these examples are illustrative.

### Errors

| Status | When |
| --- | --- |
| `400` | Image is not valid base64, the bytes are not a decodable image, or it exceeds 40 megapixels. The `detail` names the field, e.g. `gallery[1] (id='sku-102'): image is not valid base64`. |
| `401` | `ML_API_TOKEN` is set and the bearer token is missing or wrong. |
| `413` | A decoded image exceeds 5 MiB, or the request's `Content-Length` exceeds 64 MiB. |
| `422` | Request body fails schema validation (missing fields, limits below). |

## Limits

| Field | Limit |
| --- | --- |
| Decoded image size | 5 MiB per image |
| Image dimensions | 40 megapixels |
| Base64 string length | 7,200,000 characters |
| Request body (`Content-Length`) | 64 MiB |
| `gallery` | 1 to 50 images; `id` 1 to 200 characters |
| `items` | up to 500 strings, each 1 to 2,000 characters |
| `tags` | up to 200 strings, each 1 to 100 characters |
| `a`, `b`, `text` | 1 to 10,000 characters |
| `top_k` | 1 to 50 |
| `threshold` | 0.0 to 1.0 |

Limits are defined in [`app/config.py`](app/config.py). The `Content-Length` check does not cover chunked request bodies; enforce a body-size limit at your reverse proxy as well.

## Configuration

| Variable | Default | Effect |
| --- | --- | --- |
| `ML_API_TOKEN` | unset | When set, every `POST` requires `Authorization: Bearer <token>`. The comparison is constant-time (`secrets.compare_digest`). `/health` and the docs stay public. |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins, e.g. `https://app.example.com,http://localhost:3000`. |
| `PRELOAD_MODELS` | `0` | `1` loads all models at startup instead of on first use. |
| `TFHUB_CACHE_DIR` | `~/.tfhub_cache` | Where the Universal Sentence Encoder is cached. |

See [`.env.example`](.env.example). The app reads the process environment; it does not load `.env` files itself, so export the variables or use `docker run --env-file .env`.

## Quickstart

The pinned runtime stack (TensorFlow 2.20) targets **Python 3.10 to 3.12**; there are no TensorFlow wheels for newer interpreters such as 3.14. It uses the standard `tensorflow` package, which runs on CPU on Linux, macOS and Windows (the separate `tensorflow-cpu` 2.20 wheel fails to import on Linux, which the CI smoke test caught).

### Local virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The first image request downloads the MobileNetV2 weights (roughly 10 to 15 MB each for the classifier and the feature extractor) and the first text request downloads the Universal Sentence Encoder (several hundred MB) from TF-Hub, so expect slow first calls or set `PRELOAD_MODELS=1`.

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/semantic-similarity \
  -H "Content-Type: application/json" \
  -d '{"a": "How do I reset my password?", "b": "I forgot my login password"}'
```

### Docker

```bash
docker build -t jackson-ml-api .
docker run --rm -p 8000:8000 --env-file .env jackson-ml-api
```

The image is based on `python:3.11-slim`, runs as a non-root user and has a `HEALTHCHECK` against `/health`. Mount a volume at `/home/appuser` to keep downloaded weights between container restarts.

## Architecture

```
app/
  main.py     FastAPI app factory, routes, auth dependency, error handling, CORS
  schemas.py  Pydantic request/response models with the limits above
  models.py   ModelBackend protocol + TensorFlowBackend (lazy loading)
  images.py   base64 / data-URL decoding and image validation (Pillow, no TF)
  text.py     fallback embedding, cosine similarity, clustering, ranking (numpy only)
  config.py   limits and environment-driven settings
```

- **Lazy model loading.** TensorFlow is imported only inside `TensorFlowBackend`'s loaders, and each model is built on first use behind a lock. Importing `app.main` therefore costs milliseconds and does not need TensorFlow installed, which is what lets the test suite run without it.
- **Swappable backend.** Routes depend on a `ModelBackend` protocol obtained through a FastAPI dependency (`get_backend`). Tests replace it with a deterministic fake via `app.dependency_overrides`; the same seam would accept a remote model server.
- **Text embeddings with a fallback.** The Universal Sentence Encoder is loaded from TF-Hub on the first text request. If that fails (no network, corrupt cache), the error is recorded, reported by `/health`, and the service falls back to a hashed character-trigram embedding. **The fallback is lexical, not semantic**: it scores shared spelling, so "car" vs "automobile" is near zero. It is good enough for near-duplicate detection of short strings, much weaker for paraphrase or tag ranking. It is deterministic across processes (FNV-1a hashing, not Python's randomised `hash()`). A failed USE load is not retried until the process restarts.
- **Batching.** `/vision-embed-search` decodes every image first (so a bad item fails fast with a 400 that names it) and then runs one `predict` call over the whole batch.
- **Sync handlers.** Inference routes are plain `def` functions, so FastAPI runs them in its threadpool and the event loop stays responsive.

## Testing

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # no TensorFlow
pytest
```

The suite covers authentication (no token, wrong token, wrong scheme, correct token), CORS, every schema limit, invalid base64 and non-image payloads (400), oversized images and bodies (413), data-URL prefix stripping, fallback-embedding determinism (including across processes with different hash seeds), cosine bounds, duplicate clustering, tag ranking order, single-batch image embedding, and that TensorFlow is never imported. CI runs it on Python 3.11 and 3.12 for pushes to `main` and for pull requests.

The unit tests stub the model layer, so CI also runs `scripts/smoke_tf.py` against the real MobileNetV2 weights: it classifies an image through the app and checks that an image ranks as its own nearest neighbour in `/vision-embed-search`. (The Universal Sentence Encoder is skipped there because it is a ~1 GB download.)

```bash
pip install -r requirements.txt httpx
python scripts/smoke_tf.py
```

## Production notes and next steps

This service is a solid single-node baseline. To run it at higher scale, the next steps would be:

- **Model serving.** Move inference to TensorFlow Serving or Triton (behind the existing `ModelBackend` protocol) and use GPU instances for throughput. Export MobileNetV2 to a SavedModel to avoid the Keras download at startup.
- **Embedding cache.** Cache text and image embeddings keyed by a content hash (for example in Redis) so repeated gallery items and tags are not re-embedded on every request.
- **Search at scale.** `/vision-embed-search` compares against a gallery sent with each request, which is fine for tens of images. For catalogue-scale search, precompute embeddings and store them in Postgres with `pgvector` (or another ANN index) and query by vector.
- **Multiple workers.** Each uvicorn worker process loads its own copy of the models; size worker count to available memory.
- **Observability.** Add request metrics and per-model latency, and alert when `/health` reports `text_backend: "fallback"`.

---

Built by Jackson Kojo Osei — [LinkedIn](https://www.linkedin.com/in/jackson-kojo-osei-740846189)
