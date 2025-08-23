# Jackson ML API (FastAPI + TensorFlow)

Endpoints:
- `GET /health` — basic health & embedding status
- `POST /classify-image` — base64 image -> ImageNet top-5
- `POST /vision-embed-search` — image embedding similarity (MobileNetV2)
- `POST /semantic-similarity` — USE (w/ fallback) cosine
- `POST /duplicate-clusters` — cluster near-duplicate strings
- `POST /tag-suggest` — suggest top-k tags for a text

## Local run
```bash
python -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
set TFHUB_CACHE_DIR=%USERPROFILE%\.tfhub_cache
uvicorn server:app --host 127.0.0.1 --port 8000
GoodLuck 😒😒😒😒😒😒😒😒😒😒😒😒