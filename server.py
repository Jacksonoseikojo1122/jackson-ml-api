# server.py
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, pathlib, io, base64
import numpy as np
import tensorflow as tf
import tf_keras as keras            # TF 2.17.x compatible Keras
from PIL import Image

# ---------- Stable TF-Hub cache (prevents corrupt temp downloads on Windows) ----------
os.environ.setdefault("TFHUB_CACHE_DIR", str(pathlib.Path.home() / ".tfhub_cache"))

# Optional auth: set ML_API_TOKEN in your cloud env to require Bearer token on POSTs
API_TOKEN = os.environ.get("ML_API_TOKEN")

def _check_auth(auth_header: str | None):
    if API_TOKEN and auth_header != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------- App ----------
app = FastAPI(title="Jackson ML API")

# If all access goes through your Next.js proxy, you can set this to your Vercel domain(s)
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Vision models (load at import; MobileNetV2 is small) ----------
_cls_model = keras.applications.MobileNetV2(weights="imagenet")
_preproc   = keras.applications.mobilenet_v2.preprocess_input
_decode    = keras.applications.mobilenet_v2.decode_predictions
_emb_model = keras.applications.MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# ---------- Lazy text embeddings (avoid crashing if TF-Hub download hiccups) ----------
_USE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
_use = None
_use_err: str | None = None

def get_use():
    """Load USE once on first call. Record any error instead of crashing."""
    global _use, _use_err
    if _use is not None or _use_err is not None:
        return _use
    try:
        import tensorflow_hub as hub
        _use = hub.load(_USE_URL)
    except Exception as e:
        _use_err = f"{type(e).__name__}: {e}"
        _use = None
    return _use

# Cheap, deterministic fallback so endpoints still respond if USE fails
def _fallback_embed(texts: list[str], dim: int = 512) -> np.ndarray:
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        h = 2166136261
        for ch in t:
            h ^= ord(ch); h = (h * 16777619) & 0xFFFFFFFF
            vecs[i][h % dim] += 1.0
        n = np.linalg.norm(vecs[i]) + 1e-8
        vecs[i] /= n
    return vecs

def _embed_texts(texts: list[str]) -> np.ndarray:
    m = get_use()
    if m is not None:
        return m(texts).numpy()
    return _fallback_embed(texts)

# ---------- Helpers ----------
def _img_from_b64(b64: str, size=224):
    arr = base64.b64decode(b64)
    img = Image.open(io.BytesIO(arr)).convert("RGB").resize((size, size))
    x = np.array(img, dtype=np.float32)[None, ...]
    return _preproc(x)

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.clip(np.dot(a, b), -1.0, 1.0))

# ---------- Schemas ----------
class B64Image(BaseModel):
    image_b64: str  # base64 payload without "data:*;base64,"

class VisionSearchIn(BaseModel):
    query_b64: str
    gallery: list[dict]  # [{id: "...", b64: "..."}]
    top_k: int = 3

class SimilarityIn(BaseModel):
    a: str
    b: str

class DuplicateIn(BaseModel):
    items: list[str]
    threshold: float = 0.82

class TagSuggestIn(BaseModel):
    text: str
    tags: list[str]
    top_k: int = 5

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "use_ready": _use is not None, "use_error": _use_err}

@app.post("/classify-image")
def classify_image(payload: B64Image, authorization: str | None = Header(None)):
    _check_auth(authorization)
    x = _img_from_b64(payload.image_b64, 224)
    preds = _cls_model.predict(x, verbose=0)
    top5 = _decode(preds, top=5)[0]
    return {
        "top": [
            {"label": cls_id, "name": name, "score": float(score)}
            for (cls_id, name, score) in top5
        ]
    }

@app.post("/vision-embed-search")
def vision_search(payload: VisionSearchIn, authorization: str | None = Header(None)):
    _check_auth(authorization)
    q = _img_from_b64(payload.query_b64, 224)
    qv = _emb_model.predict(q, verbose=0)[0]
    results = []
    for g in payload.gallery:
        x = _img_from_b64(g["b64"], 224)
        gv = _emb_model.predict(x, verbose=0)[0]
        results.append({"id": g["id"], "similarity": _cos(qv, gv)})
    results.sort(key=lambda r: r["similarity"], reverse=True)
    return {"results": results[: max(1, payload.top_k)]}

@app.post("/semantic-similarity")
def semantic_similarity(payload: SimilarityIn, authorization: str | None = Header(None)):
    _check_auth(authorization)
    emb = _embed_texts([payload.a, payload.b])
    return {"similarity": _cos(emb[0], emb[1])}

@app.post("/duplicate-clusters")
def duplicate_clusters(payload: DuplicateIn, authorization: str | None = Header(None)):
    _check_auth(authorization)
    if not payload.items:
        return {"clusters": []}
    emb = _embed_texts(payload.items)
    n = emb.shape[0]
    used = [False] * n
    clusters = []
    for i in range(n):
        if used[i]: continue
        clu = [i]; used[i] = True
        for j in range(i + 1, n):
            if not used[j] and _cos(emb[i], emb[j]) >= payload.threshold:
                used[j] = True; clu.append(j)
        clusters.append({"rep": payload.items[i], "indices": clu})
    return {"clusters": clusters}

@app.post("/tag-suggest")
def tag_suggest(payload: TagSuggestIn, authorization: str | None = Header(None)):
    _check_auth(authorization)
    if not payload.tags:
        return {"tags": []}
    emb = _embed_texts([payload.text] + payload.tags)
    q = emb[0]
    scores = [{"tag": tag, "score": _cos(q, emb[i])} for i, tag in enumerate(payload.tags, 1)]
    scores.sort(key=lambda x: x["score"], reverse=True)
    return {"tags": scores[: max(1, payload.top_k)]}
