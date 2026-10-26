import os
import hashlib
import numpy as np
from typing import List, Optional
from backend.app.config import settings
from backend.app.core.logging import logger

_model_cache = {}

def get_embedder(model_name: Optional[str] = None):
    name = model_name or settings.EMBEDDING_MODEL_DEFAULT
    if name not in _model_cache:
        if os.environ.get("SCHOLARRAG_OFFLINE_EMBEDDINGS", "0") == "1":
            _model_cache[name] = None
            return None
            
        try:
            logger.info("Loading embedding model '%s' on CPU...", name)
            from sentence_transformers import SentenceTransformer
            # Try loading directly from local HF cache first for zero network delay
            try:
                _model_cache[name] = SentenceTransformer(name, device="cpu", local_files_only=True)
            except Exception:
                _model_cache[name] = SentenceTransformer(name, device="cpu", local_files_only=False)
            logger.info("Embedding model '%s' loaded successfully.", name)
        except Exception as e:
            logger.warning("SentenceTransformer '%s' not cached locally or offline: %s. Using local vector encoder.", name, str(e))
            _model_cache[name] = None
    return _model_cache.get(name)

def warmup_embedder(model_name: Optional[str] = None):
    """Pre-warm the embedder in memory so the first user query doesn't experience a cold-start delay."""
    try:
        emb = get_embedder(model_name)
        if emb is not None:
            emb.encode(["warmup"], show_progress_bar=False)
            logger.info("Embedder '%s' pre-warmed.", model_name or settings.EMBEDDING_MODEL_DEFAULT)
    except Exception as e:
        logger.warning("Embedder warmup error: %s", e)

def _fallback_embed(text: str, dim: int = 384) -> List[float]:
    """Fast, deterministic semantic-hash vector representation (384-dim) for offline environments."""
    vec = np.zeros(dim, dtype=np.float32)
    words = text.lower().split()
    for w in words:
        h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        val = ((h >> 8) % 200 - 100) / 100.0
        vec[idx] += val
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()

def embed_texts(texts: List[str], model_name: Optional[str] = None, batch_size: int = 32) -> List[List[float]]:
    """Generate dense vector embeddings for a list of texts."""
    if not texts:
        return []
        
    embedder = get_embedder(model_name)
    if embedder is not None:
        try:
            embeddings = embedder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.warning("Embedding inference failed, falling back to local vector encoder: %s", str(e))
            
    return [_fallback_embed(t) for t in texts]
