"""
Qdrant client singleton + Ollama embeddings.

This engine uses Qdrant as its only vector database (RAG, agent memory, brand
collections). There is no ChromaDB client path in this codebase.

All engine collections use cosine distance and the same embedding model/dimension
(see Settings.embedding_model / embedding_vector_size).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import get_settings

logger = logging.getLogger(__name__)

_qdrant: QdrantClient | None = None


def stable_point_id(id_str: str) -> int:
    """Deterministic unsigned-ish int for Qdrant point id (legacy migrations used string ids)."""
    h = hashlib.sha256(id_str.encode("utf-8")).digest()[:8]
    return int.from_bytes(h, "big", signed=False) % (2**63 - 1)


def get_qdrant_client() -> QdrantClient | None:
    """Lazy singleton; returns None if misconfigured or client init fails."""
    global _qdrant
    if _qdrant is not None:
        return _qdrant
    s = get_settings()
    base = (s.qdrant_url or "").strip()
    if not base:
        return None
    try:
        kwargs: dict[str, Any] = {"url": base.rstrip("/")}
        if getattr(s, "qdrant_api_key", None):
            kwargs["api_key"] = s.qdrant_api_key
        _qdrant = QdrantClient(**kwargs)
        return _qdrant
    except Exception as e:
        logger.warning("Qdrant: client unavailable: %s", e)
        return None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed each non-empty string via Ollama /api/embeddings.
    Returns one vector per input (same order); skips failed entries as empty list
    only if all fail — on partial failure raises from last error.
    """
    s = get_settings()
    base = (s.ollama_base_url or "").rstrip("/")
    if not base or not texts:
        return []
    model = s.embedding_model or "nomic-embed-text"
    out: list[list[float]] = []
    with httpx.Client(timeout=120.0) as http:
        for text in texts:
            if not (text or "").strip():
                out.append([])
                continue
            payload = {"model": model, "prompt": text}
            r = http.post(f"{base}/api/embeddings", json=payload)
            if r.status_code != 200:
                r = http.post(f"{base}/api/embeddings", json={"model": model, "input": text})
            r.raise_for_status()
            data = r.json()
            vec = data.get("embedding")
            if not vec and isinstance(data.get("embeddings"), list) and data["embeddings"]:
                vec = data["embeddings"][0]
            if not vec:
                raise RuntimeError(f"Ollama embeddings returned no vector for model={model!r}")
            out.append(vec)
    return out


def _vector_size_expected() -> int:
    return int(get_settings().embedding_vector_size or 768)


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int | None = None) -> None:
    """Create collection with cosine distance if it does not exist."""
    size = vector_size or _vector_size_expected()
    names = {c.name for c in client.get_collections().collections}
    if collection_name in names:
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=size, distance=Distance.COSINE),
    )


def upsert_points(
    client: QdrantClient,
    collection_name: str,
    ids: list[int],
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
    vector_size: int | None = None,
) -> None:
    if not ids:
        return
    ensure_collection(client, collection_name, vector_size)
    points = [
        PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
        for i in range(len(ids))
    ]
    client.upsert(collection_name=collection_name, points=points, wait=True)


def search_points(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    limit: int,
    vector_size: int | None = None,
) -> list[dict[str, Any]]:
    """
    Returns list of {id, score, payload} where score is cosine similarity (higher = closer).
    """
    names = {c.name for c in client.get_collections().collections}
    if collection_name not in names:
        return []
    res = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit,
        with_payload=True,
    )
    rows: list[dict[str, Any]] = []
    for hit in res:
        rows.append({"id": hit.id, "score": float(hit.score), "payload": hit.payload or {}})
    return rows


def qdrant_collection_count(client: QdrantClient, collection_name: str) -> int:
    names = {c.name for c in client.get_collections().collections}
    if collection_name not in names:
        return 0
    try:
        return int(client.count(collection_name=collection_name, exact=True).count)
    except Exception:
        info = client.get_collection(collection_name=collection_name)
        return int(getattr(info, "points_count", 0) or 0)
