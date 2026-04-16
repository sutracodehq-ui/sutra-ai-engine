"""Vector store helpers (Qdrant + shared embeddings)."""

from app.services.vector.qdrant_store import (
    embed_texts,
    ensure_collection,
    get_qdrant_client,
    qdrant_collection_count,
    search_points,
    stable_point_id,
    upsert_points,
)

__all__ = [
    "embed_texts",
    "ensure_collection",
    "get_qdrant_client",
    "qdrant_collection_count",
    "search_points",
    "stable_point_id",
    "upsert_points",
]
