"""
Knowledge Base Service — Manages vector storage and retrieval.

RAG-AI: Interfaces with ChromaDB to store tenant-specific facts 
and retrieve them during the chat pipeline.
"""

import logging
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

from app.config import get_settings

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    """Service to interface with vector database."""

    def __init__(self):
        settings = get_settings()
        # In-memory for now, or persistent if directory provided
        self._client = chromadb.PersistentClient(path="./data/chroma")
        self._embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name="text-embedding-3-small"
        )

    def get_collection(self, tenant_id: int):
        """Get or create a collection for a specific tenant."""
        return self._client.get_or_create_collection(
            name=f"tenant_{tenant_id}_kb",
            embedding_function=self._embedding_fn
        )

    async def add_documents(self, tenant_id: int, chunks: List[str], metadatas: List[dict], ids: List[str]):
        """Index document chunks into the vector store."""
        collection = self.get_collection(tenant_id)
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Indexed {len(chunks)} chunks for tenant {tenant_id}")

    async def query(self, tenant_id: int, query_text: str, n_results: int = 3) -> List[str]:
        """Search for relevant chunks in the knowledge base."""
        try:
            collection = self.get_collection(tenant_id)
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            # Flatten the nested structure from Chroma
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            logger.error(f"KnowledgeBase query failed: {e}")
            return []
