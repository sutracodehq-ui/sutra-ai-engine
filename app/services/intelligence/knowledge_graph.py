"""
Knowledge Graph Builder — Entity extraction and relationship mapping.

Software Factory Principle: Continuous Improvement.

Extracts entities and relationships from all scanned content
(RSS articles, stock data, AI trends) and builds a traversable
knowledge graph stored in ChromaDB. Agents use this for deeper
contextual insights beyond simple vector search.

Architecture:
    Raw Content → Entity Extraction → Relationship Mapping → ChromaDB Storage
                                                             ↕
                                              Agent Queries traverse the graph
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import yaml
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)


def _load_kg_config() -> dict:
    """Load knowledge graph config from YAML."""
    config_path = Path("intelligence_config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    return config.get("knowledge_graph", {})


class Entity:
    """A node in the knowledge graph."""

    def __init__(self, name: str, entity_type: str, properties: dict | None = None):
        self.name = name
        self.entity_type = entity_type  # company, person, technology, concept
        self.properties = properties or {}
        self.id = f"{entity_type}:{name.lower().replace(' ', '_')}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.entity_type,
            "properties": self.properties,
        }


class Relationship:
    """An edge in the knowledge graph."""

    def __init__(self, source: str, target: str, relation: str, weight: float = 1.0):
        self.source = source
        self.target = target
        self.relation = relation  # "competes_with", "develops", "acquires", etc.
        self.weight = weight

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
        }


class KnowledgeGraphBuilder:
    """
    Builds and maintains a knowledge graph from scanned content.

    The graph supports:
    - Entity extraction (companies, technologies, concepts)
    - Relationship mapping (competes, develops, acquires)
    - Graph traversal (find related entities N hops away)
    - Context injection (enrich agent prompts with graph data)
    """

    COLLECTION_NAME = "knowledge_graph"

    def __init__(self):
        self._client = None
        self._collection = None

    def _get_collection(self):
        """Get or create the knowledge graph ChromaDB collection."""
        if self._collection is None:
            import chromadb
            settings = get_settings()
            self._client = chromadb.HttpClient(
                host=settings.chroma_host, port=int(settings.chroma_port)
            )
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "Knowledge graph entities and relationships"},
            )
        return self._collection

    # ─── Entity Extraction ──────────────────────────────────

    async def extract_entities(self, text: str, source: str = "unknown") -> list[Entity]:
        """
        Use LLM to extract named entities from text.
        Returns structured Entity objects.
        """
        config = _load_kg_config()
        entity_types = config.get("entity_types", [
            "company", "person", "technology", "product", "concept",
            "market_sector", "cryptocurrency", "stock_ticker",
        ])

        from app.services.llm_service import get_llm_service
        llm = get_llm_service()

        prompt = f"""Extract named entities from this text. For each entity, provide:
- name: The entity name
- type: One of {entity_types}
- properties: Any relevant attributes (e.g., ticker, market_cap, founded_year)

Return ONLY valid JSON array. Example:
[{{"name": "NVIDIA", "type": "company", "properties": {{"ticker": "NVDA", "sector": "semiconductors"}}}}]

Text: {text[:2000]}"""

        response = await llm.complete(
            prompt=prompt,
            system_prompt="You extract named entities from text. Return only valid JSON.",
        )

        entities = []
        try:
            import re
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                raw = json.loads(json_match.group())
                for item in raw:
                    entities.append(Entity(
                        name=item.get("name", ""),
                        entity_type=item.get("type", "concept"),
                        properties={
                            **item.get("properties", {}),
                            "source": source,
                            "extracted_at": datetime.now(timezone.utc).isoformat(),
                        },
                    ))
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"KG: entity extraction parse failed: {e}")

        logger.info(f"KG: extracted {len(entities)} entities from {source}")
        return entities

    # ─── Relationship Extraction ────────────────────────────

    async def extract_relationships(
        self, text: str, entities: list[Entity]
    ) -> list[Relationship]:
        """Extract relationships between known entities from text."""
        if len(entities) < 2:
            return []

        entity_names = [e.name for e in entities]
        from app.services.llm_service import get_llm_service
        llm = get_llm_service()

        prompt = f"""Given these entities: {entity_names}

Extract relationships between them from this text.
Relationship types: competes_with, develops, acquires, partners_with,
invests_in, uses, affects, related_to

Return ONLY valid JSON array. Example:
[{{"source": "NVIDIA", "target": "AMD", "relation": "competes_with", "weight": 0.9}}]

Text: {text[:2000]}"""

        response = await llm.complete(
            prompt=prompt,
            system_prompt="You extract entity relationships. Return only valid JSON.",
        )

        relationships = []
        try:
            import re
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                raw = json.loads(json_match.group())
                for item in raw:
                    relationships.append(Relationship(
                        source=item.get("source", ""),
                        target=item.get("target", ""),
                        relation=item.get("relation", "related_to"),
                        weight=float(item.get("weight", 0.5)),
                    ))
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"KG: relationship extraction failed: {e}")

        logger.info(f"KG: extracted {len(relationships)} relationships")
        return relationships

    # ─── Storage ────────────────────────────────────────────

    async def store(self, entities: list[Entity], relationships: list[Relationship]) -> dict:
        """Store entities and relationships in ChromaDB."""
        collection = self._get_collection()
        stored_entities = 0
        stored_rels = 0

        # Store entities
        for entity in entities:
            try:
                doc = json.dumps(entity.to_dict(), ensure_ascii=False)
                collection.upsert(
                    ids=[entity.id],
                    documents=[doc],
                    metadatas=[{
                        "name": entity.name,
                        "type": entity.entity_type,
                        "node_type": "entity",
                    }],
                )
                stored_entities += 1
            except Exception as e:
                logger.debug(f"KG: failed to store entity {entity.name}: {e}")

        # Store relationships as documents
        for rel in relationships:
            try:
                rel_id = f"rel:{rel.source}:{rel.relation}:{rel.target}"
                doc = json.dumps(rel.to_dict(), ensure_ascii=False)
                collection.upsert(
                    ids=[rel_id],
                    documents=[doc],
                    metadatas=[{
                        "source": rel.source,
                        "target": rel.target,
                        "relation": rel.relation,
                        "node_type": "relationship",
                    }],
                )
                stored_rels += 1
            except Exception as e:
                logger.debug(f"KG: failed to store relationship: {e}")

        return {"stored_entities": stored_entities, "stored_relationships": stored_rels}

    # ─── Query ──────────────────────────────────────────────

    async def query_entity(self, name: str) -> dict:
        """Find an entity and its relationships."""
        collection = self._get_collection()

        # Find the entity
        results = collection.query(query_texts=[name], n_results=5)

        entity_data = None
        relationships = []

        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                try:
                    data = json.loads(doc)
                    meta = results["metadatas"][0][i]
                    if meta.get("node_type") == "entity":
                        entity_data = data
                    elif meta.get("node_type") == "relationship":
                        relationships.append(data)
                except json.JSONDecodeError:
                    pass

        return {
            "entity": entity_data,
            "relationships": relationships,
            "total_connections": len(relationships),
        }

    async def get_context_for_agent(self, query: str, max_entities: int = 5) -> str:
        """
        Build a context string from knowledge graph for agent injection.
        """
        collection = self._get_collection()
        results = collection.query(query_texts=[query], n_results=max_entities)

        if not results or not results["documents"]:
            return ""

        context_parts = []
        for i, doc in enumerate(results["documents"][0]):
            try:
                data = json.loads(doc)
                meta = results["metadatas"][0][i]
                if meta.get("node_type") == "entity":
                    context_parts.append(
                        f"[KG] {data['name']} ({data['type']}): {json.dumps(data.get('properties', {}))}"
                    )
                elif meta.get("node_type") == "relationship":
                    context_parts.append(
                        f"[KG] {data['source']} —{data['relation']}→ {data['target']}"
                    )
            except (json.JSONDecodeError, KeyError):
                pass

        return "\n".join(context_parts)

    # ─── Full Pipeline ──────────────────────────────────────

    async def process_text(self, text: str, source: str = "unknown") -> dict:
        """
        Full knowledge graph pipeline:
        Extract entities → Extract relationships → Store
        """
        entities = await self.extract_entities(text, source)
        relationships = await self.extract_relationships(text, entities)
        result = await self.store(entities, relationships)
        result["entities_extracted"] = len(entities)
        result["relationships_extracted"] = len(relationships)
        return result


# ─── Singleton ──────────────────────────────────────────────
_kg: KnowledgeGraphBuilder | None = None


def get_knowledge_graph() -> KnowledgeGraphBuilder:
    global _kg
    if _kg is None:
        _kg = KnowledgeGraphBuilder()
    return _kg
