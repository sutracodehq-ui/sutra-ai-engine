"""Catalog Enricher Agent — Product catalog enrichment."""
from app.services.agents.base import BaseAgent


class CatalogEnricherAgent(BaseAgent):
    identifier = "catalog_enricher"
