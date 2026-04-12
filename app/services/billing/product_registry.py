"""
Product Registry — Manages sector-wise product segmentation.

Each product is a deployable unit with its own set of agents,
pricing, and target audience. One engine, 23 products.

Usage:
    registry = get_product_registry()
    product = registry.get_product("tryambaka_marketing")
    print(product.agents)  # List of agent IDs in this product
    
    # Check if agent belongs to product
    registry.is_agent_in_product("copywriter", "tryambaka_marketing")  # True
    
    # Get products for a tenant's tier
    registry.available_products("starter")  # Products available on starter
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Product:
    """A deployable product containing sector-specific agents."""
    identifier: str
    name: str
    tagline: str
    icon: str
    price_inr: int
    agents: list[str]
    target_audience: str = ""


@dataclass
class Bundle:
    """A bundle of products at a discounted price."""
    identifier: str
    name: str
    tagline: str
    price_inr: int
    products: list[str] = field(default_factory=list)
    includes_all: bool = False


class ProductRegistry:
    """
    Manages products and bundles.
    Loads from product_catalog.yaml.
    """

    def __init__(self, catalog_path: str = ""):
        self._products: dict[str, Product] = {}
        self._bundles: dict[str, Bundle] = {}
        self._agent_to_product: dict[str, list[str]] = {}

        if not catalog_path:
            catalog_path = str(
                Path(__file__).resolve().parents[3] / "product_catalog.yaml"
            )
        self._load(catalog_path)

    def _load(self, path: str):
        """Load product catalog from YAML."""
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"ProductRegistry: failed to load catalog: {e}")
            return

        for pid, pdata in data.get("products", {}).items():
            product = Product(
                identifier=pid,
                name=pdata["name"],
                tagline=pdata.get("tagline", ""),
                icon=pdata.get("icon", "🤖"),
                price_inr=pdata.get("price_inr", 0),
                agents=pdata.get("agents", []),
                target_audience=pdata.get("target_audience", ""),
            )
            self._products[pid] = product

            for agent_id in product.agents:
                self._agent_to_product.setdefault(agent_id, []).append(pid)

        for bid, bdata in data.get("bundles", {}).items():
            bundle = Bundle(
                identifier=bid,
                name=bdata["name"],
                tagline=bdata.get("tagline", ""),
                price_inr=bdata.get("price_inr", 0),
                products=bdata.get("products", []),
                includes_all=bdata.get("includes_all_products", False),
            )
            self._bundles[bid] = bundle

        logger.info(
            f"ProductRegistry: loaded {len(self._products)} products, "
            f"{len(self._bundles)} bundles, "
            f"{sum(len(p.agents) for p in self._products.values())} total agent slots"
        )

    def get_product(self, product_id: str) -> Optional[Product]:
        """Get a product by ID."""
        return self._products.get(product_id)

    def list_products(self) -> list[dict]:
        """List all products (for pricing page)."""
        return [
            {
                "id": p.identifier,
                "name": p.name,
                "tagline": p.tagline,
                "icon": p.icon,
                "price_inr": p.price_inr,
                "agent_count": len(p.agents),
                "target_audience": p.target_audience,
            }
            for p in self._products.values()
        ]

    def list_bundles(self) -> list[dict]:
        """List all bundles."""
        return [
            {
                "id": b.identifier,
                "name": b.name,
                "tagline": b.tagline,
                "price_inr": b.price_inr,
                "includes_all": b.includes_all,
                "products": b.products,
            }
            for b in self._bundles.values()
        ]

    def is_agent_in_product(self, agent_id: str, product_id: str) -> bool:
        """Check if an agent belongs to a product."""
        product = self._products.get(product_id)
        return product is not None and agent_id in product.agents

    def get_products_for_agent(self, agent_id: str) -> list[str]:
        """Which products contain this agent?"""
        return self._agent_to_product.get(agent_id, [])

    def get_agents_for_product(self, product_id: str) -> list[str]:
        """List all agents in a product."""
        product = self._products.get(product_id)
        return product.agents if product else []

    def get_bundle_agents(self, bundle_id: str) -> list[str]:
        """Get all agents in a bundle (across all bundled products)."""
        bundle = self._bundles.get(bundle_id)
        if not bundle:
            return []

        if bundle.includes_all:
            agents = []
            for p in self._products.values():
                agents.extend(p.agents)
            return list(set(agents))

        agents = []
        for pid in bundle.products:
            agents.extend(self.get_agents_for_product(pid))
        return list(set(agents))

    def stats(self) -> dict:
        """Product catalog statistics."""
        all_agents = set()
        for p in self._products.values():
            all_agents.update(p.agents)

        return {
            "total_products": len(self._products),
            "total_bundles": len(self._bundles),
            "total_unique_agents": len(all_agents),
            "products": {
                p.identifier: {"name": p.name, "agents": len(p.agents), "price": p.price_inr}
                for p in self._products.values()
            },
        }


# ─── Singleton ──────────────────────────────────────────────

_registry: ProductRegistry | None = None

def get_product_registry() -> ProductRegistry:
    global _registry
    if _registry is None:
        _registry = ProductRegistry()
    return _registry
