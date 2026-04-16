"""
Shared configuration loader for all intelligence services.

Software Factory Principle: Config-driven, zero hardcoded data.
"""

import logging
import time
from pathlib import Path
from functools import lru_cache

import yaml

logger = logging.getLogger(__name__)

# ─── Config Loader (cached singleton) ─────────────────────────

_config_cache: dict | None = None
_config_loaded_at: float = 0
_CONFIG_TTL: float = 60.0  # reload YAML every 60s


def get_intelligence_config() -> dict:
    """Load the full intelligence_config.yaml. Cached with 60s TTL."""
    global _config_cache, _config_loaded_at

    now = time.monotonic()
    if _config_cache is not None and (now - _config_loaded_at) < _CONFIG_TTL:
        return _config_cache

    config_path = Path("intelligence_config.yaml")
    if not config_path.exists():
        logger.warning("intelligence_config.yaml NOT FOUND. Using empty defaults.")
        _config_cache = {}
        _config_loaded_at = now
        return _config_cache

    try:
        with open(config_path) as f:
            full = yaml.safe_load(f) or {}
        _config_cache = full
        _config_loaded_at = now
        return _config_cache
    except Exception as e:
        logger.error(f"Failed to load intelligence_config.yaml: {e}")
        return _config_cache or {}


def get_provider_config(name: str) -> dict:
    """Get metadata for a specific provider (base_url, etc.) from YAML."""
    config = get_intelligence_config()
    return config.get("providers", {}).get(name, {})


def get_global_driver_chain() -> list[str]:
    """Canonical driver order from resilience.global_driver_chain (single source of truth)."""
    g = (get_intelligence_config().get("resilience") or {}).get("global_driver_chain")
    if isinstance(g, list) and g:
        return [str(x).strip() for x in g if str(x).strip()]
    return []


def get_routing_section() -> dict:
    """routing.* — local vs cloud classification + hybrid local target."""
    return get_intelligence_config().get("routing") or {}


def get_local_driver_ids() -> frozenset[str]:
    """Drivers treated as on-box / local for hybrid + route hints (routing.local_driver_ids)."""
    raw = get_routing_section().get("local_driver_ids")
    if isinstance(raw, list) and raw:
        return frozenset(str(x).strip() for x in raw if str(x).strip())
    return frozenset()


def get_hybrid_local_driver() -> str:
    """Brain hybrid path: which driver answers the local attempt (routing.hybrid_local_driver)."""
    v = get_routing_section().get("hybrid_local_driver")
    if isinstance(v, str) and v.strip():
        return v.strip()
    loc = get_local_driver_ids()
    for d in get_global_driver_chain():
        if d in loc:
            return d
    return ""


def order_chain_by_global_reference(chain: list[str]) -> list[str]:
    """Stable-sort *chain* to follow resilience.global_driver_chain order, then append unknown tails."""
    ref = get_global_driver_chain()
    if not ref or not chain:
        return list(chain)
    in_chain = set(chain)
    head = [d for d in ref if d in in_chain]
    tail = [d for d in chain if d not in head]
    return head + tail


def first_non_local_driver_from_chain() -> str | None:
    """First entry in global_driver_chain not listed in routing.local_driver_ids."""
    loc = get_local_driver_ids()
    for d in get_global_driver_chain():
        if d not in loc:
            return d
    return None


def get_live_knowledge_llm_config() -> dict:
    """live_knowledge.driver / live_knowledge.model — optional; falls back to first non-local + tier model."""
    return get_intelligence_config().get("live_knowledge") or {}


def load_clusters() -> dict:
    """Load agent-to-cluster mapping from config/model_clusters.yaml."""
    path = Path("config/model_clusters.yaml")
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}
