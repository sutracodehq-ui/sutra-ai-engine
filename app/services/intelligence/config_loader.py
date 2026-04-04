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


def load_clusters() -> dict:
    """Load agent-to-cluster mapping from config/model_clusters.yaml."""
    path = Path("config/model_clusters.yaml")
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}
