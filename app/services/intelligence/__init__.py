"""
Intelligence — Software Factory Entry Point.

Exposes the 4 core engines that consolidate 68 legacy files.
All thresholds and rules are config-driven via `intelligence_config.yaml`.
"""

from app.services.intelligence.brain import get_brain
from app.services.intelligence.guardian import get_guardian
from app.services.intelligence.memory import get_memory
from app.services.intelligence.driver import get_driver_registry

__all__ = ["get_brain", "get_guardian", "get_memory", "get_driver_registry"]
