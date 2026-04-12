"""
VoIP Personas — Config-driven voice persona manager.

Software Factory Principle: Polymorphic Pattern.

One engine, many personas. Each persona combines:
- Voice: Edge-TTS voice ID (Indian voices for India market)
- Agent: Which AI agent provides the brains
- Tone: Personality instructions injected into agent
- Languages: Which languages this persona can handle
- Greetings: Per-language greeting messages
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_personas_config() -> dict:
    """Load persona configs from YAML."""
    config_path = Path("intelligence_config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    return config.get("voip", {}).get("personas", {})


@dataclass
class VoicePersona:
    """A VoIP persona with voice, agent, and behavior config."""
    id: str
    name: str
    voice_id: str                    # Edge-TTS voice name
    agent_id: str                    # AI agent identifier
    tone: str                        # Personality description
    languages: list[str]             # Supported language codes
    greetings: dict[str, str] = field(default_factory=dict)
    max_turns: int = 20              # Max conversation turns
    transfer_to_human: bool = True   # Allow warm handoff


class PersonaManager:
    """
    Manages VoIP personas loaded from YAML config.

    Example use:
        mgr = get_persona_manager()
        persona = mgr.get_persona("sales_agent_india")
        # persona.voice_id → "hi-IN-MadhurNeural"
        # persona.agent_id → "campaign_strategist"
    """

    def __init__(self):
        self._personas: dict[str, VoicePersona] = {}
        self._load()

    def _load(self):
        """Load all personas from YAML config."""
        raw = _load_personas_config()
        for pid, cfg in raw.items():
            self._personas[pid] = VoicePersona(
                id=pid,
                name=cfg.get("name", pid.replace("_", " ").title()),
                voice_id=cfg.get("voice", "en-IN-NeerjaNeural"),
                agent_id=cfg.get("agent", "chatbot_trainer"),
                tone=cfg.get("tone", "professional and helpful"),
                languages=cfg.get("languages", ["en", "hi"]),
                greetings=cfg.get("greetings", {}),
                max_turns=cfg.get("max_turns", 20),
                transfer_to_human=cfg.get("transfer_to_human", True),
            )
        logger.info(f"VoIP: loaded {len(self._personas)} personas")

    def get_persona(self, persona_id: str) -> Optional[VoicePersona]:
        """Get a persona by ID."""
        return self._personas.get(persona_id)

    def list_personas(self) -> list[dict]:
        """List all personas with their config."""
        return [
            {
                "id": p.id,
                "name": p.name,
                "voice": p.voice_id,
                "agent": p.agent_id,
                "languages": p.languages,
                "tone": p.tone,
            }
            for p in self._personas.values()
        ]

    def get_greeting(self, persona_id: str, language: str = "en") -> str:
        """Get the greeting message for a persona in a specific language."""
        persona = self.get_persona(persona_id)
        if not persona:
            return "Hello!"
        return persona.greetings.get(language, persona.greetings.get("en", "Hello!"))

    def get_persona_for_language(self, language: str) -> Optional[VoicePersona]:
        """Find the first persona that supports a given language."""
        for persona in self._personas.values():
            if language in persona.languages:
                return persona
        return None


# ─── Singleton ──────────────────────────────────────────────
_manager: PersonaManager | None = None


def get_persona_manager() -> PersonaManager:
    global _manager
    if _manager is None:
        _manager = PersonaManager()
    return _manager
