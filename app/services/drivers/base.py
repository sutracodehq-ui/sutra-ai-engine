"""
Abstract LLM Driver — the contract every provider must implement.

Software Factory pattern: every driver is a standardized, interchangeable
production unit. Swap OpenAI → Gemini → Ollama with zero consumer changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator


@dataclass
class LlmResponse:
    """Standardized response from any LLM driver."""

    content: str = ""
    raw_response: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    driver: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "raw_response": self.raw_response,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "driver": self.driver,
        }


class LlmDriver(ABC):
    """
    Abstract base for all LLM drivers.

    Every driver implements 3 methods:
    - complete(): single-turn prompt → response
    - chat(): multi-turn messages → response
    - stream(): multi-turn messages → async chunk generator
    """

    @abstractmethod
    def name(self) -> str:
        """Driver identifier (e.g., 'openai', 'gemini')."""
        ...

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str, **options) -> LlmResponse:
        """Single-turn completion: system + user → response."""
        ...

    @abstractmethod
    async def chat(self, messages: list[dict], **options) -> LlmResponse:
        """Multi-turn chat completion: [{role, content}, ...] → response."""
        ...

    @abstractmethod
    async def stream(self, messages: list[dict], **options) -> AsyncGenerator[str, None]:
        """Streaming chat completion: yields text chunks."""
        ...
