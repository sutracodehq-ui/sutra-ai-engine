"""
Driver — Polymorphic LLM adapter with config-driven provider registry.

Software Factory Principle: One file for all LLM providers.
Adding a new provider = add YAML config entry. Zero code changes
for OpenAI-compatible providers.

Absorbs: driver_manager.py + all 8 driver files
         (openai, groq, ollama, gemini, anthropic, nvidia, sarvam, mock)

Architecture:
    YAML defines provider type + connection info
    → 4 adapter types handle all providers:
       1. openai_compatible  — OpenAI, Groq, Nvidia, Sarvam (AsyncOpenAI)
       2. ollama             — Local Ollama HTTP API
       3. google_genai       — Google Gemini SDK
       4. anthropic_native   — Anthropic SDK
"""

import importlib
import json
import logging
import threading
from typing import AsyncGenerator, Optional

import httpx

from app.config import get_settings
from app.services.drivers.base import LlmDriver, LlmResponse

logger = logging.getLogger(__name__)


# ─── OpenAI-Compatible Adapter (covers: openai, groq, nvidia, sarvam) ──

class OpenAICompatDriver(LlmDriver):
    """Single driver for all OpenAI-API-compatible providers."""

    def __init__(self, driver_name: str, *, api_key: str, base_url: str,
                 model: str, max_tokens: int = 2048, temperature: float = 0.7):
        from openai import AsyncOpenAI
        self._name = driver_name
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def name(self) -> str:
        return self._name

    async def complete(self, system_prompt: str, user_prompt: str, **opts) -> LlmResponse:
        return await self.chat([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], **opts)

    async def chat(self, messages: list[dict], **opts) -> LlmResponse:
        resp = await self._client.chat.completions.create(
            model=opts.get("model", self._model), messages=messages,
            max_tokens=opts.get("max_tokens", self._max_tokens),
            temperature=opts.get("temperature", self._temperature),
        )
        choice = resp.choices[0]
        usage = resp.usage
        return LlmResponse(
            content=choice.message.content or "", raw_response=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            model=resp.model, driver=self._name,
        )

    async def stream(self, messages: list[dict], **opts) -> AsyncGenerator[str, None]:
        stream = await self._client.chat.completions.create(
            model=opts.get("model", self._model), messages=messages,
            max_tokens=opts.get("max_tokens", self._max_tokens),
            temperature=opts.get("temperature", self._temperature), stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ─── Ollama Adapter (local HTTP API) ──────────────────────────

class OllamaAdapter(LlmDriver):
    """Ollama local inference via HTTP API."""

    def __init__(self, *, base_url: str, model: str, max_tokens: int = 2048, temperature: float = 0.7):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def name(self) -> str:
        return "ollama"

    async def complete(self, system_prompt: str, user_prompt: str, **opts) -> LlmResponse:
        return await self.chat([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], **opts)

    async def chat(self, messages: list[dict], **opts) -> LlmResponse:
        model = opts.get("model", self._model)
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"num_predict": opts.get("max_tokens", self._max_tokens),
                               "temperature": opts.get("temperature", self._temperature)}}
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = data.get("message", {}).get("content", "")
        return LlmResponse(
            content=content, raw_response=content,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            model=model, driver="ollama",
        )

    async def stream(self, messages: list[dict], **opts) -> AsyncGenerator[str, None]:
        model = opts.get("model", self._model)
        payload = {"model": model, "messages": messages, "stream": True,
                   "options": {"num_predict": opts.get("max_tokens", self._max_tokens),
                               "temperature": opts.get("temperature", self._temperature)}}
        timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.strip():
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content


# ─── Gemini Adapter (Google GenAI SDK) ────────────────────────

class GeminiAdapter(LlmDriver):
    """Google Gemini via google-generativeai SDK."""

    def __init__(self, *, api_key: str, model: str, max_tokens: int = 2048, temperature: float = 0.7):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def name(self) -> str:
        return "gemini"

    async def complete(self, system_prompt: str, user_prompt: str, **opts) -> LlmResponse:
        return await self.chat([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], **opts)

    async def chat(self, messages: list[dict], **opts) -> LlmResponse:
        model_name = opts.get("model", self._model_name)
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        model = self._genai.GenerativeModel(
            model_name=model_name, system_instruction=system,
            generation_config=self._genai.types.GenerationConfig(
                max_output_tokens=opts.get("max_tokens", self._max_tokens),
                temperature=opts.get("temperature", self._temperature),
            ),
        )
        history = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            role = "model" if msg["role"] == "assistant" else "user"
            history.append({"role": role, "parts": msg.get("parts") or [msg["content"]]})
        chat_hist = history[:-1] if len(history) > 1 else []
        last = history[-1]["parts"] if history else [""]
        chat = model.start_chat(history=chat_hist)
        resp = chat.send_message(last)
        content = resp.text or ""
        usage = resp.usage_metadata
        return LlmResponse(
            content=content, raw_response=content,
            prompt_tokens=usage.prompt_token_count if usage else 0,
            completion_tokens=usage.candidates_token_count if usage else 0,
            total_tokens=usage.total_token_count if usage else 0,
            model=model_name, driver="gemini",
        )

    async def stream(self, messages: list[dict], **opts) -> AsyncGenerator[str, None]:
        model_name = opts.get("model", self._model_name)
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        model = self._genai.GenerativeModel(
            model_name=model_name, system_instruction=system,
            generation_config=self._genai.types.GenerationConfig(
                max_output_tokens=opts.get("max_tokens", self._max_tokens),
                temperature=opts.get("temperature", self._temperature),
            ),
        )
        history = [{"role": "model" if m["role"] == "assistant" else "user",
                     "parts": [m["content"]]} for m in messages if m["role"] != "system"]
        chat_hist = history[:-1] if len(history) > 1 else []
        last = history[-1]["parts"][0] if history else ""
        chat = model.start_chat(history=chat_hist)
        for chunk in chat.send_message(last, stream=True):
            if chunk.text:
                yield chunk.text


# ─── Anthropic Adapter (native SDK) ──────────────────────────

class AnthropicAdapter(LlmDriver):
    """Anthropic Claude via native SDK."""

    def __init__(self, *, api_key: str, model: str, max_tokens: int = 2048, temperature: float = 0.7):
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def name(self) -> str:
        return "anthropic"

    async def complete(self, system_prompt: str, user_prompt: str, **opts) -> LlmResponse:
        return await self._call([{"role": "user", "content": user_prompt}], system_prompt, **opts)

    async def chat(self, messages: list[dict], **opts) -> LlmResponse:
        system = ""
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_msgs.append(m)
        return await self._call(chat_msgs, system, **opts)

    async def _call(self, messages: list[dict], system: str, **opts) -> LlmResponse:
        resp = await self._client.messages.create(
            model=opts.get("model", self._model),
            max_tokens=opts.get("max_tokens", self._max_tokens),
            temperature=opts.get("temperature", self._temperature),
            system=system, messages=messages,
        )
        content = resp.content[0].text if resp.content else ""
        return LlmResponse(
            content=content, raw_response=content,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
            model=resp.model, driver="anthropic",
        )

    async def stream(self, messages: list[dict], **opts) -> AsyncGenerator[str, None]:
        system = ""
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_msgs.append(m)
        async with self._client.messages.stream(
            model=opts.get("model", self._model),
            max_tokens=opts.get("max_tokens", self._max_tokens),
            temperature=opts.get("temperature", self._temperature),
            system=system, messages=chat_msgs,
        ) as stream:
            async for text in stream.text_stream:
                yield text


# ─── Driver Registry (absorbs driver_manager.py) ─────────────

class DriverRegistry:
    """
    Config-driven driver factory.

    Creates driver instances lazily from settings.
    Integrates Guardian's circuit breaker + retry for fallback chains.
    """

    def __init__(self):
        self._instances: dict[str, LlmDriver] = {}

    def _create_driver(self, name: str) -> LlmDriver:
        """Create a driver instance from env settings."""
        s = get_settings()

        if name == "ollama":
            return OllamaAdapter(
                base_url=s.ollama_base_url, model=s.ollama_model,
                max_tokens=s.ollama_max_tokens, temperature=s.ollama_temperature,
            )
        if name == "gemini":
            return GeminiAdapter(
                api_key=s.gemini_api_key, model=s.gemini_model,
                max_tokens=s.gemini_max_tokens, temperature=s.gemini_temperature,
            )
        if name == "anthropic":
            return AnthropicAdapter(
                api_key=s.anthropic_api_key, model=s.anthropic_model,
                max_tokens=s.anthropic_max_tokens, temperature=s.anthropic_temperature,
            )

        # All OpenAI-compatible providers
        provider_config = {
            "openai": {"api_key": s.openai_api_key, "base_url": "https://api.openai.com/v1",
                        "model": s.openai_model, "max_tokens": s.openai_max_tokens, "temperature": s.openai_temperature},
            "groq": {"api_key": s.groq_api_key, "base_url": "https://api.groq.com/openai/v1",
                      "model": s.groq_model, "max_tokens": s.groq_max_tokens, "temperature": s.groq_temperature},
            "nvidia": {"api_key": s.nvidia_api_key, "base_url": "https://integrate.api.nvidia.com/v1",
                        "model": s.nvidia_model, "max_tokens": s.nvidia_max_tokens, "temperature": s.nvidia_temperature},
            "sarvam": {"api_key": s.sarvam_api_key, "base_url": "https://api.sarvam.ai/v1",
                        "model": s.sarvam_model, "max_tokens": s.sarvam_max_tokens, "temperature": s.sarvam_temperature},
        }

        cfg = provider_config.get(name)
        if not cfg:
            raise ValueError(f"Unknown driver: {name}")
        return OpenAICompatDriver(name, **cfg)

    def driver(self, name: str) -> LlmDriver:
        """Get or create a driver instance (cached)."""
        if name not in self._instances:
            self._instances[name] = self._create_driver(name)
        return self._instances[name]

    def driver_chain(self) -> list[str]:
        """Get ordered driver fallback chain."""
        s = get_settings()
        chain = [s.ai_driver]
        if s.ai_fallback_driver and s.ai_fallback_driver != s.ai_driver:
            chain.append(s.ai_fallback_driver)
        return chain

    # Keys to strip from options before passing to drivers
    _STRIP = {"messages", "driver_override", "model_override", "driver", "model_name"}

    def _clean(self, opts: dict, model_override: str | None = None) -> dict:
        cleaned = {k: v for k, v in opts.items() if k not in self._STRIP}
        if model_override:
            cleaned["model"] = model_override
        return cleaned

    async def complete(self, system_prompt: str, user_prompt: str,
                       driver_override: str | None = None, model_override: str | None = None,
                       **opts) -> LlmResponse:
        """Run completion through override or fallback chain."""
        from app.services.intelligence.guardian import get_guardian
        guardian = get_guardian()
        clean = self._clean(opts, model_override)

        if driver_override:
            d = self.driver(driver_override)
            result = await guardian.with_retry(d.complete, system_prompt, user_prompt, **clean)
            guardian.circuit_breaker.record_success(driver_override)
            return result

        return await self._with_fallback(
            lambda d: d.complete(system_prompt, user_prompt, **clean), "complete"
        )

    async def chat(self, messages: list[dict], driver_override: str | None = None,
                   model_override: str | None = None, **opts) -> LlmResponse:
        from app.services.intelligence.guardian import get_guardian
        guardian = get_guardian()
        clean = self._clean(opts, model_override)

        if driver_override:
            d = self.driver(driver_override)
            result = await guardian.with_retry(d.chat, messages, **clean)
            guardian.circuit_breaker.record_success(driver_override)
            return result

        return await self._with_fallback(lambda d: d.chat(messages, **clean), "chat")

    async def stream(self, system_prompt: str | None = None, user_prompt: str | None = None,
                     messages: list[dict] | None = None, driver_override: str | None = None,
                     model_override: str | None = None, **opts) -> AsyncGenerator[str, None]:
        from app.services.intelligence.guardian import get_guardian
        guardian = get_guardian()
        clean = self._clean(opts, model_override)

        if messages is None:
            messages = [{"role": "system", "content": system_prompt or "You are a helpful assistant."},
                        {"role": "user", "content": user_prompt or ""}]

        if driver_override:
            d = self.driver(driver_override)
            async for chunk in d.stream(messages, **clean):
                yield chunk
            guardian.circuit_breaker.record_success(driver_override)
            return

        chain = self.driver_chain()
        for driver_name in chain:
            if not guardian.circuit_breaker.is_available(driver_name):
                continue
            try:
                d = self.driver(driver_name)
                async for chunk in d.stream(messages, **clean):
                    yield chunk
                guardian.circuit_breaker.record_success(driver_name)
                return
            except Exception as e:
                guardian.circuit_breaker.record_failure(driver_name)
                logger.warning(f"Driver: {driver_name} stream failed: {e}")
        raise RuntimeError("All AI drivers failed")

    async def _with_fallback(self, callback, operation: str) -> LlmResponse:
        from app.services.intelligence.guardian import get_guardian
        guardian = get_guardian()
        chain = self.driver_chain()
        last_err = None
        for driver_name in chain:
            if not guardian.circuit_breaker.is_available(driver_name):
                continue
            try:
                d = self.driver(driver_name)
                result = await guardian.with_retry(callback, d)
                guardian.circuit_breaker.record_success(driver_name)
                return result
            except Exception as e:
                guardian.circuit_breaker.record_failure(driver_name)
                last_err = e
                logger.warning(f"Driver: {driver_name} failed for {operation}: {e}")
        raise last_err or RuntimeError("All AI drivers failed")


# ─── Singleton ──────────────────────────────────────────────────

_registry: DriverRegistry | None = None
_registry_lock = threading.Lock()


def get_driver_registry() -> DriverRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = DriverRegistry()
    return _registry
