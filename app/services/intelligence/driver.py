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

import asyncio
import importlib
import json
import logging
import threading
from typing import AsyncGenerator, Optional

import httpx

from app.config import get_settings
from app.services.intelligence.config_loader import get_provider_config
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

    def __init__(self, *, base_url: str, model: str, max_tokens: int = 2048,
                 temperature: float = 0.7, timeout_connect: int = 10, timeout_read: int = 300):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        # Pre-build httpx.Timeout once at init — O(1) reuse on every call
        self._timeout = httpx.Timeout(
            connect=float(timeout_connect), read=float(timeout_read),
            write=10.0, pool=10.0,
        )

    def name(self) -> str:
        return "ollama"

    @staticmethod
    def _expects_json(messages: list[dict]) -> bool:
        """Detect if the system prompt expects JSON output."""
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        lower = system.lower()
        return any(k in lower for k in ("json", "response_schema", "structured", "{", "fields:"))

    async def complete(self, system_prompt: str, user_prompt: str, **opts) -> LlmResponse:
        return await self.chat([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], **opts)

    async def chat(self, messages: list[dict], **opts) -> LlmResponse:
        model = opts.get("model", self._model)
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"num_predict": opts.get("max_tokens", self._max_tokens),
                               "temperature": opts.get("temperature", self._temperature)}}

        # Enforce structured JSON output when the prompt expects it.
        if opts.get("format") == "json" or self._expects_json(messages):
            payload["format"] = "json"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
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
        async with httpx.AsyncClient(timeout=self._timeout) as client:
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
        """Create a driver instance using a polymorphic hashmap pattern."""
        s = get_settings()

        # 1. Specialized Adapters (Polymorphic Registry)
        # Software Factory Principle: Prefer hashmap over anything.
        adapters = {
            "ollama": lambda: OllamaAdapter(
                base_url=get_provider_config("ollama").get("base_url", s.ollama_base_url),
                model=s.ollama_model,
                max_tokens=s.ollama_max_tokens,
                temperature=s.ollama_temperature,
                timeout_connect=s.ollama_timeout_connect,
                timeout_read=s.ollama_timeout_read,
            ),
            "gemini": lambda: GeminiAdapter(
                api_key=s.gemini_api_key, 
                model=s.gemini_model,
                max_tokens=s.gemini_max_tokens, 
                temperature=s.gemini_temperature,
            ),
            "anthropic": lambda: AnthropicAdapter(
                api_key=s.anthropic_api_key, 
                model=s.anthropic_model,
                max_tokens=s.anthropic_max_tokens, 
                temperature=s.anthropic_temperature,
            ),
        }

        if name in adapters:
            return adapters[name]()

        # 2. Generic OpenAI-Compatible (Hashmap-driven)
        auth_keys = {
            "openai": s.openai_api_key,
            "groq": s.groq_api_key,
            "nvidia": s.nvidia_api_key,
            "sarvam": s.sarvam_api_key,
            "bitnet": "local",
        }

        api_key = auth_keys.get(name)
        meta = get_provider_config(name)

        if not api_key or not meta:
            raise ValueError(f"Unknown driver or missing configuration: {name}")

        full_cfg = {
            # Strip provider-level timeout keys (only used by OllamaAdapter)
            **{k: v for k, v in meta.items() if not k.startswith("timeout_")},
            "api_key": api_key,
            "model": getattr(s, f"{name}_model", None),
            "max_tokens": getattr(s, f"{name}_max_tokens", 512),
            "temperature": getattr(s, f"{name}_temperature", 0.7),
        }

        return OpenAICompatDriver(name, **full_cfg)

    @property
    def circuit_breaker(self):
        """Standardized access to driver circuit states."""
        from app.services.intelligence.guardian import get_guardian
        return get_guardian().circuit_breaker

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

    async def _try_with_local_fallback(
        self, driver_override: str, callback, operation: str, guardian, **clean
    ) -> LlmResponse | None:
        """
        Try primary driver, then lighter fallback_model if it fails.
        Returns None if both fail (caller should fall to chain).
        """
        from app.services.intelligence.config_loader import get_intelligence_config

        if not guardian.circuit_breaker.is_available(driver_override):
            logger.warning(f"Driver: {driver_override} circuit OPEN, falling back to chain")
            return None

        d = self.driver(driver_override)
        try:
            result = await guardian.with_retry(callback, d, **clean)
            guardian.circuit_breaker.record_success(driver_override)
            return result
        except Exception as e:
            logger.warning(f"Driver: {driver_override} {operation} failed ({e})")

            # Try lighter fallback model before going to cloud
            provider_cfg = get_intelligence_config().get("providers", {}).get(driver_override, {})
            fallback_model = provider_cfg.get("fallback_model")
            if fallback_model:
                logger.warning(f"Driver: {driver_override} retrying with light model: {fallback_model}")
                try:
                    light_clean = {**clean, "model": fallback_model}
                    result = await guardian.with_retry(callback, d, **light_clean)
                    guardian.circuit_breaker.record_success(driver_override)
                    return result
                except Exception as e2:
                    guardian.circuit_breaker.record_failure(driver_override)
                    logger.warning(f"Driver: {driver_override} light model also failed ({e2})")
            else:
                guardian.circuit_breaker.record_failure(driver_override)

            return None

    async def complete(self, system_prompt: str, user_prompt: str,
                       driver_override: str | None = None, model_override: str | None = None,
                       **opts) -> LlmResponse:
        """Run completion with local-first model degradation."""
        from app.services.intelligence.guardian import get_guardian
        guardian = get_guardian()
        clean = self._clean(opts, model_override)

        if driver_override:
            result = await self._try_with_local_fallback(
                driver_override,
                lambda d, **kw: d.complete(system_prompt, user_prompt, **kw),
                "complete", guardian, **clean
            )
            if result is not None:
                return result

        # Strip model override — each chain driver uses its own default
        chain_clean = {k: v for k, v in clean.items() if k != "model"}
        return await self._with_fallback(
            lambda d: d.complete(system_prompt, user_prompt, **chain_clean), "complete"
        )

    async def chat(self, messages: list[dict], driver_override: str | None = None,
                   model_override: str | None = None, **opts) -> LlmResponse:
        from app.services.intelligence.guardian import get_guardian
        guardian = get_guardian()
        clean = self._clean(opts, model_override)

        if driver_override:
            result = await self._try_with_local_fallback(
                driver_override,
                lambda d, **kw: d.chat(messages, **kw),
                "chat", guardian, **clean
            )
            if result is not None:
                return result

        # Strip model override — each chain driver uses its own default
        chain_clean = {k: v for k, v in clean.items() if k != "model"}
        return await self._with_fallback(lambda d: d.chat(messages, **chain_clean), "chat")

    async def _stream_with_first_token_timeout(
        self, driver_name: str, messages: list[dict], first_token_timeout: float, **clean
    ) -> AsyncGenerator[str, None]:
        """
        Stream with a first-token deadline.

        If the driver doesn't produce the first token within `first_token_timeout` seconds,
        raises asyncio.TimeoutError so the caller can fall back to the next driver.
        After the first token arrives, subsequent tokens use the normal httpx read timeout.
        """
        d = self.driver(driver_name)
        gen = d.stream(messages, **clean)
        aiter = gen.__aiter__()

        # Wait for first token with deadline
        first_chunk = await asyncio.wait_for(aiter.__anext__(), timeout=first_token_timeout)
        yield first_chunk

        # First token arrived — stream remainder without extra timeout wrapper
        async for chunk in aiter:
            yield chunk

    async def stream(self, system_prompt: str | None = None, user_prompt: str | None = None,
                     messages: list[dict] | None = None, driver_override: str | None = None,
                     model_override: str | None = None, fallback_chain: list[str] | None = None,
                     **opts) -> AsyncGenerator[str, None]:
        """
        Stream with local-first model degradation.

        Flow (all config-driven):
        1. Primary driver + model (e.g. ollama/gemma4:e4b) with first-token timeout
        2. If timeout → same driver + lighter fallback_model (e.g. ollama/qwen3:1.7b)
        3. If that fails too → circuit breaker OPEN → fall to chain (bitnet → groq → ...)
        """
        from app.services.intelligence.guardian import get_guardian
        from app.services.intelligence.config_loader import get_intelligence_config

        guardian = get_guardian()
        clean = self._clean(opts, model_override)

        if messages is None:
            messages = [{"role": "system", "content": system_prompt or "You are a helpful assistant."},
                        {"role": "user", "content": user_prompt or ""}]

        # Read config (cached, O(1) amortized)
        intel_cfg = get_intelligence_config()
        timeouts_cfg = intel_cfg.get("timeouts", {})
        first_token_timeout = float(timeouts_cfg.get("first_token_timeout_s", 60))

        if driver_override:
            if not guardian.circuit_breaker.is_available(driver_override):
                logger.warning(f"Driver: {driver_override} circuit OPEN, falling back to chain")
            else:
                # ── Stage 1: Primary model with first-token timeout ──
                try:
                    async for chunk in self._stream_with_first_token_timeout(
                        driver_override, messages, first_token_timeout, **clean
                    ):
                        yield chunk
                    guardian.circuit_breaker.record_success(driver_override)
                    return
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Driver: {driver_override} no first token in {first_token_timeout}s"
                    )
                    # ── Stage 2: Try lighter fallback model on SAME driver ──
                    # Don't mark circuit OPEN yet — the driver may be fine,
                    # just the heavy model is slow to load.
                    provider_cfg = intel_cfg.get("providers", {}).get(driver_override, {})
                    fallback_model = provider_cfg.get("fallback_model")

                    if fallback_model:
                        logger.warning(
                            f"Driver: {driver_override} retrying with light model: {fallback_model}"
                        )
                        try:
                            light_clean = {**clean, "model": fallback_model}
                            async for chunk in self._stream_with_first_token_timeout(
                                driver_override, messages, first_token_timeout, **light_clean
                            ):
                                yield chunk
                            guardian.circuit_breaker.record_success(driver_override)
                            return
                        except Exception as e2:
                            # Light model also failed → driver is truly down
                            guardian.circuit_breaker.record_failure(driver_override)
                            logger.warning(
                                f"Driver: {driver_override} light model also failed ({e2}), "
                                f"driver is down — falling to chain"
                            )
                    else:
                        # No fallback model configured — mark circuit OPEN
                        guardian.circuit_breaker.record_failure(driver_override)
                        logger.warning(f"Driver: {driver_override} no fallback_model, falling to chain")

                except Exception as e:
                    guardian.circuit_breaker.record_failure(driver_override)
                    logger.warning(f"Driver: {driver_override} stream failed ({e}), falling back to chain")

        # ── Stage 3: Fall back through chain (bitnet → groq → gemini → ...) ──
        chain = fallback_chain or self.driver_chain()
        if driver_override:
            chain = [d for d in chain if d != driver_override]
        # Strip model override — it was meant for the primary driver,
        # each chain driver should use its own default model.
        chain_clean = {k: v for k, v in clean.items() if k != "model"}
        for driver_name in chain:
            if not guardian.circuit_breaker.is_available(driver_name):
                continue
            try:
                d = self.driver(driver_name)
                async for chunk in d.stream(messages, **chain_clean):
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
