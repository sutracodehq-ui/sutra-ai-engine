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
import contextlib
import json
import logging
import re
import threading
from datetime import datetime
from typing import AsyncGenerator, Optional

import httpx

from app.config import get_settings
from app.services.intelligence.config_loader import get_intelligence_config, get_provider_config
from app.services.drivers.base import LlmDriver, LlmResponse

logger = logging.getLogger(__name__)

# Strip YAML-only keys before passing provider dict into OpenAICompatDriver(...)
_PROVIDER_CFG_STRIP = frozenset(
    {
        "fallback_model",
        "fallback_models",
        "timeout_connect_s",
        "timeout_read_s",
        "profiles",
    }
)


def provider_fallback_model_list(driver: str) -> list[str]:
    """Ordered tiny-model cascade for one driver (from intelligence_config providers + resilience cap)."""
    intel = get_intelligence_config()
    res = intel.get("resilience") or {}
    max_n = max(1, int(res.get("max_fallback_models_per_driver", 8)))
    pc = (intel.get("providers") or {}).get(driver, {}) or {}
    raw = pc.get("fallback_models")
    legacy = pc.get("fallback_model")
    out: list[str] = []
    if isinstance(raw, list):
        out.extend(str(x).strip() for x in raw if x)
    elif raw:
        out.append(str(raw).strip())
    if legacy:
        leg = str(legacy).strip()
        if leg and leg not in out:
            out.insert(0, leg)
    seen: set[str] = set()
    uniq: list[str] = []
    for m in out:
        if m and m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq[:max_n]


def _has_text_response(resp: LlmResponse | None) -> bool:
    """Treat empty/whitespace bodies as failed attempts for fallback progression."""
    return bool(resp and isinstance(resp.content, str) and resp.content.strip())


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
                 temperature: float = 0.7, timeout_connect: int = 5, timeout_read: int = 20):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        # Pre-build httpx.Timeout once at init — O(1) reuse on every call
        self._timeout = httpx.Timeout(
            connect=float(timeout_connect), read=float(timeout_read),
            write=10.0, pool=10.0,
        )
        self._cfg = get_intelligence_config()

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
        profile_opts = self._profile_options(stream=False, opts=opts)
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"num_predict": opts.get("max_tokens", self._max_tokens),
                               "temperature": opts.get("temperature", self._temperature),
                               **profile_opts}}

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
        profile_opts = self._profile_options(stream=True, opts=opts)
        payload = {"model": model, "messages": messages, "stream": True,
                   "options": {"num_predict": opts.get("max_tokens", self._max_tokens),
                               "temperature": opts.get("temperature", self._temperature),
                               **profile_opts}}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.strip():
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content

    def _profile_options(self, *, stream: bool, opts: dict) -> dict:
        """Apply Ollama profile tuning from YAML (low_latency for stream, quality otherwise)."""
        prov = (self._cfg.get("providers") or {}).get("ollama", {}) or {}
        profiles = prov.get("profiles", {}) if isinstance(prov.get("profiles"), dict) else {}
        default_profile = "low_latency" if stream else "quality"
        profile_name = str(opts.get("profile", default_profile))
        selected = profiles.get(profile_name, {})
        if not isinstance(selected, dict):
            return {}
        # Avoid overriding explicit caller params in payload.
        merged = dict(selected)
        if "max_tokens" in opts or "num_predict" in opts:
            merged.pop("num_predict", None)
        if "temperature" in opts:
            merged.pop("temperature", None)
        return merged


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


# ─── Offline emergency driver (no keys, no network) ───────────

class EmergencyFallbackDriver(LlmDriver):
    """
    Last-resort synthesizer when every cloud/local model is down or circuit-open.

    Returns safe, user-visible text so agents never surface RuntimeError to end users.
    """

    _CHUNK = 56

    def name(self) -> str:
        return "offline"

    @staticmethod
    def _flatten_content(content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
                elif isinstance(p, str):
                    parts.append(p)
            return " ".join(parts)
        return str(content)

    @classmethod
    def _gather_messages(cls, messages: list[dict]) -> tuple[str, str]:
        systems: list[str] = []
        users: list[str] = []
        for m in messages:
            role = m.get("role", "")
            text = cls._flatten_content(m.get("content"))
            if role == "system":
                systems.append(text)
            elif role == "user":
                users.append(text)
        return "\n".join(systems), (users[-1] if users else "")

    @staticmethod
    def _tenant_from_system(system: str) -> str | None:
        for pat in (
            r"assistant for \*\*([^*]+)\*\*",
            r"personal AI assistant for \*\*([^*]+)\*\*",
            r"\*\*Brand\*\*:\s*\*\*([^*]+)\*\*",
        ):
            m = re.search(pat, system, re.IGNORECASE)
            if m:
                return re.sub(r"\*+", "", m.group(1)).strip() or None
        return None

    @staticmethod
    def _wants_strict_json(system: str) -> bool:
        lower = system.lower()
        if "strict json" in lower or ("valid json" in lower and "only" in lower):
            return True
        if "required top-level keys:" in lower:
            return True
        if "json only" in lower and ("must" in lower or "respond" in lower):
            return True
        return False

    @staticmethod
    def _sanitize_user_hint(hint: str) -> str:
        """
        Last user message often includes injected [SYSTEM CONTEXT] / policy blocks.
        Never echo that into user-visible briefing or JSON content.
        """
        if not hint or not str(hint).strip():
            return ""
        h = str(hint).strip()
        lower = h.lower()
        pollution = (
            "[system context]",
            "your name is",
            "critical rule:",
            "you are a powerful ai",
            "you are the personal ai assistant",
            "response_schema",
            "output format",
            "required top-level keys",
            "conversational markdown",
            "vidyantra ai",
            "embedded inside",
            "education management",
        )
        if any(p in lower for p in pollution) or len(h) > 800:
            return ""
        return h[:350]

    @staticmethod
    def _parse_json_keys(system: str) -> list[str]:
        m = re.search(r"Required top-level keys:\s*([^\n]+)", system, re.IGNORECASE)
        if m:
            keys = re.findall(r'"([^"]+)"', m.group(1))
            if keys:
                return keys
        return []

    def _day_part(self) -> str:
        h = datetime.now().hour
        if h < 12:
            return "morning"
        if h < 17:
            return "afternoon"
        return "evening"

    def _briefing_markdown(self, tenant: str | None, user_hint: str) -> str:
        who = tenant or "there"
        day = self._day_part()
        hint = self._sanitize_user_hint(user_hint)
        hint_line = f"\n\n*You asked about:* {hint}\n" if hint else ""

        return (
            f"## Good {day}, {who}!\n\n"
            f"Your live AI models are **temporarily unavailable** (network, keys, or load). "
            f"Here is a **standby briefing** so your day can keep moving.{hint_line}\n"
            f"### 📋 Today at a glance\n"
            f"- Skim your calendar and task list for must-dos.\n"
            f"- Block 25 minutes for your hardest task first.\n\n"
            f"### 🌤️ Weather & commute\n"
            f"- Check your local weather app before you head out.\n\n"
            f"### 📰 News & markets\n"
            f"- Open your preferred finance and news sources for a two-minute scan.\n\n"
            f"### 💡 Quick tip\n"
            f"- Pick **one** outcome that would make today a win — protect time for it.\n\n"
            f"**Suggestions:**\n"
            f"- Retry this briefing in a few minutes when models are back.\n"
            f"- Open tasks or calendar in your workspace to sync priorities.\n"
        )

    def _generic_markdown(self, tenant: str | None, user_hint: str) -> str:
        who = tenant or "there"
        day = self._day_part()
        hint = self._sanitize_user_hint(user_hint)
        body = (
            f"Good {day}, **{who}** — connected assistants are offline right now, "
            f"but you can keep working with saved data in the app.\n\n"
        )
        if hint:
            body += f"*Regarding:* {hint}\n\n"
        body += (
            "**Suggestions:**\n"
            "- Retry shortly.\n"
            "- Verify API keys and local model services if you administer this deployment.\n"
        )
        return body

    def _compose_text(self, messages: list[dict]) -> str:
        system, user = self._gather_messages(messages)
        tenant = self._tenant_from_system(system)
        user_safe = self._sanitize_user_hint(user)
        lower = system.lower()
        is_briefing = any(
            w in lower for w in ("briefing", "daily brief", "morning brief", "morning intelligence")
        )
        if self._wants_strict_json(system):
            keys = self._parse_json_keys(system)
            if not keys:
                keys = ["content", "suggestions"]
            payload: dict = {}
            for k in keys:
                kl = k.lower()
                if kl == "suggestions":
                    payload[k] = [
                        "Retry when live models are available.",
                        "Continue with manual steps using data already in your workspace.",
                    ]
                elif kl in ("content", "message", "response", "answer", "text", "body", "summary"):
                    # Human-visible string only — never inject raw system policy or mega user blobs.
                    if is_briefing:
                        inner = self._briefing_markdown(tenant, user_safe)
                    else:
                        inner = self._generic_markdown(tenant, user_safe)
                    payload[k] = inner[:8000] if len(inner) > 8000 else inner
                elif kl == "result" and "result" in keys:
                    payload[k] = {
                        "advice": self._briefing_markdown(tenant, user_safe)
                        if is_briefing
                        else self._generic_markdown(tenant, user_safe),
                        "action_items": [
                            "Retry this assistant when AI services are back.",
                            "Use saved dashboard data in the meantime.",
                        ],
                    }
                else:
                    payload[k] = ""
            return json.dumps(payload, ensure_ascii=False)

        if is_briefing:
            return self._briefing_markdown(tenant, user_safe)
        return self._generic_markdown(tenant, user_safe)

    async def complete(self, system_prompt: str, user_prompt: str, **opts) -> LlmResponse:
        return await self.chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            **opts,
        )

    async def chat(self, messages: list[dict], **opts) -> LlmResponse:
        text = self._compose_text(messages)
        return LlmResponse(
            content=text,
            raw_response=text,
            prompt_tokens=0,
            completion_tokens=len(text.split()),
            total_tokens=len(text.split()),
            model="offline-emergency",
            driver=self.name(),
        )

    async def stream(self, messages: list[dict], **opts) -> AsyncGenerator[str, None]:
        text = self._compose_text(messages)
        for i in range(0, len(text), self._CHUNK):
            yield text[i : i + self._CHUNK]


# ─── Driver Registry (absorbs driver_manager.py) ─────────────

class DriverRegistry:
    """
    Config-driven driver factory.

    Creates driver instances lazily from settings.
    Integrates Guardian's circuit breaker + retry for fallback chains.
    """

    def __init__(self):
        self._instances: dict[str, LlmDriver] = {}
        self._local_stream_sem: asyncio.Semaphore | None = None
        self._local_stream_slots: int = 0

    def _create_driver(self, name: str) -> LlmDriver:
        """Create a driver instance using a polymorphic hashmap pattern."""
        s = get_settings()

        # 1. Specialized Adapters (Polymorphic Registry)
        # Software Factory Principle: Prefer hashmap over anything.
        def _ollama_adapter() -> OllamaAdapter:
            pc = get_provider_config("ollama") or {}
            return OllamaAdapter(
                base_url=pc.get("base_url", s.ollama_base_url),
                model=s.ollama_model,
                max_tokens=s.ollama_max_tokens,
                temperature=s.ollama_temperature,
                timeout_connect=int(pc.get("timeout_connect_s", s.ollama_timeout_connect)),
                timeout_read=int(pc.get("timeout_read_s", s.ollama_timeout_read)),
            )

        adapters = {
            "ollama": _ollama_adapter,
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
            "fast_local": lambda: OpenAICompatDriver(
                "fast_local",
                api_key=s.fast_local_api_key or "local",
                base_url=(get_provider_config("fast_local") or {}).get("base_url", "http://localhost:8000/v1"),
                model=s.fast_local_model,
                max_tokens=s.fast_local_max_tokens,
                temperature=s.fast_local_temperature,
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
            "fast_local": s.fast_local_api_key or "local",
        }

        api_key = auth_keys.get(name)
        meta = get_provider_config(name)

        if not api_key or not meta:
            raise ValueError(f"Unknown driver or missing configuration: {name}")

        full_cfg = {
            **{
                k: v
                for k, v in meta.items()
                if not k.startswith("timeout_") and k not in _PROVIDER_CFG_STRIP
            },
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

    def _driver_configured(self, name: str) -> bool:
        """True if this driver can be constructed (keys + minimal provider metadata)."""
        s = get_settings()
        if name == "ollama":
            meta = get_provider_config("ollama")
            return bool((meta or {}).get("base_url") or s.ollama_base_url)
        if name == "gemini":
            return bool(s.gemini_api_key and get_provider_config("gemini"))
        if name == "anthropic":
            return bool(s.anthropic_api_key and get_provider_config("anthropic"))
        if name == "fast_local":
            meta = get_provider_config("fast_local")
            return bool((meta or {}).get("base_url"))
        auth_keys = {
            "openai": s.openai_api_key,
            "groq": s.groq_api_key,
            "nvidia": s.nvidia_api_key,
            "sarvam": s.sarvam_api_key,
            "bitnet": "local",
            "fast_local": s.fast_local_api_key or "local",
        }
        api_key = auth_keys.get(name)
        meta = get_provider_config(name)
        if not api_key or not isinstance(meta, dict) or not meta.get("base_url"):
            return False
        return True

    def driver_chain(self) -> list[str]:
        """Ordered fallback chain: Settings.ai_driver_chain > YAML global_driver_chain > primary+fallback."""
        s = get_settings()
        raw = (getattr(s, "ai_driver_chain", None) or "").strip()
        if raw:
            chain = [x.strip() for x in raw.split(",") if x.strip()]
        else:
            intel = get_intelligence_config()
            g = (intel.get("resilience") or {}).get("global_driver_chain")
            if isinstance(g, list) and g:
                chain = [str(x).strip() for x in g if str(x).strip()]
            else:
                chain = [s.ai_driver]
                if s.ai_fallback_driver and s.ai_fallback_driver != s.ai_driver:
                    chain.append(s.ai_fallback_driver)
        return [d for d in chain if self._driver_configured(d)]

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
        Try primary driver, then each tiny model in fallback_models (YAML), then chain.
        """
        if not guardian.circuit_breaker.is_available(driver_override):
            logger.warning(f"Driver: {driver_override} circuit OPEN, falling back to chain")
            return None

        d = self.driver(driver_override)
        try:
            result = await guardian.with_retry(callback, d, **clean)
            if _has_text_response(result):
                guardian.circuit_breaker.record_success(driver_override)
                result.metadata = result.metadata or {}
                result.metadata.setdefault("fallback_hop", 0)
                result.metadata.setdefault("fallback_driver", driver_override)
                return result
            logger.warning(
                f"Driver: {driver_override} {operation} returned empty content; trying tiny-model chain"
            )
        except Exception as e:
            logger.warning(f"Driver: {driver_override} {operation} failed ({e})")

        models = provider_fallback_model_list(driver_override)
        primary_model = clean.get("model")
        for hop, fb_model in enumerate(models):
            if primary_model and fb_model == primary_model:
                continue
            logger.warning(
                f"Driver: {driver_override} retrying tiny-model hop {hop + 1}/{len(models)}: {fb_model}"
            )
            try:
                light_clean = {**clean, "model": fb_model}
                result = await guardian.with_retry(callback, d, **light_clean)
                if _has_text_response(result):
                    guardian.circuit_breaker.record_success(driver_override)
                    result.metadata = result.metadata or {}
                    result.metadata["fallback_hop"] = hop + 1
                    result.metadata["fallback_driver"] = driver_override
                    result.metadata["fallback_model"] = fb_model
                    return result
                logger.warning(
                    f"Driver: {driver_override} model {fb_model} returned empty content; continuing"
                )
            except Exception as e2:
                logger.warning(f"Driver: {driver_override} model {fb_model} failed ({e2})")

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
        self,
        driver_name: str,
        messages: list[dict],
        first_token_timeout: float,
        stream_inactivity_timeout: float | None = None,
        **clean,
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
        while True:
            try:
                if stream_inactivity_timeout:
                    chunk = await asyncio.wait_for(
                        aiter.__anext__(), timeout=stream_inactivity_timeout
                    )
                else:
                    chunk = await aiter.__anext__()
            except StopAsyncIteration:
                break
            yield chunk

    def _ensure_local_stream_semaphore(self):
        cfg = (get_intelligence_config().get("resilience") or {}).get("admission_control", {})
        slots = max(1, int(cfg.get("max_local_streams", 2)))
        if self._local_stream_sem is None or self._local_stream_slots != slots:
            self._local_stream_slots = slots
            self._local_stream_sem = asyncio.Semaphore(slots)

    def _resolve_vps_profile(self) -> dict:
        """Select the active VPS load profile based on current stream count."""
        vps_cfg = get_intelligence_config().get("vps_admission", {})
        if not vps_cfg.get("enabled", False):
            return {}
        profiles = vps_cfg.get("profiles", {})
        active = self._local_stream_slots - (self._local_stream_sem._value if self._local_stream_sem else self._local_stream_slots)
        for level in ("critical", "high_load"):
            profile = profiles.get(level, {})
            trigger = profile.get("trigger_active_streams")
            if trigger is not None and active >= int(trigger):
                logger.debug(f"VPS admission: {level} profile (active={active})")
                return profile
        return profiles.get("normal", {})

    @contextlib.asynccontextmanager
    async def _admission_slot(self, driver_name: str):
        local_drivers = {"ollama", "bitnet", "fast_local"}
        if driver_name not in local_drivers:
            yield
            return
        self._ensure_local_stream_semaphore()
        profile = self._resolve_vps_profile()
        if profile:
            max_streams = profile.get("max_local_streams")
            if max_streams is not None and int(max_streams) == 0:
                raise asyncio.TimeoutError(
                    f"VPS critical: local streams disabled for {driver_name}"
                )
        fallback_cfg = (get_intelligence_config().get("resilience") or {}).get("admission_control", {})
        wait_s = float(profile.get("max_queue_wait_s", fallback_cfg.get("max_queue_wait_s", 0.35)))
        assert self._local_stream_sem is not None
        try:
            await asyncio.wait_for(self._local_stream_sem.acquire(), timeout=wait_s)
        except asyncio.TimeoutError as e:
            raise asyncio.TimeoutError(
                f"admission timeout for {driver_name} after {wait_s}s"
            ) from e
        try:
            yield
        finally:
            self._local_stream_sem.release()

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
        timeouts_cfg = intel_cfg.get("timeouts", {}) or {}
        resilience_timeouts = (intel_cfg.get("resilience") or {}).get("timeouts", {}) or {}
        first_token_timeout = float(
            timeouts_cfg.get(
                "first_token_timeout_s",
                resilience_timeouts.get(
                    "first_token_timeout_s",
                    resilience_timeouts.get("stream_first_token_s", 60),
                ),
            )
        )
        stream_inactivity_timeout = float(
            timeouts_cfg.get(
                "stream_inactivity_timeout_s",
                resilience_timeouts.get("stream_inactivity_timeout_s", 8),
            )
        )

        if driver_override:
            if not guardian.circuit_breaker.is_available(driver_override):
                logger.warning(f"Driver: {driver_override} circuit OPEN, falling back to chain")
            else:
                primary_model = clean.get("model")
                tiny_chain = [None] + provider_fallback_model_list(driver_override)
                for hop, fb_model in enumerate(tiny_chain):
                    stream_clean = clean if fb_model is None else {**clean, "model": fb_model}
                    if fb_model is not None and fb_model == primary_model:
                        continue
                    label = "primary" if fb_model is None else fb_model
                    try:
                        seen_chunk = False
                        async with self._admission_slot(driver_override):
                            async for chunk in self._stream_with_first_token_timeout(
                                driver_override,
                                messages,
                                first_token_timeout,
                                stream_inactivity_timeout,
                                **stream_clean,
                            ):
                                seen_chunk = True
                                yield chunk
                        if not seen_chunk:
                            raise RuntimeError("empty stream")
                        guardian.circuit_breaker.record_success(driver_override)
                        return
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Driver: {driver_override} stream hop {hop} ({label}) "
                            f"no first token in {first_token_timeout}s"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Driver: {driver_override} stream hop {hop} ({label}) failed: {e}"
                        )
                guardian.circuit_breaker.record_failure(driver_override)
                logger.warning(
                    f"Driver: {driver_override} exhausted primary + tiny models — falling to chain"
                )

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
                self.driver(driver_name)
            except ValueError as ve:
                logger.warning(f"Driver: {driver_name} stream skip (not configured): {ve}")
                continue
            try:
                seen_chunk = False
                async with self._admission_slot(driver_name):
                    async for chunk in self._stream_with_first_token_timeout(
                        driver_name,
                        messages,
                        first_token_timeout,
                        stream_inactivity_timeout,
                        **chain_clean,
                    ):
                        seen_chunk = True
                        yield chunk
                if not seen_chunk:
                    raise RuntimeError("empty stream")
                guardian.circuit_breaker.record_success(driver_name)
                return
            except Exception as e:
                guardian.circuit_breaker.record_failure(driver_name)
                logger.warning(f"Driver: {driver_name} stream failed: {e}")
        logger.error("DriverRegistry: all drivers failed for stream; emitting offline emergency text")
        async for chunk in EmergencyFallbackDriver().stream(messages, **chain_clean):
            yield chunk

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
            except ValueError as ve:
                logger.warning(f"Driver: {driver_name} skipped (not configured): {ve}")
                continue
            try:
                result = await guardian.with_retry(callback, d)
                if _has_text_response(result):
                    guardian.circuit_breaker.record_success(driver_name)
                    result.metadata = result.metadata or {}
                    result.metadata.setdefault("fallback_driver", driver_name)
                    result.metadata.setdefault("fallback_hop", 0)
                    return result
                logger.warning(
                    f"Driver: {driver_name} returned empty content for {operation}; trying next driver"
                )
            except Exception as e:
                guardian.circuit_breaker.record_failure(driver_name)
                last_err = e
                logger.warning(f"Driver: {driver_name} failed for {operation}: {e}")
        logger.error(
            "DriverRegistry: all drivers failed for %s; returning offline emergency response",
            operation,
        )
        emergency = await callback(EmergencyFallbackDriver())
        if isinstance(emergency, LlmResponse):
            emergency.metadata = emergency.metadata or {}
            emergency.metadata["emergency_activations"] = 1
        return emergency


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
