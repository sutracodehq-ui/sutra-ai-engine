"""
Application configuration — single source of truth for all settings.

Reads from environment variables with sensible defaults.
Uses Pydantic Settings for type coercion and validation.
"""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure .env is loaded even if not injected by container runtime
env_path = Path(__file__).parents[2] / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ─── App ────────────────────────────────────────
    app_name: str = "SutraAI"
    app_env: str = "local"
    debug: bool = True
    api_v1_prefix: str = "/v1"

    # ─── Database ───────────────────────────────────
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/sutra_ai"

    # ─── Redis ──────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ─── Auth ───────────────────────────────────────
    master_api_key: str = "sk_master_change_me_in_production"

    # ─── AI Driver (primary text completion) ────────
    ai_driver: str = "ollama"
    ai_fallback_driver: str = "groq"
    # Comma-separated override for DriverRegistry chain (empty = use intelligence_config resilience.global_driver_chain)
    ai_driver_chain: str = ""

    # ─── Pipeline Routing (capability-specific) ─────
    # Vision: image analysis, OCR, visual understanding
    ai_vision_driver: str = "gemini"
    ai_vision_model: str = "gemini-2.0-flash"
    # Image generation: DALL-E, Imagen, etc.
    ai_image_driver: str = "openai"
    ai_image_model: str = "dall-e-3"

    # ─── Fal.ai (Image Generation) ─────────────────
    fal_key: str = ""

    # ─── ElevenLabs (Premium TTS) ──────────────────
    elevenlabs_api_key: str = ""

    # ─── OpenAI ─────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_max_tokens: int = 2048
    openai_temperature: float = 0.7

    # ─── Anthropic ──────────────────────────────────
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_max_tokens: int = 2048
    anthropic_temperature: float = 0.7

    # ─── Google Gemini ──────────────────────────────
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_max_tokens: int = 2048
    gemini_temperature: float = 0.7

    # ─── Groq ───────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    groq_max_tokens: int = 2048
    groq_temperature: float = 0.7

    # ─── Sarvam AI (Indian language models) ─────────
    sarvam_api_key: str = ""
    sarvam_model: str = "sarvam-m"
    sarvam_max_tokens: int = 2048
    sarvam_temperature: float = 0.7

    # ─── NVIDIA NIM (high-perf frontier models) ─────
    nvidia_api_key: str = ""
    nvidia_model: str = "meta/llama-3.1-405b-instruct"
    nvidia_max_tokens: int = 2048
    nvidia_temperature: float = 0.7

    # ─── Ollama ─────────────────────────────────────
    ollama_base_url: str = "http://sutra-ai-ollama:11434"
    ollama_model: str = "gemma4:e4b"
    ollama_max_tokens: int = 2048
    ollama_temperature: float = 0.7
    ollama_timeout_connect: int = 5       # Fail fast if Ollama is down
    ollama_timeout_read: int = 20         # Fast fallback to cloud if local hangs

    # ─── Fast Local (vLLM/TGI OpenAI-compatible) ───
    # Optional hot-path server for ultra-low-latency streaming.
    fast_local_api_key: str = "local"
    fast_local_model: str = "qwen2.5:3b"
    fast_local_max_tokens: int = 1024
    fast_local_temperature: float = 0.3

    # ─── LLM Queue ──────────────────────────────────
    llm_max_parallel: int = 10  # concurrent inference slots (cloud APIs handle unlimited)

    # ─── Qdrant (vector store) ──────────────────────
    qdrant_url: str = "http://sutra-ai-qdrant:6333"
    qdrant_api_key: str = ""
    embedding_model: str = "nomic-embed-text"
    # Must match the embedding model output size (nomic-embed-text → 768).
    embedding_vector_size: int = 768

    # ─── Cloudflare R2 ──────────────────────────────
    r2_access_key: str = ""
    r2_secret_key: str = ""
    r2_bucket: str = "sutra-ai-storage"
    r2_endpoint: str = ""

    # ─── Tavily (Web Search) ────────────────────────
    tavily_api_key: str = ""

    # ─── Intelligence Features ─────────────────────
    ai_web_search_enabled: bool = True
    ai_chain_of_thought_enabled: bool = True

    # ─── Quality Gate ───────────────────────────────
    ai_quality_gate_enabled: bool = True
    ai_quality_gate_threshold: int = 6
    ai_quality_gate_retries: int = 1

    # ─── Caching ────────────────────────────────────
    ai_prompt_cache_enabled: bool = True
    ai_prompt_cache_ttl: int = 7200
    ai_semantic_cache_enabled: bool = True
    ai_semantic_cache_threshold: float = 0.83
    ai_semantic_cache_ttl: int = 28800

    # ─── Smart Router ──────────────────────────────
    ai_smart_router_enabled: bool = True

    # ─── Circuit Breaker ───────────────────────────
    ai_circuit_breaker_threshold: int = 3
    ai_circuit_breaker_cooldown: int = 60

    # ─── Rate Limits ───────────────────────────────
    ai_rate_limit_rpm: int = 30

    # ─── Token Budget ──────────────────────────────
    ai_token_budget_enabled: bool = True
    ai_token_budget_monthly: int = 500000

    # ─── Auto Learning ─────────────────────────────
    ai_auto_learning_enabled: bool = True
    ai_meta_prompt_enabled: bool = True
    ai_meta_prompt_model: str = "gemini-2.0-flash"  # Flash is fast and cheap for reasoning
    ai_meta_prompt_threshold: int = 20              # Min feedback items before optimization
    ai_edit_analysis: bool = True
    ai_ab_testing: bool = True
    ai_explore_rate: float = 0.2
    ai_agent_memory_enabled: bool = False

    # ─── Hybrid Routing ──────────────────────────────
    ai_hybrid_routing: bool = True
    ai_hybrid_quality_threshold: int = 7
    ai_hybrid_fast_path_threshold: float = 8.0
    ai_hybrid_direct_cloud_threshold: float = 5.0

    # ─── Celery ────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"


    # ─── BitNet (1-bit ultra-light local) ─────
    bitnet_api_url: str = "http://sutra-ai-bitnet:8081/v1"
    bitnet_model: str = "bitnet-2b"
    bitnet_max_tokens: int = 512
    bitnet_temperature: float = 0.3

@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — parsed once per process."""
    return Settings()
