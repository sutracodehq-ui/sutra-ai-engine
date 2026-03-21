"""
Application configuration — single source of truth for all settings.

Reads from environment variables with sensible defaults.
Uses Pydantic Settings for type coercion and validation.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ─── App ────────────────────────────────────────
    app_name: str = "SutraAI"
    app_env: str = "local"
    debug: bool = True

    # ─── Database ───────────────────────────────────
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/sutra_ai"

    # ─── Redis ──────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ─── Auth ───────────────────────────────────────
    master_api_key: str = "sk_master_change_me_in_production"

    # ─── AI Driver (primary text completion) ────────
    ai_driver: str = "mock"
    ai_fallback_driver: str = "gemini"

    # ─── Pipeline Routing (capability-specific) ─────
    # Vision: image analysis, OCR, visual understanding
    ai_vision_driver: str = "gemini"
    ai_vision_model: str = "gemini-2.0-flash"
    # Image generation: DALL-E, Imagen, etc.
    ai_image_driver: str = "openai"
    ai_image_model: str = "dall-e-3"

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
    groq_model: str = "llama-3.3-70b-versatile"
    groq_max_tokens: int = 2048
    groq_temperature: float = 0.7

    # ─── Ollama ─────────────────────────────────────
    ollama_base_url: str = "http://sutra-ai-ollama:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_max_tokens: int = 2048
    ollama_temperature: float = 0.7

    # ─── ChromaDB ───────────────────────────────────
    chromadb_url: str = "http://sutra-ai-chromadb:8000"
    embedding_model: str = "nomic-embed-text"

    # ─── Cloudflare R2 ──────────────────────────────
    r2_access_key: str = ""
    r2_secret_key: str = ""
    r2_bucket: str = "sutra-ai-storage"
    r2_endpoint: str = ""

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

    # ─── Celery ────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — parsed once per process."""
    return Settings()
