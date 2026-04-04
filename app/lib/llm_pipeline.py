"""
LLM Pipeline — config-driven, single-function LLM execution engine.

Software Factory: ALL LLM calls go through run_pipeline().
Zero hardcoded prompts, driver chains, or timeouts in service code.

Usage:
    from app.lib.llm_pipeline import run_pipeline, get_pipeline_config

    # JSON pipeline — returns parsed dict
    result = await run_pipeline("brand_analyze", {"url": "...", "content": "..."})
    # → {"name": "Vidyantra", "mission": "...", ...}

    # Text pipeline — returns raw string
    result = await run_pipeline("language_translate", {"text": "...", "target_lang": "hi"})
    # → "अनुवादित पाठ..."

Config lives in intelligence_config.yaml → intelligence_pipelines section.
Adding a new pipeline = add YAML entry. Zero Python changes.
"""

import logging
from typing import Any

from app.lib.json_repair import extract_json
from app.services.intelligence.config_loader import get_intelligence_config

logger = logging.getLogger(__name__)


def get_pipeline_config(name: str) -> dict:
    """
    Get pipeline configuration from YAML.

    Returns the pipeline dict or empty dict if not found.
    Cached via get_intelligence_config() (60s TTL).
    """
    config = get_intelligence_config()
    pipelines = config.get("intelligence_pipelines", {})
    return pipelines.get(name, {})


async def run_pipeline(name: str, variables: dict[str, Any]) -> dict | str | None:
    """
    Execute a config-driven LLM pipeline.

    Steps:
      1. Load pipeline config from YAML (prompt, drivers, timeout, etc.)
      2. Format the prompt template with input variables
      3. Loop through driver chain with circuit breaker
      4. Parse output (json_repair for JSON, raw for text)
      5. Validate expected fields
      6. Return clean result or fallback_response

    Args:
        name: Pipeline name (key in intelligence_pipelines YAML section)
        variables: Dict of template variables (e.g., {"url": "...", "content": "..."})

    Returns:
        Parsed dict (json_mode=true), raw string (json_mode=false), or None
    """
    cfg = get_pipeline_config(name)
    if not cfg:
        logger.error(f"LLMPipeline: pipeline '{name}' not found in config")
        return None

    # ─── Build prompt from template ─────────────────────
    system_prompt = cfg.get("system_prompt", "You are a helpful assistant.")
    prompt_template = cfg.get("prompt_template", "")

    if not prompt_template:
        logger.error(f"LLMPipeline: pipeline '{name}' has no prompt_template")
        return None

    # Truncate content if max_content_chars is set
    max_chars = cfg.get("max_content_chars")
    if max_chars:
        for key in ["content", "text"]:
            if key in variables and len(str(variables[key])) > max_chars:
                variables[key] = str(variables[key])[:max_chars]

    try:
        prompt = prompt_template.format(**variables)
    except KeyError as e:
        logger.error(f"LLMPipeline: missing variable {e} for pipeline '{name}'")
        return cfg.get("fallback_response")

    # ─── Pipeline options ───────────────────────────────
    driver_chain = cfg.get("driver_chain", ["groq", "gemini", "anthropic", "ollama"])
    temperature = cfg.get("temperature", 0.7)
    json_mode = cfg.get("json_mode", False)
    expected_fields = cfg.get("expected_fields", [])
    fallback = cfg.get("fallback_response")

    # ─── Execute through driver chain ───────────────────
    from app.services.intelligence.driver import get_driver_registry
    registry = get_driver_registry()

    for driver_name in driver_chain:
        # Check circuit breaker
        if not registry.circuit_breaker.is_available(driver_name):
            logger.debug(f"LLMPipeline[{name}]: {driver_name} circuit OPEN, skipping")
            continue

        try:
            response = await registry.complete(
                system_prompt=system_prompt,
                user_prompt=prompt,
                driver_override=driver_name,
                temperature=temperature,
                json_mode=json_mode,
            )

            if not response.content:
                logger.warning(f"LLMPipeline[{name}]: {driver_name} returned empty content")
                continue

            # ─── Parse output ───────────────────────────
            if json_mode:
                data = extract_json(response.content)
                if data is None:
                    logger.warning(f"LLMPipeline[{name}]: {driver_name} returned invalid JSON")
                    continue

                # Validate expected fields
                if expected_fields and isinstance(data, dict):
                    missing = [f for f in expected_fields if f not in data]
                    if missing:
                        logger.warning(
                            f"LLMPipeline[{name}]: {driver_name} missing fields: {missing}"
                        )
                        # Still return if we got at least some fields
                        if len(missing) > len(expected_fields) / 2:
                            continue  # Too many missing — try next driver

                logger.info(
                    f"LLMPipeline[{name}]: SUCCESS via {driver_name} "
                    f"({response.total_tokens} tokens)"
                )
                return data
            else:
                # Text mode — return raw content
                logger.info(
                    f"LLMPipeline[{name}]: SUCCESS via {driver_name} "
                    f"({response.total_tokens} tokens)"
                )
                return response.content

        except Exception as e:
            logger.warning(f"LLMPipeline[{name}]: {driver_name} failed: {e}")
            continue

    # ─── All drivers failed ─────────────────────────────
    logger.error(f"LLMPipeline[{name}]: ALL drivers failed. Returning fallback.")
    return fallback
