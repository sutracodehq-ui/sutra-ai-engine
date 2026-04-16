"""
LLM Pipeline — config-driven, single-function LLM execution engine.

Software Factory: ALL LLM calls go through run_pipeline().
Zero hardcoded prompts, driver chains, or timeouts in service code.

Usage:
    from app.lib.llm_pipeline import run_pipeline, get_pipeline_config

    # JSON pipeline — returns parsed dict
    result = await run_pipeline("brand_analyze", {"url": "...", "content": "..."}, tenant_id=5)
    # → {"brand_identity": {...}, "voice_and_tone": {...}, ...}

    # Text pipeline — returns raw string
    result = await run_pipeline("language_translate", {"text": "...", "target_lang": "hi"})
    # → "अनुवादित पाठ..."

Config lives in intelligence_config.yaml → intelligence_pipelines section.
Tenant learning config lives in intelligence_config.yaml → tenant_learning section.
Adding a new pipeline = add YAML entry. Zero Python changes.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.lib.response_normalizer import field_present, parse_json_like, split_expected_fields
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


def _get_learning_config() -> dict:
    """Get tenant_learning section from YAML."""
    config = get_intelligence_config()
    return config.get("tenant_learning", {})


def _get_pipeline_learning(name: str) -> dict:
    """Get per-pipeline learning settings (store/retrieve flags)."""
    lc = _get_learning_config()
    return lc.get("pipelines", {}).get(name, {})


async def run_pipeline(
    name: str,
    variables: dict[str, Any],
    tenant_id: int | None = None,
) -> dict | str | None:
    """
    Execute a config-driven LLM pipeline.

    Steps:
      1. Load pipeline config from YAML (prompt, drivers, timeout, etc.)
      2. Format the prompt template with input variables
      3. (If tenant_id) Retrieve past analyses from Qdrant for context
      4. Loop through driver chain with circuit breaker
      5. Parse output (json_repair for JSON, raw for text)
      6. Validate expected fields
      7. (If tenant_id) Store result in tenant's Qdrant collection (fire-and-forget)
      8. Return clean result or fallback_response

    Args:
        name: Pipeline name (key in intelligence_pipelines YAML section)
        variables: Dict of template variables (e.g., {"url": "...", "content": "..."})
        tenant_id: Tenant ID for scoped learning. None = no learning.

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

    # ─── Tenant context retrieval ───────────────────────
    if tenant_id:
        context = await _retrieve_tenant_context(name, prompt, tenant_id)
        if context:
            prompt = f"{prompt}\n\n{context}"

    # ─── Pipeline options ───────────────────────────────
    from app.services.intelligence.config_loader import get_global_driver_chain

    driver_chain = cfg.get("driver_chain") or list(get_global_driver_chain())
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
                data = parse_json_like(response.content)
                if data is None:
                    logger.warning(f"LLMPipeline[{name}]: {driver_name} returned invalid JSON")
                    continue

                # Validate expected fields
                if expected_fields and isinstance(data, dict):
                    req, opt = split_expected_fields(expected_fields)
                    required_missing = [f for f in req if not field_present(data, f)]
                    optional_missing = [f for f in opt if not field_present(data, f)]
                    if required_missing or optional_missing:
                        logger.warning(
                            f"LLMPipeline[{name}]: {driver_name} missing fields: "
                            f"required={required_missing}, optional={optional_missing}"
                        )
                        # Required fields are strict; optional fields are advisory.
                        if req and required_missing:
                            continue
                        total_expected = max(1, len(req) + len(opt))
                        if (len(required_missing) + len(optional_missing)) > (total_expected / 2):
                            continue  # Too many missing — try next driver

                logger.info(
                    f"LLMPipeline[{name}]: SUCCESS via {driver_name} "
                    f"({response.total_tokens} tokens)"
                )

                # ─── Store in tenant's Qdrant collection (fire-and-forget) ───
                if tenant_id:
                    asyncio.create_task(
                        _store_tenant_result(name, variables, data, tenant_id, driver_name)
                    )

                return data
            else:
                # Text mode — return raw content
                logger.info(
                    f"LLMPipeline[{name}]: SUCCESS via {driver_name} "
                    f"({response.total_tokens} tokens)"
                )

                # Store text results too
                if tenant_id:
                    asyncio.create_task(
                        _store_tenant_result(name, variables, response.content, tenant_id, driver_name)
                    )

                return response.content

        except Exception as e:
            logger.warning(f"LLMPipeline[{name}]: {driver_name} failed: {e}")
            continue

    # ─── All drivers failed ─────────────────────────────
    logger.error(f"LLMPipeline[{name}]: ALL drivers failed. Returning fallback.")
    return fallback


# ─── Tenant Learning: Qdrant Store & Retrieve ───────────────────


async def _store_tenant_result(
    pipeline_name: str,
    variables: dict,
    result: dict | str,
    tenant_id: int,
    driver_name: str = "unknown",
) -> None:
    """
    Store pipeline result in tenant-scoped Qdrant collection.

    Runs as fire-and-forget (asyncio.create_task) so it never
    blocks the main response. All config comes from YAML.
    """
    lc = _get_learning_config()
    if not lc.get("enabled", False):
        return

    pl = _get_pipeline_learning(pipeline_name)
    if not pl.get("store", False):
        return

    try:
        from app.services.vector.qdrant_store import (
            embed_texts,
            get_qdrant_client,
            stable_point_id,
            upsert_points,
        )

        client = get_qdrant_client()
        if not client:
            return

        prefix = lc.get("collection_prefix", "tenant")
        collection_name = f"{prefix}_{tenant_id}_intelligence"

        key_var = _extract_key_variable(variables)
        doc_id = hashlib.md5(
            f"{pipeline_name}:{key_var}".encode()
        ).hexdigest()[:16]

        if isinstance(result, dict):
            doc_text = f"{pipeline_name} | {key_var} | {json.dumps(result, ensure_ascii=False)[:3000]}"
        else:
            doc_text = f"{pipeline_name} | {key_var} | {str(result)[:3000]}"

        vecs = embed_texts([doc_text])
        if not vecs or not vecs[0]:
            return
        pid = stable_point_id(f"intel_{doc_id}")
        payload = {
            "document": doc_text,
            "pipeline": pipeline_name,
            "key": str(key_var)[:500],
            "tenant_id": str(tenant_id),
            "driver": driver_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        upsert_points(client, collection_name, [pid], [vecs[0]], [payload])
        logger.info(
            f"TenantLearning: stored {pipeline_name} for tenant {tenant_id} "
            f"(key={key_var[:50]})"
        )
    except Exception as e:
        # Fire-and-forget — never crash the main flow
        logger.warning(f"TenantLearning: store failed for {pipeline_name}: {e}")


async def _retrieve_tenant_context(
    pipeline_name: str,
    prompt: str,
    tenant_id: int,
) -> str | None:
    """
    Retrieve relevant past analyses from tenant's Qdrant collection.

    Returns formatted context string or None if nothing relevant found.
    All thresholds and templates come from YAML config.
    """
    lc = _get_learning_config()
    if not lc.get("enabled", False):
        return None

    pl = _get_pipeline_learning(pipeline_name)
    if not pl.get("retrieve", False):
        return None

    try:
        from app.services.vector.qdrant_store import (
            embed_texts,
            get_qdrant_client,
            qdrant_collection_count,
            search_points,
        )

        client = get_qdrant_client()
        if not client:
            return None

        prefix = lc.get("collection_prefix", "tenant")
        collection_name = f"{prefix}_{tenant_id}_intelligence"

        cnt = qdrant_collection_count(client, collection_name)
        if cnt == 0:
            return None

        max_chunks = lc.get("max_context_chunks", 3)
        min_relevance = lc.get("min_relevance", 0.4)

        qvecs = embed_texts([prompt[:500]])
        if not qvecs or not qvecs[0]:
            return None

        rows = search_points(
            client,
            collection_name,
            qvecs[0],
            limit=min(max_chunks * 2, cnt),
        )

        if not rows:
            return None

        chunks = []
        for row in rows:
            similarity = float(row.get("score", 0.0))
            if similarity >= min_relevance:
                pl = row.get("payload") or {}
                doc = pl.get("document", "")
                pipeline = pl.get("pipeline", "unknown")
                chunks.append(f"[{pipeline}] {doc}")

            if len(chunks) >= max_chunks:
                break

        if not chunks:
            return None

        context_text = "\n\n".join(chunks)

        # Use injection template from YAML
        template = lc.get(
            "context_injection_template",
            "### Tenant Context:\n{context}"
        )
        formatted = template.format(context=context_text)

        logger.info(
            f"TenantLearning: retrieved {len(chunks)} chunks for tenant {tenant_id} "
            f"(pipeline={pipeline_name})"
        )
        return formatted

    except Exception as e:
        logger.debug(f"TenantLearning: retrieve failed for {pipeline_name}: {e}")
        return None


def _extract_key_variable(variables: dict) -> str:
    """Extract the primary identifying variable from pipeline inputs.

    Priority: url > text (hashed) > first value
    """
    if "url" in variables:
        return str(variables["url"])
    if "text" in variables:
        text = str(variables["text"])
        if len(text) > 100:
            return hashlib.md5(text.encode()).hexdigest()[:12]
        return text
    # Fallback: first value
    for v in variables.values():
        return str(v)[:100]
    return "unknown"
