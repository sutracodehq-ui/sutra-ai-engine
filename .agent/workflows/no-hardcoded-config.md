---
description: Rule — Never hardcode configuration, always use YAML config files
---

# No Hardcoded Configuration — Agent Coding Rule

## Rule
**Never hardcode any configurable value inside Python engine files.** All values that might change between environments, deployments, or tuning sessions MUST live in `intelligence_config.yaml`.

## What Must Be In YAML
- **Model names** (e.g., `qwen2.5:3b`, `llama-3.3-70b-versatile`) → `fallback_models:` section
- **Model costs** (input/output per 1K tokens) → `budget.model_costs:` section
- **Timeouts** (HTTP, queue, moderation) → `timeouts:` section
- **Regex patterns** (PII, sentiment) → `safety.pii.patterns:` and `sentiment:` sections
- **Unicode ranges** (Indic script detection) → `smart_router.indic_ranges:` section
- **Keyword lists** (sentiment, signal words) → `sentiment:` and `smart_router:` sections
- **Thresholds** (quality, RAG, speculative) → respective config sections
- **Budget defaults** (monthly tokens, TTL) → `budget:` section

## Allowed Fallback Defaults
The **only** place a hardcoded value is acceptable is as a `default=` argument to `_cfg()` or `.get()` calls, as a safety net in case the YAML key is missing:
```python
# ✅ Good — YAML-first with safety default
model = _cfg("fallback_models", "local", default="qwen2.5:3b")

# ❌ Bad — hardcoded model name
model = "qwen2.5:3b"
```

## How to Add New Config
1. Add the value to the appropriate section in `intelligence_config.yaml`
2. Load it via `_cfg("section", "key", default=fallback)` in Brain/Memory
3. Load it via `_sec("section", {}).get("key", fallback)` in Guardian
4. For regex patterns: pre-compile in `__init__()`, store as `self._pattern`
5. For lookup tables: build in `_ensure_sets()`, store as module-level global

## Reference Sections in intelligence_config.yaml
```
smart_router:       # routing, Indic ranges, script labels, signal words
safety:             # PII patterns, sentiment words, moderation
resilience:         # circuit breaker, retry, rate limiter
quality:            # scoring weights, thresholds, tracking
budget:             # model costs, monthly limits, forecast
timeouts:           # all HTTP and queue timeouts
fallback_models:    # local, meta_optimizer
rag:                # auto-cut threshold, max chunks
speculative:        # draft driver/model, quality gate
swarm:              # decompose, synthesize, alliances
sentiment:          # positive/negative word lists
```
