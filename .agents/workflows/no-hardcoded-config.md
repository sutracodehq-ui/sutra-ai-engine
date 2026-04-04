---
description: Rule — Never hardcode configuration, always use YAML config files
---

# No Hardcoded Config Rule

## Rule
**NEVER hardcode configuration values, constants, or data mappings directly in Python files.**

All configurable data must live in YAML files that are loaded at runtime.

## What Must Be in YAML
- RSS feed URLs, API endpoints
- Stock symbols, crypto symbols
- Agent-to-tool mappings
- Agent chain configurations (review prompts, retry counts)
- Prompt templates for distillation
- Promotion thresholds and scoring constants
- Any list of items that might grow or change

## Where to Put Configs
| Config Type | File |
|-------------|------|
| Agent system prompts, capabilities, rules | `agent_config/<agent>.yaml` |
| RSS feeds and scan sources | `scanner_feeds.yaml` |
| Intelligence module configs (tools, chains, thresholds) | `intelligence_config.yaml` |
| Environment-specific values (URLs, keys, flags) | `.env` → `app/config.py` |

## Pattern
```python
# ❌ BAD — hardcoded in Python
STOCK_SYMBOLS = ["AAPL", "GOOGL", "MSFT"]

# ✅ GOOD — loaded from YAML
def _load_config():
    with open("intelligence_config.yaml") as f:
        return yaml.safe_load(f)
```

## Rationale
- Non-developers can modify configs without touching code
- Config changes don't require redeployment
- Single source of truth for all configurable values
- Easier to audit and review

## Intelligence Pipeline Rule (Critical)

Services that call `run_pipeline()` MUST be **schema-agnostic**:

```python
# ❌ BAD — hardcoded field names in Python
if result and "brand_identity" in result:
    return {"name": result["name"], "mission": result["mission"]}

# ✅ GOOD — pass through whatever YAML defines
if result and isinstance(result, dict) and len(result) > 0:
    return result
```

**What belongs in YAML (`intelligence_config.yaml → intelligence_pipelines`):**
- `system_prompt` — LLM system instruction
- `prompt_template` — prompt with `{variables}`
- `driver_chain` — failover order `[groq, gemini, anthropic, ollama]`
- `timeout_seconds` — route-level timeout
- `expected_fields` — validation fields (checked by `run_pipeline()`)
- `fallback_response` — returned on total failure
- `temperature`, `json_mode`, `max_content_chars`

**What belongs in Python services:**
- Input validation (min text length, URL format)
- Calling `run_pipeline(name, {variables})`
- Checking "is the result a non-empty dict/string?"
- That's it. Nothing else.

**To change any pipeline behavior:**
```
→ Edit intelligence_config.yaml
→ Restart container
→ Done. Zero code changes.
```
