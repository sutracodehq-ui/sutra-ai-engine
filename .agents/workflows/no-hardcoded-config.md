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
