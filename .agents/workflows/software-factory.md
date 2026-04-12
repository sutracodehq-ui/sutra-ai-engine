---
description: Software Factory Principles — VMware-inspired approach for the SutraCode AI Engine
---

# Software Factory Principles

Based on [VMware Software Factory](https://www.vmware.com/topics/software-factory). Every module, agent, and service MUST follow these principles.

---

## 1. Repeatability & Standardization
- Every agent follows the **same pattern**: YAML config → Python class → Hub registration
- System prompts, capabilities, rules, response schemas — all in YAML, never inline
- New agents are created by adding a `.yaml` + a 6-line Python class. Zero hub code changes.
- **Pattern**: `agent_config/<name>.yaml` + `services/agents/<name>.py` + auto-registered in `hub.py`

## 2. Automation First
- **CI/CD**: Every feature auto-tested before merge
- **Self-healing**: Circuit breakers, retry strategies, fallback drivers — automatic
- **Self-optimizing**: Prompt Engine auto-promotes winning prompts, no manual intervention
- **Self-learning**: HybridRouter auto-trains local model from cloud responses
- **Scheduled**: Celery Beat handles all recurring work (scans, optimizations, exports)

## 3. Modular Architecture
- Every intelligence module is a **standalone, swappable unit**:
  - `HybridRouter`, `QualityTracker`, `ToolRegistry`, `AgentChain`, `PromptEngine`, `WebScanner`
- Modules communicate via clean interfaces, not tight coupling
- **Enable/disable any module via config flags** — no code changes
- Config-driven: `intelligence_config.yaml`, `scanner_feeds.yaml`, `.env`

## 4. Quality Control at Every Layer
- **QualityGate**: Scores every response (1-10) on completeness, accuracy, format
- **AgentChain**: Review step catches bad output before it reaches the user
- **PromptEngine**: A/B tests prompts, auto-retires low performers
- **QualityTracker**: Per-agent rolling quality scores → adaptive routing

## 5. No Hardcoded Configuration
- **YAML for all configs**: `intelligence_config.yaml`, `scanner_feeds.yaml`, `agent_config/*.yaml`
- **Environment variables for secrets**: API keys, connection strings → `.env` → `config.py`
- **Never hardcode**: URLs, symbols, thresholds, mappings, templates, or agent behaviors in Python
- If it might change → it goes in YAML

## 6. Continuous Improvement (DevOps Loop)
```
User Request → Agent → QualityGate → Score
    ↓                                    ↓
 Feedback ←──────────────────── PromptEngine records
    ↓                                    ↓
 TrainingData → Fine-tune      Promote winning prompts
    ↓                           Retire losing prompts
 Better Local Model             Generate new candidates
```

## 7. Cloud-Native & Infrastructure-Agnostic
- Runs in Podman containers (local dev) or Kubernetes (production)
- Redis, PostgreSQL, ChromaDB, Ollama — all containerized services
- Horizontal scaling: add more Celery workers, not bigger machines
- Driver abstraction: swap Ollama ↔ Groq ↔ Gemini ↔ Anthropic without code changes

## 8. Operational Responsibility
- Every module logs its actions with structured logging
- Circuit breakers prevent cascade failures
- Rate limiters protect external APIs
- Token budgets prevent cost overruns
- WebScanner stores context for real-time grounding — agents don't hallucinate

## Checklist for Every New Feature
- [ ] Does it follow the standard pattern? (YAML config + Python class)
- [ ] Is all configuration in YAML, not Python?
- [ ] Can it be enabled/disabled via a config flag?
- [ ] Does it have error handling + logging?
- [ ] Does it have a quality feedback loop?
- [ ] Is it modular and independently testable?
- [ ] Does it auto-improve over time?
