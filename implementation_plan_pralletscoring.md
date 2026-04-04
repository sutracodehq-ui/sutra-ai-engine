# Brain V3 — Self-Learning Collaborative Orchestrator

Evolution of the `Brain` from a static router into a **self-improving, multi-model consensus engine** with web-augmented knowledge.

## Software Factory Compliance Audit

Measured against the 8 VMware Software Factory principles in [software-factory.md](file:///Users/piyushprashant/Documents/personal-projects/sutracode-workspace/sutracode-ai-engine/.agents/workflows/software-factory.md):

| # | Principle | Current Plan Status | Gap |
|---|-----------|-------------------|-----|
| 1 | **Repeatability & Standardization** | ⚠️ Partial | New modules (Scout, Judge, Consensus) are described as methods inside `brain.py` but lack the standard YAML config + Python class pattern |
| 2 | **Automation First** | ⚠️ Partial | Learning loop mentioned but no Celery Beat scheduled task for periodic self-optimization cycles |
| 3 | **Modular Architecture** | ❌ Missing | Plan dumps everything into `brain.py` — Scout, Judge, Consensus, WebAugment should be **swappable modules** toggled via config flags |
| 4 | **Quality Control at Every Layer** | ⚠️ Partial | Consensus Judge is mentioned but no scoring rubric, no QualityGate integration, no per-model accuracy tracking |
| 5 | **No Hardcoded Configuration** | ❌ Missing | Plan hardcodes "complexity > 7" threshold, model lists, scraper triggers — ALL must be in `intelligence_config.yaml` |
| 6 | **Continuous Improvement Loop** | ⚠️ Partial | Learning loop exists conceptually but no feedback flow diagram, no promotion/retirement for model preferences |
| 7 | **Cloud-Native & Infra-Agnostic** | ✅ OK | Uses existing containerized services |
| 8 | **Operational Responsibility** | ❌ Missing | No logging strategy, no circuit breaker for new modules, no token budget for parallel execution |

## User Review Required

> [!IMPORTANT]
> **Cost Control**: Parallel execution doubles API token usage. A YAML-configurable `token_budget_per_request` cap is mandatory before enabling.

> [!WARNING]
> **Breaking Change**: The BitNet scorer (`scorer.enabled`) will be replaced by the Scout LLM scorer. Existing `score_to_complexity` YAML mappings will be preserved but the endpoint changes.

## Proposed Changes

### Component 1: Scout Scorer (replaces BitNet)

Replace the BitNet binary scorer with a config-driven LLM scout that returns structured complexity + intent.

#### [MODIFY] `intelligence_config.yaml` — New `scout` section
```yaml
smart_router:
  scout:
    enabled: true                          # Feature flag (Principle 3)
    driver: ollama                         # Which LLM to use for scouting
    model: qwen3:1.7b                     # Small, fast model
    timeout_ms: 2000                       # Fail fast → heuristic fallback
    max_prompt_chars: 300
    system_prompt: |
      You are a task complexity classifier. Given a user prompt, respond with JSON only:
      {"complexity": 1-10, "intent": "question|generation|analysis|creative|code", "needs_web": true/false}
    complexity_tiers:
      simple: [1, 3]                       # Range: 1-3 = simple
      moderate: [4, 6]                     # Range: 4-6 = moderate 
      complex: [7, 10]                     # Range: 7-10 = complex
```

#### [MODIFY] [brain.py](file:///Users/piyushprashant/Documents/personal-projects/sutracode-workspace/sutracode-ai-engine/app/services/intelligence/brain.py)
- Replace `_bitnet_score()` with `_scout_score()` — calls LLM via existing driver registry
- Falls back to `_heuristic_complexity()` on timeout (preserves Stage 2 fallback)
- Returns `{complexity, intent, needs_web}` instead of just an int

---

### Component 2: Parallel Consensus Executor

For complex tasks, run multiple models in parallel and use a Judge to pick/merge the best output.

#### [MODIFY] `intelligence_config.yaml` — New `consensus` section
```yaml
consensus:
  enabled: true                            # Feature flag (Principle 3)
  min_complexity_tier: complex             # Only trigger for "complex" tasks
  parallel_drivers: [groq, gemini]         # Which drivers to race
  judge_driver: groq                       # Which driver judges the outputs
  judge_model: llama-3.3-70b-versatile
  token_budget_per_request: 8000           # Max total tokens across all parallel calls
  judge_prompt: |
    You are a quality judge. You will receive two AI responses to the same prompt.
    Compare them on: accuracy, completeness, formatting, and usefulness.
    Respond with JSON: {"winner": "A" or "B", "confidence": 0.0-1.0, "merged_response": "..."}
    If both are good, merge the best parts into merged_response.
```

#### [MODIFY] [brain.py](file:///Users/piyushprashant/Documents/personal-projects/sutracode-workspace/sutracode-ai-engine/app/services/intelligence/brain.py)
- New `_execute_consensus()` method using `asyncio.gather()` for parallel calls
- New `_judge_responses()` method that calls the judge LLM
- Wrap with circuit breaker — if consensus fails, fall back to single-model execution
- Track per-model win rates in Memory for adaptive routing

---

### Component 3: Knowledge Augmentation (Web Scout)

Integrate `WebScraperService` into the reasoning path when the Scout detects `needs_web: true` or quality gates fail.

#### [MODIFY] `intelligence_config.yaml` — New `knowledge_augment` section
```yaml
knowledge_augment:
  enabled: true                            # Feature flag (Principle 3)
  trigger_on_scout_web: true               # Trigger when scout says needs_web
  trigger_on_quality_fail: true            # Trigger when quality gate fails
  max_scrape_pages: 3
  max_context_chars: 2000                  # Cap injected web context
  cache_ttl_hours: 24                      # Cache scraped results in Memory
  timeout_s: 10                            # Hard timeout for scraping
```

#### [MODIFY] [brain.py](file:///Users/piyushprashant/Documents/personal-projects/sutracode-workspace/sutracode-ai-engine/app/services/intelligence/brain.py)
- New `_augment_with_web()` helper — calls `WebScraperService`, extracts body text, injects as context
- Injected into the execution path between Scout and Driver call
- Results cached in ChromaDB via Memory service

---

### Component 4: Cross-Agent Learning (Peer Teaching)

Leverage the existing `teaching_alliances` YAML config to enable knowledge sharing.

#### [MODIFY] [memory.py](file:///Users/piyushprashant/Documents/personal-projects/sutracode-workspace/sutracode-ai-engine/app/services/intelligence/memory.py)
- New `search_alliance_traces()` method — queries ChromaDB for successful traces from *allied* agents
- Uses `alliances` config from YAML to scope the search
- Injects top-K traces as few-shot examples in the system prompt

#### [MODIFY] [brain.py](file:///Users/piyushprashant/Documents/personal-projects/sutracode-workspace/sutracode-ai-engine/app/services/intelligence/brain.py)
- In `execute()`, after scout scoring, inject alliance traces into system prompt
- New `_record_gold_trace()` — stores successful high-quality responses for future teaching

---

### Component 5: Self-Improvement Loop (Celery Beat)

#### [MODIFY] `intelligence_config.yaml` — New `self_improvement` section
```yaml
self_improvement:
  enabled: true
  cycle_interval_hours: 6                  # Run every 6 hours via Celery Beat
  min_traces_before_learning: 20           # Need N traces before analyzing patterns
  model_preference_decay_days: 30          # Forget old model preferences after N days
  promote_model_win_rate: 70.0             # Promote a model if win_rate > N%
  demote_model_win_rate: 30.0              # Demote a model if win_rate < N%
```

#### [MODIFY] Celery Beat schedule
- New periodic task `brain_self_improvement_cycle` that:
  1. Analyzes model win rates from consensus logs
  2. Updates `parallel_drivers` order based on performance
  3. Runs prompt optimization cycle (existing `run_optimization_cycle()`)
  4. Prunes stale teaching traces from ChromaDB

---

## Architecture Diagram

```
User Prompt
    │
    ▼
┌─────────────────────────────────────────────────┐
│  BRAIN.route()                                   │
│  ┌───────────┐    ┌────────────┐                │
│  │   Scout    │───▶│  Heuristic │ (fallback)     │
│  │ (LLM 1.7b)│    │  (O(1))    │                │
│  └─────┬─────┘    └────────────┘                │
│        │ {complexity, intent, needs_web}          │
│        ▼                                         │
│  ┌─────────────┐   YES                           │
│  │ needs_web?  │───────▶ WebScraperService       │
│  └──────┬──────┘         (inject context)        │
│         │                                        │
│         ▼                                        │
│  ┌──────────────┐                                │
│  │ complexity?  │                                │
│  └──┬───┬───┬───┘                                │
│     │   │   │                                    │
│  simple moderate complex                         │
│     │   │       │                                │
│     ▼   ▼       ▼                                │
│  Single  Single  Parallel Consensus              │
│  Driver  Driver  ┌─────┐ ┌─────┐                │
│     │      │     │Groq │ │Gemini│                │
│     │      │     └──┬──┘ └──┬──┘                │
│     │      │        └───┬───┘                    │
│     │      │            ▼                        │
│     │      │     ┌──────────┐                    │
│     │      │     │  Judge   │                    │
│     │      │     │(70B LLM) │                    │
│     │      │     └────┬─────┘                    │
│     │      │          │                          │
│     ▼      ▼          ▼                          │
│  ┌──────────────────────────┐                    │
│  │     QualityGate          │                    │
│  │  (score, validate)       │                    │
│  └─────────┬────────────────┘                    │
│            │                                     │
│     PASS ──┼── FAIL ──▶ retry with web context   │
│            │                                     │
│            ▼                                     │
│  ┌──────────────────────────┐                    │
│  │  Record Gold Trace       │ ──▶ ChromaDB       │
│  │  Update Model Win Rates  │ ──▶ Memory         │
│  │  Share to Alliance       │ ──▶ Peer Teaching  │
│  └──────────────────────────┘                    │
└─────────────────────────────────────────────────┘
```

## Software Factory Checklist

- [x] Every new module (Scout, Judge, Consensus, WebAugment) has a **YAML feature flag** — can be enabled/disabled without code changes
- [x] All thresholds, prompts, model names, timeout values are in `intelligence_config.yaml` — zero hardcoded values
- [x] Existing `teaching_alliances` YAML config is reused for peer learning — not a new concept
- [x] Circuit breakers wrap every new async call — cascade failures prevented
- [x] Token budget cap prevents cost overruns from parallel execution
- [x] Celery Beat task for periodic self-improvement — automation first
- [x] Structured logging at every decision point with rationale
- [x] Modular: Scout, Judge, WebAugment are independent helpers — not monolithic

## Open Questions

> [!IMPORTANT]
> 1. **Scout Model**: Should we use `qwen3:1.7b` (Ollama, free, ~200ms) or `llama-3.1-8b-instant` (Groq, free tier, ~100ms) for the Scout? Tradeoff: local-first vs. latency.
> 2. **Consensus Trigger**: Should consensus be opt-in per agent (via `agent_config/*.yaml`) or global for all complex tasks?
> 3. **BitNet Container**: Should we keep the BitNet container for potential future use or remove it entirely to reduce resource footprint?

## Verification Plan

### Automated Tests
- `pytest tests/test_scout_scorer.py` — Mock Ollama response, verify complexity tier mapping
- `pytest tests/test_consensus.py` — Mock two LLM outputs, verify Judge picks the correct winner
- `pytest tests/test_web_augment.py` — Mock WebScraperService, verify context injection into prompt
- `pytest tests/test_alliance_teaching.py` — Verify cross-agent trace retrieval from ChromaDB

### Integration Tests
- `curl` the stream endpoint with a complex prompt → verify logs show "Consensus triggered: Groq vs Gemini"
- `curl` with a prompt requiring current info → verify logs show "WebAugment: injected N chars from URL"
- Check Celery Beat logs after 6 hours → verify "Self-improvement cycle: promoted model X, demoted model Y"
