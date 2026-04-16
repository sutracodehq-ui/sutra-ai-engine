# Lesson 01: Routing Algorithms

## Overview

The Brain (`app/services/intelligence/brain.py`) uses a multi-stage routing pipeline
to pick the best driver + model for every request. Everything is config-driven via
`intelligence_config.yaml`.

## Stage 1: Scout Assessment

Every request first passes through the **Scout LLM** (a tiny model like `llama-3.1-8b-instant`
on Groq) that classifies:

- **complexity** (1-10, bucketed into simple/moderate/complex)
- **intent** (question, generation, analysis, creative, code)
- **needs_web** (boolean)
- **modality** (text_chat, structured_json, image_gen, vision, voice_stt, voice_tts)

If Scout fails (timeout, bad JSON), the O(1) **heuristic fallback** uses signal words,
prompt length buckets, and agent tier from a pre-computed decision table.

## Stage 2: Driver Chain Selection

The `driver_chains` config maps `(language, complexity) -> [driver1, driver2, ...]`.
This chain is then reordered by:

1. **Route hint** (from Guardian's Redis quality history)
2. **Stream optimization** (fast_local first for short prompts)
3. **MOR ranking** (Multi-Objective Router)

## Stage 3: Multi-Objective Router (MOR)

MOR scores each driver candidate across 4 dimensions:

```
MOR_score = (0.40 * quality) + (0.25 * latency) + (0.20 * cost) + (0.15 * reliability)
```

- **Quality**: from Redis rolling average (0-10 scale, normalized to 0-1)
- **Latency**: ratio of target_ms / actual_avg_ms (closer to target = higher score)
- **Cost**: inverse of per-1k-token cost from YAML
- **Reliability**: circuit breaker state (closed=1.0, half_open=0.5, open=0.0)

## Stage 4: Model Selection

`_pick_model()` selects the model tier for the chosen driver. Scout confidence
can nudge the tier up/down (e.g., score=9 on a "simple" tier nudges to "moderate").

## Thompson Sampling Bandit

Prompt selection uses **Thompson Sampling** instead of random exploration.
Each prompt variant tracks successes/failures. The bandit samples from
`Beta(successes + 1, failures + 1)` and picks the highest sample.

## Key Config Sections

- `smart_router.multi_objective` -- MOR weights and cost map
- `smart_router.scout` -- Scout LLM config
- `smart_router.driver_chains` -- per-language/complexity chains
- `smart_router.model_tiers` -- per-driver/complexity model selection
- `prompt_engine` -- bandit algorithm and thresholds
