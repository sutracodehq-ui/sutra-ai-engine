# Lesson 02: Quality Guardrails

## Overview

The Guardian (`app/services/intelligence/guardian.py`) is the unified safety, quality,
and resilience engine. It protects every request path.

## Quality Scoring

Every LLM response is scored across 4 weighted dimensions:

| Dimension    | Weight | What it measures                                    |
|-------------|--------|-----------------------------------------------------|
| format      | 0.35   | JSON field coverage (required vs optional fields)    |
| completeness| 0.30   | Word count thresholds (10/40/100+ words)            |
| length      | 0.15   | Goldilocks zone (50-2000 words = good)              |
| coherence   | 0.20   | Sentence-ending punctuation count                    |

Total score (0-10) determines: `passed = total >= threshold` (default 6).

## Circuit Breaker

Per-driver state machine: `CLOSED -> OPEN -> HALF_OPEN -> CLOSED`.

- 3 consecutive failures = OPEN (driver skipped for 60s)
- After cooldown, one test request (HALF_OPEN)
- If test succeeds: CLOSED. If fails: back to OPEN.

## Adaptive Routing Hints

Guardian tracks quality scores per agent_type in Redis rolling windows.
The `get_route_hint()` method returns:

- `fast_local` -- if avg quality >= 8.0 (local models handle this well)
- `direct_cloud` -- if avg quality <= 5.0 (skip local, go straight to cloud)
- `standard` -- default (try local first, escalate if needed)

## Latency Tracking

Every `record_quality()` call now also records driver latency.
This feeds the MOR scorer for smarter routing decisions.

## Other Protections

- **Content moderation**: OpenAI Moderation API
- **PII redaction**: regex patterns from YAML
- **Rate limiting**: Redis sliding window per tenant
- **Token budget**: monthly caps per tenant with WARN/BLOCK levels
- **Retry strategy**: exponential backoff with jitter
