# Lesson 05: VPS Operations & Deployment

## Overview

The SutraCode AI Engine runs on a Hostinger VPS with k3s (lightweight Kubernetes).
This lesson covers deployment, load management, and operational procedures.

## Architecture on k3s

```
k3s cluster (Hostinger VPS)
├── sutra-ai-api        (FastAPI app)
├── sutra-ai-ollama     (Ollama local inference)
├── sutra-ai-redis      (caching, rate limits, quality tracking)
├── sutra-ai-qdrant     (RAG vector store)
└── sutra-ai-postgres   (persistent storage)
```

## Resource Constraints

Typical Hostinger VPS: 4 vCPU, 8-16GB RAM.
Key limits:
- Ollama: 1-2 concurrent inference streams max
- RAM: each Ollama model ~2-4GB loaded
- Network: shared bandwidth, variable latency to cloud APIs

## VPS Load Profiles

The system automatically degrades under load:

| Profile    | Local Streams | Queue Wait | Behavior                    |
|-----------|---------------|------------|-----------------------------|
| normal    | 2             | 0.5s       | Local-first, standard chain |
| high_load | 1             | 0.3s       | Cloud-preferred             |
| critical  | 0             | 0.1s       | Cloud-only, no local        |

Profile selection is automatic based on active stream count.

## OSS-First Policy

The routing policy enforces open-source-first model selection:

```yaml
routing_policy:
  oss_first: true
  free_providers: [ollama, fast_local, bitnet, groq, gemini, nvidia, sarvam]
  paid_providers: [openai, anthropic]
  paid_escalation_only_on:
    - quality_gate_fail
    - circuit_breaker_open
    - latency_budget_exceeded
```

Paid providers are only used when free options fail quality/latency gates.

## Scaling Strategies

1. **Vertical**: Upgrade VPS tier (more RAM = larger Ollama models)
2. **Horizontal**: Add k3s worker nodes for API replicas
3. **Offload**: Use Groq/Gemini free tiers for overflow
4. **Model selection**: Smaller models (qwen3:1.7b) for simple tasks

## Monitoring Checklist

- `GET /health` -- basic liveness
- Redis `sutra:quality:{agent}` -- per-agent quality trends
- Redis `sutra:latency:{agent}:{driver}` -- driver latency trends
- Circuit breaker states via Guardian API
- Ollama `/api/tags` -- loaded models

## Rollout Gates

Before deploying changes, run the factory benchmark:

```bash
PYTHONPATH=. python scripts/benchmark_full_factory.py --all
```

All gates must pass (exit code 0) before production deployment.

## Emergency Procedures

1. **All models down**: EmergencyFallbackDriver activates automatically
2. **Ollama OOM**: Circuit breaker opens after 3 failures, routes to cloud
3. **Cloud API keys expired**: Falls through chain to next provider
4. **VPS overload**: Load profiles auto-engage, shed local work to cloud
