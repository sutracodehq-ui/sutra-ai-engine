# Lesson 03: Streaming Operations

## Overview

Streaming is the primary output mode for user-facing interactions. The pipeline
ensures low-latency, properly formatted output even on resource-constrained VPS.

## Admission Control

Local models (ollama, bitnet, fast_local) share a semaphore-based admission gate:

```yaml
resilience:
  admission_control:
    max_local_streams: 2      # concurrent local streams
    max_queue_wait_s: 0.35    # max wait before falling to cloud
```

If the semaphore is full, the request immediately falls to the next driver in chain.

## VPS Load Profiles

Dynamic load management adapts to the Hostinger VPS / k3s environment:

```yaml
vps_admission:
  profiles:
    normal:    { max_local_streams: 2, max_queue_wait_s: 0.5 }
    high_load: { max_local_streams: 1, max_queue_wait_s: 0.3, trigger_active_streams: 2 }
    critical:  { max_local_streams: 0, max_queue_wait_s: 0.1, trigger_active_streams: 3 }
```

When active streams >= trigger threshold, the system automatically:
- Reduces local concurrency
- Shortens queue wait times
- In `critical`: disables local models entirely, routes to cloud

## First-Token Timeout

Every stream has a first-token deadline (default 10s). If the driver doesn't
produce output in time, the system immediately falls to the next driver.

After the first token, an inactivity timeout (default 8s) catches stalled streams.

## Stream Normalizer

`app/lib/stream_normalizer.py` buffers tiny chunks for proper formatting:

- **Markdown mode**: emits on newline/sentence boundaries
- **JSON mode**: emits on comma/brace boundaries (respects string escaping)
- Min emit: 28 chars, Max emit: 240 chars
- Always flushes at stream end

## Ollama Profiles

Ollama has two tuning profiles (from YAML):

- **low_latency** (streaming): `num_predict: 320, temperature: 0.2, top_k: 30`
- **quality** (non-streaming): `num_predict: 1024, temperature: 0.5, top_k: 45`

## Fallback Chain

Full stream fallback: primary driver -> tiny-model cascade -> driver chain -> EmergencyFallbackDriver.
The emergency driver generates safe offline text, never leaves users with a blank screen.
