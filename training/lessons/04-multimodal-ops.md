# Lesson 04: Multimodal Operations

## Overview

The Brain now detects request **modality** and routes to specialized execution paths.
This unifies text, JSON, image, voice, and vision under a single routing interface.

## Modality Detection

Two-stage detection:

1. **Scout LLM** classifies modality in its JSON response
2. **Heuristic fallback** based on agent_type and system prompt

Valid modalities:
- `text_chat` -- general text responses (default)
- `structured_json` -- strict JSON output expected
- `image_gen` -- image/video generation
- `vision` -- image understanding / analysis
- `voice_stt` -- speech-to-text
- `voice_tts` -- text-to-speech

## Agent-to-Modality Mapping

Heuristic rules (no Scout needed):

| Agent Type              | Modality        |
|------------------------|-----------------|
| image_generator        | image_gen       |
| video_generator        | image_gen       |
| social_clip_maker      | image_gen       |
| voip_support           | voice_stt       |
| cold_call_scripter     | voice_stt       |
| System prompt has JSON | structured_json |
| Everything else        | text_chat       |

## Execution Paths

Each modality routes to existing specialized services:

- **text_chat / structured_json**: Brain's standard `execute()` path
  - JSON mode: Ollama prioritized (deterministic `format: json`)
- **image_gen**: `ImageGenerationService` (Fal -> OpenAI DALL-E chain)
- **voice_stt**: Groq Whisper or Faster-Whisper via VoIP engine
- **voice_tts**: Edge-TTS (free) or ElevenLabs (premium)
- **vision**: `ai_vision_driver` / `ai_vision_model` from settings

## NVIDIA Small Models

NVIDIA provides specialized small models for different modalities:

| Use Case | Model                           | Size  |
|----------|---------------------------------|-------|
| Text     | mistral-nemo-minitron-8b-base   | 8B    |
| Text     | meta/llama-3.1-8b-instruct      | 8B    |
| Voice    | parakeet-tdt-0.6b-v2            | 0.6B  |

## Image Generation Config

```yaml
image_generation:
  driver_chain: [fal, openai]
  providers:
    fal: { model: "fal-ai/flux/schnell", cost: $0.003/image }
    openai: { model: "dall-e-3", cost: $0.04/image }
```

## Voice Config

```yaml
voice:
  default_provider: edge    # free, no API key
  stt_provider: groq        # Whisper via Groq (free tier)
```
