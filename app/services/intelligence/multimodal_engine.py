"""
Multi-Modal Output Engine — Agents respond like humans.

Capabilities:
1. TEXT — Rich markdown with formatting
2. VOICE — Edge-TTS (Indian voices) for spoken responses
3. IMAGE — AI-generated visuals via Ollama vision models
4. VIDEO_SCRIPT — Structured scripts ready for video generation
5. STEPS — Step-by-step visual guides with numbered instructions
6. CHART_DATA — Structured JSON for frontend chart rendering

Flow:
    Agent response (text) → MultiModalEngine → {text, audio_base64, image_prompts, steps}
"""

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class OutputMode(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"
    VIDEO_SCRIPT = "video_script"
    STEPS = "steps"
    CHART = "chart"
    FULL = "full"  # All applicable modes


@dataclass
class MultiModalResponse:
    """Response with multiple output formats."""
    # Always present
    text: str
    agent_id: str
    # Optional based on requested modes
    voice_audio: Optional[str] = None       # Base64 audio
    voice_format: str = "mp3"
    image_prompts: list[str] = field(default_factory=list)
    video_script: Optional[dict] = None
    steps: list[dict] = field(default_factory=list)
    chart_data: Optional[dict] = None
    # Meta
    modes_generated: list[str] = field(default_factory=list)


class MultiModalEngine:
    """
    Transforms text agent responses into multi-modal outputs.
    
    Self-hosted stack:
    - Voice: Edge-TTS (free, 12 Indian voices)
    - Images: Ollama vision model descriptions
    - Video: Structured script format (for frontend renderers)
    - Steps: Auto-extract numbered instructions
    """

    # Indian voices for Edge-TTS
    VOICES = {
        "hi-male": "hi-IN-MadhurNeural",
        "hi-female": "hi-IN-SwaraNeural",
        "en-male": "en-IN-PrabhatNeural",
        "en-female": "en-IN-NeerjaNeural",
        "ta-male": "ta-IN-ValluvarNeural",
        "ta-female": "ta-IN-PallaviNeural",
        "te-male": "te-IN-MohanNeural",
        "te-female": "te-IN-ShrutiNeural",
        "bn-male": "bn-IN-BashkarNeural",
        "bn-female": "bn-IN-TanishaaNeural",
        "mr-male": "mr-IN-ManoharNeural",
        "mr-female": "mr-IN-AarohiNeural",
    }

    def __init__(self):
        """Initialize multi-modal engine."""
        pass

    async def generate(
        self,
        text: str,
        agent_id: str,
        modes: list[OutputMode] | None = None,
        voice: str = "en-female",
        language: str = "en",
    ) -> MultiModalResponse:
        """
        Generate multi-modal output from text.
        
        Args:
            text: Agent text response
            agent_id: Which agent generated this
            modes: Which output formats to generate [text, voice, image, steps, ...]
            voice: Voice key from VOICES map
            language: Primary language for TTS
        """
        modes = modes or [OutputMode.TEXT]
        response = MultiModalResponse(text=text, agent_id=agent_id)

        tasks = []

        if OutputMode.VOICE in modes or OutputMode.FULL in modes:
            tasks.append(self._generate_voice(text, voice, response))

        if OutputMode.IMAGE in modes or OutputMode.FULL in modes:
            tasks.append(self._generate_image_prompts(text, agent_id, response))

        if OutputMode.VIDEO_SCRIPT in modes or OutputMode.FULL in modes:
            tasks.append(self._generate_video_script(text, agent_id, response))

        if OutputMode.STEPS in modes or OutputMode.FULL in modes:
            tasks.append(self._extract_steps(text, response))

        if OutputMode.CHART in modes or OutputMode.FULL in modes:
            tasks.append(self._extract_chart_data(text, response))

        # Run all modalities in parallel
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        response.modes_generated = [m.value for m in modes]
        return response

    async def _generate_voice(self, text: str, voice_key: str, response: MultiModalResponse):
        """Generate speech audio using Edge-TTS."""
        try:
            import edge_tts

            voice_name = self.VOICES.get(voice_key, self.VOICES["en-female"])

            # Clean text for TTS (remove markdown, excess formatting)
            clean_text = self._clean_for_tts(text)

            communicate = edge_tts.Communicate(clean_text, voice_name)
            audio_bytes = b""

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]

            if audio_bytes:
                response.voice_audio = base64.b64encode(audio_bytes).decode()
                response.voice_format = "mp3"
                response.modes_generated.append("voice")
                logger.info(f"MultiModal: voice generated ({len(audio_bytes)} bytes)")

        except ImportError:
            logger.warning("MultiModal: edge-tts not installed, skipping voice")
        except Exception as e:
            logger.error(f"MultiModal: voice generation failed: {e}")

    async def _generate_image_prompts(self, text: str, agent_id: str, response: MultiModalResponse):
        """Generate image description prompts from text content."""
        try:
            # Domain-specific visual prompt generation
            domain_styles = {
                "diet_planner": "food photography style, Indian cuisine",
                "trip_planner": "travel photography, landscape, vibrant colors",
                "property_valuator": "architectural photography, real estate",
                "crop_advisor": "agricultural photography, Indian farming",
                "visual_designer": "modern design, clean UI/UX",
                "product_description_writer": "product photography, e-commerce",
            }

            style = domain_styles.get(agent_id, "professional, clean, informative")

            # Extract key concepts from text for image generation
            # Use first 200 chars as context
            summary = text[:200].replace('\n', ' ')
            prompts = [
                f"Create a professional visual for: {summary}. Style: {style}",
            ]

            response.image_prompts = prompts
            response.modes_generated.append("image")

        except Exception as e:
            logger.error(f"MultiModal: image prompt generation failed: {e}")

    async def _generate_video_script(self, text: str, agent_id: str, response: MultiModalResponse):
        """Generate structured video script from text."""
        try:
            # Split text into scenes
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            scenes = []

            for i, para in enumerate(paragraphs[:10]):  # Max 10 scenes
                scenes.append({
                    "scene_number": i + 1,
                    "narration": para[:300],
                    "visual_description": f"Visual supporting: {para[:100]}",
                    "duration_seconds": max(5, len(para.split()) // 3),  # ~3 words/sec
                    "transition": "fade" if i > 0 else "none",
                })

            response.video_script = {
                "title": f"Video: {text[:50]}...",
                "total_scenes": len(scenes),
                "estimated_duration": sum(s["duration_seconds"] for s in scenes),
                "scenes": scenes,
                "music_mood": "informative" if agent_id in ["tax_planner", "gst_compliance"] else "upbeat",
            }
            response.modes_generated.append("video_script")

        except Exception as e:
            logger.error(f"MultiModal: video script generation failed: {e}")

    async def _extract_steps(self, text: str, response: MultiModalResponse):
        """Extract numbered steps from text for step-by-step guide."""
        try:
            import re

            steps = []

            # Pattern 1: numbered steps (1. 2. 3.)
            numbered = re.findall(r'(\d+)[.)\s]+(.+?)(?=\n\d+[.)\s]|\n\n|$)', text, re.DOTALL)
            if numbered:
                for num, content in numbered:
                    steps.append({
                        "step": int(num),
                        "title": content.strip().split('\n')[0][:100],
                        "detail": content.strip()[:500],
                        "icon": self._step_icon(content),
                    })

            # Pattern 2: bullet points
            if not steps:
                bullets = re.findall(r'[-•*]\s+(.+)', text)
                for i, bullet in enumerate(bullets, 1):
                    steps.append({
                        "step": i,
                        "title": bullet.strip()[:100],
                        "detail": bullet.strip(),
                        "icon": self._step_icon(bullet),
                    })

            if steps:
                response.steps = steps[:20]  # Max 20 steps
                response.modes_generated.append("steps")

        except Exception as e:
            logger.error(f"MultiModal: step extraction failed: {e}")

    async def _extract_chart_data(self, text: str, response: MultiModalResponse):
        """Extract numerical data for chart rendering."""
        try:
            import re

            # Look for patterns like "₹1,234" or "12%" or "500 calls"
            amounts = re.findall(r'₹[\d,]+(?:\.\d+)?', text)
            percentages = re.findall(r'(\d+(?:\.\d+)?)%', text)

            if amounts or percentages:
                response.chart_data = {
                    "has_financial_data": len(amounts) > 0,
                    "amounts": amounts[:10],
                    "percentages": [float(p) for p in percentages[:10]],
                    "chart_type": "bar" if amounts else "pie",
                }
                response.modes_generated.append("chart")

        except Exception as e:
            logger.error(f"MultiModal: chart data extraction failed: {e}")

    def _clean_for_tts(self, text: str) -> str:
        """Clean markdown for natural speech."""
        import re

        clean = text
        clean = re.sub(r'#{1,6}\s+', '', clean)       # Remove headers
        clean = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', clean)  # Bold/italic
        clean = re.sub(r'`(.+?)`', r'\1', clean)      # Code
        clean = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', clean)  # Links
        clean = re.sub(r'[-*•]\s+', '', clean)         # Bullets
        clean = re.sub(r'\|.+\|', '', clean)           # Tables
        clean = re.sub(r'\n{3,}', '\n\n', clean)       # Excess newlines
        clean = re.sub(r'[📋🌤️📰📈🎂💡🔴🟡🟢⚠️❌✅]', '', clean)  # Emojis
        return clean.strip()[:5000]  # TTS max 5K chars

    @staticmethod
    def _step_icon(content: str) -> str:
        """Pick an icon for a step based on content."""
        content_lower = content.lower()
        if any(w in content_lower for w in ["download", "install", "setup"]):
            return "⬇️"
        if any(w in content_lower for w in ["login", "sign", "register"]):
            return "🔑"
        if any(w in content_lower for w in ["upload", "submit", "send"]):
            return "📤"
        if any(w in content_lower for w in ["verify", "check", "confirm"]):
            return "✅"
        if any(w in content_lower for w in ["pay", "fee", "cost", "₹"]):
            return "💳"
        if any(w in content_lower for w in ["wait", "process", "days"]):
            return "⏳"
        return "📌"


# ─── Singleton ──────────────────────────────────────────────

_engine: MultiModalEngine | None = None

def get_multimodal_engine() -> MultiModalEngine:
    global _engine
    if _engine is None:
        _engine = MultiModalEngine()
    return _engine
