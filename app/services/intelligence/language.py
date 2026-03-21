"""
Language Service — detects input language and manages translations.

Enables 'Native Language' support where the AI automatically 
responds in the user's tongue without explicit instructions.
"""

import logging
from typing import TypedDict

from app.services.llm_service import get_llm_service
from app.config import get_settings

logger = logging.getLogger(__name__)


class LanguageResult(TypedDict):
    code: str      # ISO 639-1 code (en, es, hi, etc.)
    name: str      # English, Spanish, Hindi, etc.
    confidence: float


class LanguageService:
    """Service for multilingual support and detection."""

    @classmethod
    async def detect(cls, text: str) -> LanguageResult:
        """Detect the language of the provided text."""
        if not text or len(text) < 10:
            return {"code": "en", "name": "English", "confidence": 1.0}

        settings = get_settings()
        service = get_llm_service()
        
        prompt = f"""
Identify the language of the text below. 
Return ONLY a JSON object with:
{{
  "code": "iso_639_1_code",
  "name": "Full Language Name",
  "confidence": float (0.0 to 1.0)
}}

TEXT: "{text[:500]}"
""".strip()

        try:
            result = await service.complete(
                prompt=prompt,
                system_prompt="You are a language detector. Output raw JSON ONLY.",
                model=settings.ai_meta_prompt_model,
                temperature=0.0,
                json_mode=True
            )
            
            import json
            data = json.loads(result.get("content", "{}"))
            return {
                "code": data.get("code", "en"),
                "name": data.get("name", "English"),
                "confidence": data.get("confidence", 0.0)
            }
        except Exception as e:
            logger.error(f"Language Detection failed: {e}")
            return {"code": "en", "name": "English", "confidence": 0.0}
            
    @classmethod
    async def translate(cls, text: str, target_lang: str) -> str:
        """Translate text to the target language code."""
        service = get_llm_service()
        prompt = f"Translate the following to {target_lang}:\n\n{text}"
        
        result = await service.complete(
            prompt=prompt,
            system_prompt="You are a professional translator. Only return the translated text.",
            temperature=0.3
        )
        return result.get("content", text)
