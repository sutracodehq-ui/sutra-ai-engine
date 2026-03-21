"""
Edit Analyzer — reverse-engineers user edits to find negative constraints.

If a user edits "Buy now! 🚀" to "Buy now.", the analyzer detects 
"Avoid emojis" as a stylistic preference.
"""

import logging
from typing import List

from app.services.llm_service import get_llm_service
from app.config import get_settings

logger = logging.getLogger(__name__)


class EditAnalyzer:
    """Service to analyze the 'delta' between AI output and User correction."""

    @classmethod
    async def analyze_edits(cls, agent_type: str, edits: List[dict]) -> str | None:
        """
        Analyze a batch of edits for a specific agent.
        Returns a list of 'Negative Constraints' or 'Stylistic Rules'.
        """
        if not edits:
            return None

        # Prepare the analysis prompt
        edits_text = ""
        for i, edit in enumerate(edits[:10]):  # Focus on the 10 most recent
            edits_text += f"\n--- Edit {i+1} ---\n"
            edits_text += f"ORIGINAL: {edit.get('original')}\n"
            edits_text += f"USER EDITED TO: {edit.get('edited')}\n"

        meta_prompt = f"""
I am analyzing user edits for the '{agent_type}' AI agent.
The user felt the AI's output wasn't perfect and manually corrected it.
Your task is to reverse-engineer WHY they made these changes.

{edits_text}

### Instructions:
1. Identify common patterns in the user's edits (e.g., "removed emojis", "shortened sentences", "made it more formal").
2. Translate these patterns into SPECIFIC INSTRUCTIONS for the AI.
3. Your output should be a bulleted list of "Negative Constraints" or "Style Rules" that should be added to the system prompt.
4. Keep it extremely concise.

RULES:
"""
        settings = get_settings()
        service = get_llm_service()
        
        result = await service.complete(
            prompt=meta_prompt,
            system_prompt="You are a linguistic analyst. You detect subtle shifts in tone and style between two versions of text.",
            model=settings.ai_meta_prompt_model,
            temperature=0.0
        )

        return result.get("content")
