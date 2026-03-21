"""
Chatbot Engine — Per-brand embeddable AI chatbot brain.

Software Factory Principle: Polymorphic + Self-Learning.

Each brand gets its own personal AI agent that:
1. Answers using brand-specific knowledge base
2. Auto-detects language and responds accordingly
3. Supports text chat + WebRTC voice
4. Escalates unknown queries to brand owner via WhatsApp
5. Learns from owner answers for future use

Architecture:
    Customer → Chat/Voice → ChatbotEngine(brand_id)
        → Brand Knowledge lookup
            → FOUND: respond instantly
            → NOT FOUND: respond with "let me check" + escalate to owner
                → Owner replies on WhatsApp
                    → Learn answer → reply to customer
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

import yaml

from app.config import get_settings

logger = logging.getLogger(__name__)


def _load_chatbot_config() -> dict:
    """Load chatbot config from YAML."""
    config_path = Path("intelligence_config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    return config.get("chatbot", {})


class ChatSession:
    """Tracks a single conversation session."""

    def __init__(self, session_id: str, brand_id: str, channel: str = "text"):
        self.session_id = session_id
        self.brand_id = brand_id
        self.channel = channel  # "text", "voice", "webrtc"
        self.history: list[dict] = []
        self.language: str | None = None
        self.visitor_id: str | None = None
        self.started_at = datetime.now(timezone.utc)
        self.pending_escalations: list[str] = []

    def add_message(self, role: str, content: str):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "brand_id": self.brand_id,
            "channel": self.channel,
            "language": self.language,
            "message_count": len(self.history),
            "started_at": self.started_at.isoformat(),
        }


class ChatbotEngine:
    """
    Per-brand chatbot brain.

    Flow:
    1. Customer sends message (text or voice)
    2. Check brand knowledge base for answer
    3. If found → respond via AI agent with brand context
    4. If NOT found (low confidence) → respond with holding msg + escalate
    5. When owner answers via WhatsApp → learn + reply to customer
    """

    def __init__(self):
        self._sessions: dict[str, ChatSession] = {}

    # ─── Session Management ─────────────────────────────────

    def get_or_create_session(
        self, session_id: str, brand_id: str, channel: str = "text",
    ) -> ChatSession:
        """Get existing session or create new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatSession(session_id, brand_id, channel)
            logger.info(f"Chatbot: new session {session_id} for brand {brand_id}")
        return self._sessions[session_id]

    # ─── Core Chat ──────────────────────────────────────────

    async def chat(
        self,
        session_id: str,
        brand_id: str,
        message: str,
        channel: str = "text",
        visitor_id: str | None = None,
        language: str | None = None,
        db=None,
    ) -> dict:
        """
        Process a chat message from a customer.

        Returns:
        - response: AI's response text
        - confidence: how confident the AI is (0-1)
        - escalated: whether the query was escalated to brand owner
        - session_id: for tracking
        """
        config = _load_chatbot_config()
        session = self.get_or_create_session(session_id, brand_id, channel)
        session.visitor_id = visitor_id
        if language:
            session.language = language

        # Record user message
        session.add_message("user", message)

        # 1. Check brand knowledge base
        from app.services.intelligence.brand_knowledge import get_brand_knowledge
        knowledge = get_brand_knowledge()

        knowledge_result = await knowledge.search(brand_id, message)
        has_knowledge = knowledge_result.get("found", False)
        confidence = knowledge_result.get("confidence", 0.0)

        confidence_threshold = config.get("confidence_threshold", 0.6)

        # 2. Generate response via AI agent
        from app.services.agents.hub import AiAgentHub
        hub = AiAgentHub()

        # Use chatbot_trainer agent with brand context
        agent_id = config.get("default_agent", "chatbot_trainer")
        agent = hub.get(agent_id)

        # Build context with brand knowledge
        context = {
            "brand_id": brand_id,
            "language": session.language or language,
            "channel": channel,
            "chatbot": True,
        }

        # Inject brand knowledge into prompt if available
        if has_knowledge and confidence >= confidence_threshold:
            brand_context = knowledge_result.get("context", "")
            enhanced_prompt = (
                f"Brand Context:\n{brand_context}\n\n"
                f"Customer Question: {message}\n\n"
                f"Respond helpfully using the brand context. "
                f"Be conversational and on-brand."
            )
        else:
            enhanced_prompt = message

        # Execute with conversation history
        response = await agent.execute_in_conversation(
            prompt=enhanced_prompt,
            history=session.history[:-1],  # Exclude current message (already in prompt)
            db=db,
            context=context,
        )

        response_text = response.content
        session.add_message("assistant", response_text)

        # 3. Escalate if low confidence
        escalated = False
        if not has_knowledge or confidence < confidence_threshold:
            from app.services.intelligence.escalation_manager import get_escalation_manager
            escalation = get_escalation_manager()

            escalated = await escalation.escalate(
                brand_id=brand_id,
                session_id=session_id,
                customer_message=message,
                ai_response=response_text,
                confidence=confidence,
            )

            if escalated:
                # Add a "checking" note to the response
                holding_messages = config.get("holding_messages", {})
                lang = session.language or "en"
                holding = holding_messages.get(lang, holding_messages.get(
                    "en", "Let me check on that for you. I'll get back shortly!"
                ))
                response_text = f"{response_text}\n\n{holding}"
                session.pending_escalations.append(message)

        result = {
            "session_id": session_id,
            "response": response_text,
            "confidence": round(confidence, 2),
            "escalated": escalated,
            "language": session.language,
            "channel": channel,
        }

        # 4. Log for analytics
        self._log_chat(session, message, response_text, confidence, escalated)

        return result

    # ─── Voice Chat (WebRTC) ────────────────────────────────

    async def voice_chat(
        self,
        session_id: str,
        brand_id: str,
        audio_path: str,
        db=None,
    ) -> dict:
        """
        Process a voice message via WebRTC.
        Transcribes → chats → synthesizes response audio.
        """
        # 1. Transcribe with VoIP engine (Faster-Whisper)
        from app.services.intelligence.voip_engine import get_voip_engine
        voip = get_voip_engine()

        transcription = await voip.transcribe(audio_path)
        caller_text = transcription["text"]
        detected_lang = transcription["language"]

        if not caller_text.strip():
            return {"status": "silence", "message": "No speech detected"}

        # 2. Chat (same pipeline as text)
        chat_result = await self.chat(
            session_id=session_id,
            brand_id=brand_id,
            message=caller_text,
            channel="webrtc",
            language=detected_lang,
            db=db,
        )

        # 3. Synthesize response audio
        audio_bytes = await voip.synthesize(
            text=chat_result["response"],
            language=detected_lang,
        )

        chat_result["audio_size"] = len(audio_bytes)
        chat_result["transcription"] = caller_text

        return chat_result

    # ─── Learn from Owner Answer ────────────────────────────

    async def learn_from_owner(
        self,
        brand_id: str,
        question: str,
        answer: str,
        session_id: str | None = None,
    ) -> dict:
        """
        When brand owner answers an escalated question,
        store it in knowledge base so AI knows next time.
        """
        # 1. Store in brand knowledge base
        from app.services.intelligence.brand_knowledge import get_brand_knowledge
        knowledge = get_brand_knowledge()

        await knowledge.learn(brand_id, question, answer)

        # 2. If session is still active, send the answer to customer
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.add_message("assistant", f"Update: {answer}")

        logger.info(f"Chatbot: learned new answer for brand {brand_id}")

        return {
            "status": "learned",
            "brand_id": brand_id,
            "question": question[:100],
            "answer": answer[:200],
        }

    # ─── Logging ────────────────────────────────────────────

    def _log_chat(
        self, session: ChatSession, message: str,
        response: str, confidence: float, escalated: bool,
    ):
        """Log chat interaction for analytics."""
        log_dir = Path("training/chat_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"chat_{session.brand_id}.jsonl"

        entry = {
            "session_id": session.session_id,
            "brand_id": session.brand_id,
            "channel": session.channel,
            "language": session.language,
            "message": message[:500],
            "response": response[:500],
            "confidence": confidence,
            "escalated": escalated,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Chatbot: logging failed: {e}")


# ─── Singleton ──────────────────────────────────────────────
_engine: ChatbotEngine | None = None


def get_chatbot_engine() -> ChatbotEngine:
    global _engine
    if _engine is None:
        _engine = ChatbotEngine()
    return _engine
