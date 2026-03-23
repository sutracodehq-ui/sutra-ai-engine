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

    # ─── Brand Config Resolution ────────────────────────────

    async def _get_brand_config(self, brand_id: str, db=None) -> dict | None:
        """
        Auto-resolve brand/product metadata from the tenant record.

        Reads from existing DB fields first (name, description),
        then overlays any explicit overrides from tenant.config JSON.
        No manual setup needed — works out of the box.
        """
        if not db:
            return None

        try:
            from sqlalchemy import select
            from app.models.tenant import Tenant

            # brand_id could be tenant slug or ID
            if brand_id.isdigit():
                result = await db.execute(
                    select(Tenant).where(Tenant.id == int(brand_id))
                )
            else:
                result = await db.execute(
                    select(Tenant).where(Tenant.slug == brand_id)
                )
            tenant = result.scalar_one_or_none()

            if not tenant:
                return None

            # Auto-resolve from existing tenant fields
            brand_config = {
                "brand_name": tenant.name,
            }
            if tenant.description:
                brand_config["brand_description"] = tenant.description

            # Overlay explicit overrides from config JSON (if any)
            if tenant.config and isinstance(tenant.config, dict):
                brand_config.update(tenant.config)

            return brand_config

        except Exception as e:
            logger.debug(f"Chatbot: brand config lookup skipped: {e}")

        return None

    # ─── Intent-Based Smart Routing ─────────────────────────

    # Config-driven: action keywords → specialist agent.
    # The chatbot auto-detects when the user wants a specialist
    # (e.g., "generate a quiz" → quiz_generator, "create notes" → note_generator).
    # This is generic — works across all product domains.
    INTENT_ROUTES: list[dict] = [
        # EdTech
        {"keywords": ["quiz", "mcq", "question paper", "test paper", "question bank"],
         "agent": "quiz_generator", "action_words": ["generate", "create", "make", "build", "prepare"]},
        {"keywords": ["notes", "revision notes", "summary notes", "study notes"],
         "agent": "note_generator", "action_words": ["generate", "create", "make", "write", "prepare"]},
        {"keywords": ["flashcard", "flash card"],
         "agent": "flashcard_creator", "action_words": ["generate", "create", "make", "build"]},
        {"keywords": ["lecture plan", "lesson plan", "teaching plan", "class plan"],
         "agent": "lecture_planner", "action_words": ["generate", "create", "make", "plan", "design"]},
        {"keywords": ["key points", "important points", "main points"],
         "agent": "key_points_extractor", "action_words": ["extract", "list", "give", "find", "get"]},
        # Marketing
        {"keywords": ["social media post", "instagram post", "twitter post", "facebook post"],
         "agent": "social", "action_words": ["generate", "create", "write", "make", "draft"]},
        {"keywords": ["email campaign", "newsletter", "email"],
         "agent": "email_campaign", "action_words": ["generate", "create", "write", "draft"]},
        {"keywords": ["ad copy", "advertisement", "ad creative"],
         "agent": "ad_creative", "action_words": ["generate", "create", "write", "make", "design"]},
        {"keywords": ["seo", "meta title", "meta description", "keywords"],
         "agent": "seo", "action_words": ["analyze", "generate", "optimize", "create", "write"]},
        # Finance
        {"keywords": ["stock", "share", "equity"],
         "agent": "stock_analyzer", "action_words": ["analyze", "check", "review"]},
        # Health
        {"keywords": ["diet plan", "meal plan", "nutrition plan"],
         "agent": "diet_planner", "action_words": ["generate", "create", "make", "plan", "suggest"]},
        {"keywords": ["symptoms", "feeling sick", "health issue"],
         "agent": "symptom_triage", "action_words": ["check", "assess", "evaluate", "help"]},
    ]

    def _detect_specialist_agent(self, message: str, available_agents: list[str]) -> str | None:
        """
        Detect user intent and route to specialist agent.

        Uses keyword + action word matching — no LLM call needed.
        Returns the specialist agent identifier if a match is found, None otherwise.
        """
        msg_lower = message.lower()

        for route in self.INTENT_ROUTES:
            # Check if the agent is available in the hub
            if route["agent"] not in available_agents:
                continue

            # Check if any keyword matches
            keyword_match = any(kw in msg_lower for kw in route["keywords"])
            if not keyword_match:
                continue

            # Check if any action word is present (user wants to DO something, not just ask)
            action_match = any(aw in msg_lower for aw in route.get("action_words", []))
            if action_match:
                logger.info(f"Chatbot: intent detected → {route['agent']} for: {message[:80]}")
                return route["agent"]

        return None

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

        # ─── Smart Agent Resolution ───────────────────────
        # Priority: intent detection → brand config → chatbot config → fallback
        agent_id = config.get("default_agent", "chatbot_trainer")

        # Check if brand has a custom agent override
        brand_config = await self._get_brand_config(brand_id, db)
        if brand_config:
            agent_id = brand_config.get("chatbot_agent", agent_id)

        # ─── Smart Context Population ─────────────────────
        context = {
            "brand_id": brand_id,
            "language": session.language or language,
            "channel": channel,
            "chatbot": True,
        }

        # Inject brand metadata into context (from tenant record)
        if brand_config:
            for key in (
                "brand_name", "brand_description", "organization_name",
                "organization_description", "product_name", "product_info",
                "website_url", "website_summary", "custom_instructions",
            ):
                if brand_config.get(key):
                    context[key] = brand_config[key]

        # ─── Intent-Based Specialist Routing ──────────────
        # Detect if user wants a specialist action (generate quiz, create notes, etc.)
        # If detected, execute the specialist directly and return actual content.
        specialist_id = self._detect_specialist_agent(message, hub.available_agents())

        if specialist_id:
            # Execute specialist agent directly — get ACTUAL content (quiz, notes, etc.)
            try:
                specialist_response = await hub.run(
                    specialist_id, message, db=db, context=context
                )
                response_text = specialist_response.content
                # Boost confidence since we matched a specialist
                confidence = max(confidence, 0.85)
            except Exception as e:
                logger.warning(f"Chatbot: specialist {specialist_id} failed, falling back: {e}")
                specialist_id = None  # Fall through to default agent

        if not specialist_id:
            # ─── Default Conversational Mode ──────────────
            agent = hub.get(agent_id)

            # Build enhanced prompt with brand knowledge
            prompt_parts = []
            if has_knowledge:
                brand_context = knowledge_result.get("context", "")
                if brand_context:
                    prompt_parts.append(f"Relevant Brand Knowledge:\n{brand_context}")

            prompt_parts.append(f"Customer Question: {message}")
            prompt_parts.append(
                "Respond helpfully and conversationally using whatever brand context you have. "
                "Be specific to this brand — never give generic advice."
            )

            enhanced_prompt = "\n\n".join(prompt_parts)

            response = await agent.execute_in_conversation(
                prompt=enhanced_prompt,
                history=session.history[:-1],
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
