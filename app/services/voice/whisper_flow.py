"""
WhisperFlow — VoIP-Grade real-time voice conversation orchestrator.

Software Factory:
- Config-driven: all thresholds from YAML → realtime section
- State machine: SessionState enum (polymorphic transitions)
- Pipeline pattern: _transcribe → _execute_turn → _send_tts_sentence
- Handler map: msg_type → method (no if/elif chains)
- Reuses ChatEngine, Edge-TTS, SmartVoiceRouter, Brain memory

State Machine: IDLE → LISTENING → TRANSCRIBING → THINKING → SPEAKING → IDLE
"""

import asyncio
import base64
import enum
import logging
import time
from typing import Callable, Coroutine, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.voice.config import get_voice_config
from app.services.voice.realtime_stt import get_realtime_stt, normalize_language

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  STATE MACHINE
# ═══════════════════════════════════════════════════════════════

class SessionState(str, enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CLOSED = "closed"


# ═══════════════════════════════════════════════════════════════
#  SESSION — One per WebSocket connection
# ═══════════════════════════════════════════════════════════════

class WhisperFlowSession:
    """
    Manages one real-time voice conversation session.

    Pipeline per turn:
      audio buffer → STT (transcribe) → ChatEngine (think) →
      sentence-level TTS (speak) → client audio queue → resume listen
    """

    def __init__(
        self,
        session_id: str,
        tenant,
        send_fn: Callable[[dict], Coroutine],
        db_factory: Callable,
        voice_profile_id: Optional[int] = None,
        agent_type: str = "personal_assistant",
    ):
        self.session_id = session_id
        self.tenant = tenant
        self._send = send_fn
        self._db_factory = db_factory
        self.voice_profile_id = voice_profile_id
        self.agent_type = agent_type

        # ── Config (all from YAML — never hardcode) ──
        voice_cfg = get_voice_config()
        self._cfg = voice_cfg.get("realtime", {})
        self._client_cfg = {}  # Set via websocket setup
        groq_cfg = self._cfg.get("groq", {})
        self._chunk_ms = groq_cfg.get("chunk_duration_ms", 2500)
        self._idle_timeout = self._cfg.get("idle_timeout_s", 30)
        self._max_duration = self._cfg.get("max_session_duration_s", 300)

        # Streaming config
        streaming = self._cfg.get("streaming", {})
        self._sentence_terminals = set(streaming.get("sentence_terminals", [".", "!", "?", "।", "\n"]))
        self._min_sentence_len = streaming.get("min_sentence_length_chars", 15)
        self._chunk_size = streaming.get("audio_chunk_size_bytes", 32768)
        self._auto_reply = self._cfg.get("tts", {}).get("auto_reply", True)

        # ── State ──
        self.state = SessionState.IDLE
        self._audio_buffer = bytearray()
        self._last_audio = time.monotonic()
        self._session_start = time.monotonic()
        self._conversation_id: Optional[int] = None
        self._language: Optional[str] = None
        self._turn_count = 0
        self._active_task: Optional[asyncio.Task] = None

        # ── STT provider (factory-resolved) ──
        self._stt = get_realtime_stt()

        logger.info(
            f"🎙️ Session created: {session_id}, agent={agent_type}, "
            f"chunk_ms={self._chunk_ms}"
        )

    # ── Properties ──

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self._session_start) > self._max_duration

    @property
    def is_idle_timeout(self) -> bool:
        return (time.monotonic() - self._last_audio) > self._idle_timeout

    # ═══════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ═══════════════════════════════════════════════════════════

    async def start(self):
        """Initialize session — send config to client, create DB record."""
        self.state = SessionState.IDLE
        vad = self._cfg.get("vad", {})

        await self._send({
            "type": "session_start",
            "session_id": self.session_id,
            "provider": self._cfg.get("stt_provider", "groq"),
            "max_duration_s": self._max_duration,
            "vad": {
                "poll_interval_ms": vad.get("poll_interval_ms", 100),
                "silence_threshold": vad.get("silence_threshold", 0.015),
                "speech_threshold": vad.get("speech_threshold", 0.02),
                "silence_duration_ms": vad.get("silence_duration_ms", 1500),
                "min_speech_duration_ms": vad.get("min_speech_duration_ms", 500),
            },
        })

        if self._cfg.get("persist_conversations", True):
            await self._create_conversation()

        logger.info(f"✅ Session {self.session_id} started")

    async def close(self):
        """Graceful shutdown."""
        await self._cancel_turn()
        prev = self.state
        self.state = SessionState.CLOSED
        duration = time.monotonic() - self._session_start

        if prev != SessionState.CLOSED:
            try:
                await self._send({
                    "type": "session_end",
                    "session_id": self.session_id,
                    "turns": self._turn_count,
                    "duration_s": round(duration, 1),
                    "conversation_id": self._conversation_id,
                })
            except Exception:
                pass

        logger.info(f"🔇 Session {self.session_id} closed: {self._turn_count} turns, {duration:.1f}s")

    # ═══════════════════════════════════════════════════════════
    #  HANDLERS — Called from voice_ws.py message router
    # ═══════════════════════════════════════════════════════════

    async def handle_audio_chunk(self, audio_data: bytes, content_type: str = "audio/webm"):
        """Buffer audio chunk. Barge-in if AI is speaking.

        NOTE: We never auto-process here. Only process on explicit end_turn
        from VAD — this ensures the full WebM container (header + data) is intact.
        Auto-processing mid-stream would split the container, producing chunks
        without the WebM header that Groq rejects as 'Invalid file format'.
        """
        if self.state == SessionState.CLOSED:
            return

        self._last_audio = time.monotonic()
        self._audio_buffer.extend(audio_data)

        if self.state == SessionState.SPEAKING:
            logger.info(f"🚫 Barge-in during SPEAKING: {self.session_id}")
            await self._cancel_turn()

        if self.state != SessionState.LISTENING:
            self.state = SessionState.LISTENING

    async def handle_end_turn(self, content_type: str = "audio/webm"):
        """VAD silence → process whatever is buffered."""
        if self._audio_buffer:
            await self._process_buffer(content_type)

    async def handle_interrupt(self):
        """User interrupted AI speech."""
        logger.info(f"🚫 Interrupt: {self.session_id}")
        await self._cancel_turn()
        self.state = SessionState.IDLE
        await self._send({"type": "interrupted"})

    async def handle_config(self, data: dict):
        """Update session config mid-stream."""
        if "language" in data:
            self._language = data["language"]
        if "agent_type" in data:
            self.agent_type = data["agent_type"]
        if "voice_profile_id" in data:
            self.voice_profile_id = data.get("voice_profile_id")
            
        # Store all other arbitrary client config (like browser_tts)
        self._client_cfg.update(data)

        await self._send({
            "type": "config_updated",
            "language": self._language,
            "agent_type": self.agent_type,
            "browser_tts": self._client_cfg.get("browser_tts", False)
        })

    # ═══════════════════════════════════════════════════════════
    #  PIPELINE — Transcribe → Think → Speak (sentence-level)
    # ═══════════════════════════════════════════════════════════

    async def _process_buffer(self, content_type: str = "audio/webm"):
        """Step 1: Take buffer → transcribe → spawn turn task."""
        audio = bytes(self._audio_buffer)
        self._audio_buffer.clear()

        if len(audio) < 500:
            return  # noise

        await self._cancel_turn()

        # Transcribe
        self.state = SessionState.TRANSCRIBING
        await self._send({"type": "status", "stage": "transcribing"})

        transcription = await self._transcribe_with_fallback(audio, content_type)
        if not transcription:
            self.state = SessionState.IDLE
            return

        text = transcription["text"]
        language = normalize_language(transcription.get("language", "unknown"))
        duration = transcription["duration"]

        # Language auto-lock
        if language != "unknown":
            if self._language is None:
                logger.info(f"🌐 Language locked: '{language}' for {self.session_id}")
            self._language = language

        if not text:
            self.state = SessionState.IDLE
            return

        logger.info(f"🎤 Transcribed: '{text[:80]}' (lang={language}, dur={duration:.1f}s)")

        await self._send({
            "type": "transcript_final",
            "text": text,
            "language": language,
            "duration": duration,
        })

        # Spawn turn as background task (allows barge-in)
        self._active_task = asyncio.create_task(
            self._execute_turn(text, language, duration)
        )

    async def _transcribe_with_fallback(
        self, audio: bytes, content_type: str
    ) -> Optional[dict]:
        """STT with automatic provider fallback. Returns None on total failure."""
        try:
            return await self._stt.transcribe(audio, self._language, content_type)
        except Exception as e:
            logger.error(f"STT primary failed: {e}")

        # Fallback
        try:
            fallback = get_realtime_stt("openai_batch")
            result = await fallback.transcribe(audio, self._language, content_type)
            logger.info("STT fallback succeeded")
            return result
        except Exception as err:
            logger.error(f"STT fallback failed: {err}")
            await self._send({"type": "error", "message": f"Transcription failed: {err}"})
            return None

    async def _execute_turn(self, user_text: str, language: str, audio_dur: float):
        """Step 2: LLM → sentence-level streaming TTS."""
        t0 = time.monotonic()
        self.state = SessionState.THINKING
        await self._send({"type": "status", "stage": "thinking"})
        self._turn_count += 1

        full_resp = ""
        sentence_count = 0

        try:
            # Use separate DB session for Context Gathering (very short-lived)
            async with self._db_factory() as db:
                from app.services.chat.engine import ChatEngine

                stream = await ChatEngine.execute(
                    db, self.tenant,
                    prompt=user_text,
                    conversation_id=self._conversation_id,
                    voice_profile_id=self.voice_profile_id,
                    stream=True,
                )

            # DB session closed! Streaming and TTS take many seconds; we do not hold the DB lock.
            full_resp, sentence_count = await self._stream_and_speak(
                stream, language, t0
            )

            # Signal: all TTS sent
            await self._send({"type": "tts_complete", "total_sentences": sentence_count})

            # Persist in a SEPARATE DB session and shield it so barge-in cancellation doesn't corrupt the SQLAlchemy connection
            if self._cfg.get("persist_conversations", True) and self._conversation_id:
                async def _persist():
                    async with self._db_factory() as persist_db:
                        await self._persist_turn(persist_db, user_text, full_resp, language, audio_dur)
                await asyncio.shield(asyncio.create_task(_persist()))

        except asyncio.CancelledError:
            logger.info(f"Turn {self._turn_count} cancelled (barge-in)")
            return
        except Exception as e:
            logger.error(f"Turn failed: {e}", exc_info=True)
            await self._send({"type": "error", "message": f"AI processing failed: {str(e)[:200]}"})
            self.state = SessionState.IDLE
            return

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(f"✅ Turn {self._turn_count}: {elapsed:.0f}ms")

        self.state = SessionState.IDLE
        await self._send({
            "type": "turn_complete",
            "turn": self._turn_count,
            "conversation_id": self._conversation_id,
        })

    async def _stream_and_speak(
        self, stream, language: str, t0: float
    ) -> tuple[str, int]:
        """
        Stream LLM tokens → detect sentences → TTS each one.
        Returns (full_response_text, sentence_count).
        """
        tokens: list[str] = []
        sentence_buf: list[str] = []
        sentence_idx = 0
        tts_tasks: list[asyncio.Task] = []
        first = True

        async for token in stream:
            if first:
                logger.info(f"⚡ TTFT: {(time.monotonic() - t0) * 1000:.0f}ms")
                first = False

            tokens.append(token)
            sentence_buf.append(token)
            await self._send({"type": "ai_text", "content": token, "delta": True})

            # Sentence boundary?
            current = "".join(sentence_buf).strip()
            last_ch = token.strip()[-1] if token.strip() else ""

            if (
                last_ch in self._sentence_terminals
                and len(current) >= self._min_sentence_len
                and self._auto_reply
            ):
                sentence_idx = await self._spawn_tts(
                    current, language, sentence_idx, tts_tasks
                )
                sentence_buf = []

        # Remaining text
        remaining = "".join(sentence_buf).strip()
        if remaining and len(remaining) >= 3 and self._auto_reply:
            sentence_idx = await self._spawn_tts(
                remaining, language, sentence_idx, tts_tasks
            )

        # Wait for all TTS
        if tts_tasks:
            await asyncio.gather(*tts_tasks, return_exceptions=True)

        return "".join(tokens), sentence_idx

    async def _spawn_tts(
        self, text: str, language: str, idx: int, tasks: list
    ) -> int:
        """Spawn TTS for one sentence. Returns next index."""
        self.state = SessionState.SPEAKING
        if idx == 0:
            await self._send({"type": "status", "stage": "speaking"})

        task = asyncio.create_task(self._send_tts_sentence(text, language, idx))
        tasks.append(task)
        return idx + 1

    async def _send_tts_sentence(self, text: str, language: str, idx: int):
        """Generate TTS for one sentence and stream via WebSocket."""
        if self.state == SessionState.CLOSED:
            return

        from app.services.voice.voice_service import simplify_for_voice, text_to_speech
        from app.services.voice.router import get_voice_router

        clean = await simplify_for_voice(text)
        if not clean or len(clean) < 2:
            return

        # Fast-path for Browser TTS
        if self._client_cfg.get("browser_tts"):
            await self._send({
                "type": "ai_sentence",
                "text": clean,
                "sentence_index": idx,
                "language": language
            })
            logger.info(f"⚡ Browser TTS bypassed cloud generation for sentence {idx}.")
            return

        voice_id = get_voice_router().route(clean).get("voice_id")

        try:
            audio = await text_to_speech(clean, voice_id)
        except Exception as e:
            logger.warning(f"TTS failed for sentence {idx}: {e}")
            return

        if not audio:
            return

        # Chunk and send
        b64 = base64.b64encode(audio).decode("utf-8")
        total = (len(b64) + self._chunk_size - 1) // self._chunk_size

        for i in range(total):
            await self._send({
                "type": "ai_audio",
                "data": b64[i * self._chunk_size : (i + 1) * self._chunk_size],
                "chunk_index": i,
                "total_chunks": total,
                "format": "mp3",
                "is_final": i == total - 1,
                "sentence_index": idx,
            })

        logger.info(f"🔊 TTS[{idx}]: {len(audio)}B, {total} chunks, voice={voice_id}")

    # ═══════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════

    async def _cancel_turn(self):
        """Cancel active turn task (interrupt/barge-in)."""
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
            try:
                await self._active_task
            except (asyncio.CancelledError, Exception):
                pass
            self._active_task = None

    async def _create_conversation(self):
        """Create DB conversation record."""
        try:
            async with self._db_factory() as db:
                from app.models.ai_conversation import AiConversation
                conv = AiConversation(
                    tenant_id=self.tenant.id,
                    agent_type=self.agent_type,
                    metadata_={
                        "channel": "whisper_flow",
                        "session_id": self.session_id,
                        "input_type": "voice",
                    },
                )
                db.add(conv)
                await db.commit()
                await db.refresh(conv)
                self._conversation_id = conv.id
                logger.info(f"📝 Conversation {conv.id} created for {self.session_id}")
        except Exception as e:
            logger.warning(f"DB conversation create failed: {e}")

    async def _persist_turn(
        self, db: AsyncSession, user_text: str, ai_text: str,
        language: str, audio_dur: float,
    ):
        """Save turn to DB + teach brain (RAG memory)."""
        # 1. Persist task
        try:
            from app.models.ai_task import AiTask
            db.add(AiTask(
                tenant_id=self.tenant.id,
                conversation_id=self._conversation_id,
                agent_type=self.agent_type,
                prompt=user_text,
                result={"response": ai_text},
                status="completed",
                options={
                    "channel": "whisper_flow",
                    "language": language,
                    "audio_duration": audio_dur,
                    "turn": self._turn_count,
                    "input_type": "voice",
                },
            ))
            await db.commit()
        except Exception as e:
            logger.warning(f"Persist failed: {e}")

        # 2. Teach brain
        try:
            from app.services.intelligence.memory import get_memory
            memory = get_memory()
            await memory.remember(
                agent_type=self.agent_type,
                prompt=user_text,
                response=ai_text,
                quality_score=0.75,
            )
            memory.save_training_example(
                agent_type=self.agent_type,
                prompt=user_text,
                response=ai_text,
                quality=0.75,
            )
            logger.info(f"🧠 Brain learned turn {self._turn_count}: '{user_text[:50]}...'")
        except Exception as e:
            logger.warning(f"Brain learn failed (non-fatal): {e}")
