"""
Whisper Flow — VoIP-Grade real-time voice conversation orchestrator.

Software Factory:
- Config-driven: all thresholds from config/voice.yaml → realtime
- Reuses existing ChatEngine for agent execution (inherits all safety, routing, RAG)
- Reuses existing Edge-TTS + SmartVoiceRouter for natural-sounding responses
- Persists conversation turns to ai_conversations table

State Machine: IDLE → LISTENING → TRANSCRIBING → THINKING → SPEAKING → IDLE

VoIP Pipeline (sentence-level streaming):
  1. Audio chunks arrive via WebSocket → buffer
  2. VAD silence detection (client) or buffer full → Groq Whisper STT
  3. Transcription → route to ChatEngine (agent + RAG + safety)
  4. LLM streams tokens → detect sentence boundaries → TTS each sentence
  5. Client plays sentences back-to-back (audio queue)
  6. Last sentence plays → client auto-resumes recording
  7. Language auto-lock after first detection (fixes Hindi/Hinglish)
"""

import asyncio
import base64
import enum
import logging
import re
import time
from typing import Callable, Coroutine, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.tenant import Tenant
from app.services.voice.config import get_voice_config
from app.services.voice.realtime_stt import get_realtime_stt

logger = logging.getLogger(__name__)


class SessionState(str, enum.Enum):
    """WhisperFlow session state machine."""
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CLOSED = "closed"


class WhisperFlowSession:
    """
    Manages one real-time voice conversation session.

    Each WebSocket connection creates one instance.
    The session handles audio buffering, STT, agent routing, TTS, and persistence.
    """

    def __init__(
        self,
        session_id: str,
        tenant: Tenant,
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

        # Config from YAML
        voice_cfg = get_voice_config()
        self._realtime_cfg = voice_cfg.get("realtime", {})
        groq_cfg = self._realtime_cfg.get("groq", {})
        self._chunk_duration_ms = groq_cfg.get("chunk_duration_ms", 2500)
        self._max_silence_ms = groq_cfg.get("max_silence_ms", 1500)
        self._idle_timeout_s = self._realtime_cfg.get("idle_timeout_s", 30)
        self._max_duration_s = self._realtime_cfg.get("max_session_duration_s", 300)

        # State
        self.state = SessionState.IDLE
        self._audio_buffer = bytearray()
        self._last_audio_time = time.monotonic()
        self._session_start = time.monotonic()
        self._conversation_id: Optional[int] = None
        self._language: Optional[str] = None
        self._turn_count = 0

        # Active turn task (for cancellation on interrupt/barge-in)
        self._active_turn_task: Optional[asyncio.Task] = None

        # STT provider (config-driven)
        self._stt = get_realtime_stt()

        logger.info(
            f"🎙️ WhisperFlowSession created: session={session_id}, "
            f"agent={agent_type}, chunk_ms={self._chunk_duration_ms}, "
            f"realtime_enabled={self._realtime_cfg.get('enabled', '???')}"
        )

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self._session_start) > self._max_duration_s

    @property
    def is_idle_timeout(self) -> bool:
        return (time.monotonic() - self._last_audio_time) > self._idle_timeout_s

    async def start(self):
        """Initialize the session and send welcome event."""
        self.state = SessionState.IDLE
        provider = self._realtime_cfg.get("stt_provider", "groq")

        # VAD config from YAML — sent to client for config-driven silence detection
        vad_cfg = self._realtime_cfg.get("vad", {})

        await self._send({
            "type": "session_start",
            "session_id": self.session_id,
            "provider": provider,
            "max_duration_s": self._max_duration_s,
            "vad": {
                "poll_interval_ms": vad_cfg.get("poll_interval_ms", 100),
                "silence_threshold": vad_cfg.get("silence_threshold", 0.015),
                "speech_threshold": vad_cfg.get("speech_threshold", 0.02),
                "silence_duration_ms": vad_cfg.get("silence_duration_ms", 1500),
                "min_speech_duration_ms": vad_cfg.get("min_speech_duration_ms", 500),
            },
        })

        # Create conversation record for persistence
        if self._realtime_cfg.get("persist_conversations", True):
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
                    logger.info(f"📝 Created conversation {conv.id} for session {self.session_id}")
            except Exception as e:
                logger.warning(f"Failed to create conversation record: {e}")

        logger.info(f"✅ Session {self.session_id} started, waiting for audio...")

    async def handle_audio_chunk(self, audio_data: bytes, content_type: str = "audio/webm"):
        """
        Process incoming audio chunk from WebSocket.
        Buffers chunks until we have enough duration, then transcribes.
        """
        if self.state == SessionState.CLOSED:
            return

        self._last_audio_time = time.monotonic()
        self._audio_buffer.extend(audio_data)

        # If AI is currently speaking and user sends audio → barge-in
        if self.state == SessionState.SPEAKING:
            logger.info(f"🚫 Barge-in detected during SPEAKING for {self.session_id}")
            await self._cancel_active_turn()

        if self.state != SessionState.LISTENING:
            self.state = SessionState.LISTENING

        # Buffer threshold: ~4 bytes/ms at 32kbps WebM/Opus
        min_buffer_size = int(self._chunk_duration_ms * 4)
        if len(self._audio_buffer) >= min_buffer_size:
            await self._process_buffer(content_type)

    async def handle_end_turn(self, content_type: str = "audio/webm"):
        """Manual end-of-turn signal — process whatever is in the buffer."""
        if self._audio_buffer:
            await self._process_buffer(content_type)

    async def handle_interrupt(self):
        """User interrupted AI speech — cancel everything, go idle."""
        logger.info(f"🚫 Interrupt requested for session {self.session_id}")
        await self._cancel_active_turn()
        self.state = SessionState.IDLE
        await self._send({"type": "interrupted"})

    async def _cancel_active_turn(self):
        """Cancel the running turn task (LLM + TTS)."""
        if self._active_turn_task and not self._active_turn_task.done():
            self._active_turn_task.cancel()
            try:
                await self._active_turn_task
            except (asyncio.CancelledError, Exception):
                pass
            self._active_turn_task = None

    async def handle_config(self, data: dict):
        """Update session configuration mid-stream."""
        if "language" in data:
            self._language = data["language"]
        if "agent_type" in data:
            self.agent_type = data["agent_type"]
        if "voice_profile_id" in data:
            self.voice_profile_id = data.get("voice_profile_id")

        await self._send({
            "type": "config_updated",
            "language": self._language,
            "agent_type": self.agent_type,
        })

    async def close(self):
        """Close the session gracefully."""
        await self._cancel_active_turn()
        prev_state = self.state
        self.state = SessionState.CLOSED
        duration = time.monotonic() - self._session_start

        if prev_state != SessionState.CLOSED:
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

        logger.info(
            f"🔇 Session {self.session_id} closed: "
            f"{self._turn_count} turns, {duration:.1f}s"
        )

    # ─── Internal Pipeline ────────────────────────────────────

    async def _process_buffer(self, content_type: str = "audio/webm"):
        """
        Process the accumulated audio buffer:
        1. Transcribe with Groq Whisper
        2. Route to ChatEngine → TTS pipeline
        """
        audio_bytes = bytes(self._audio_buffer)
        self._audio_buffer.clear()

        if len(audio_bytes) < 500:
            # Too small — likely just noise
            return

        # If a turn is already running, cancel it (barge-in)
        if self._active_turn_task and not self._active_turn_task.done():
            await self._cancel_active_turn()

        # ── Step 1: Transcribe ──────────────────────────
        self.state = SessionState.TRANSCRIBING
        await self._send({"type": "status", "stage": "transcribing"})

        try:
            transcription = await self._stt.transcribe(
                audio_bytes,
                language_hint=self._language,
                content_type=content_type,
            )
        except Exception as e:
            logger.error(f"STT failed: {e}", exc_info=True)
            # Fallback to OpenAI Whisper
            try:
                fallback_stt = get_realtime_stt("openai_batch")
                transcription = await fallback_stt.transcribe(
                    audio_bytes,
                    language_hint=self._language,
                    content_type=content_type,
                )
                logger.info("STT fallback to OpenAI succeeded")
            except Exception as fallback_err:
                logger.error(f"STT fallback also failed: {fallback_err}")
                await self._send({"type": "error", "message": f"Transcription failed: {fallback_err}"})
                self.state = SessionState.IDLE
                return

        text = transcription.get("text", "").strip()
        language = transcription.get("language", "unknown")
        duration = transcription.get("duration", 0)

        # ── Language Auto-Lock ─────────────────────────────
        # After first successful detection, lock the language
        # so future turns don't misdetect (fixes Hindi→Greek issue)
        if language != "unknown" and self._language is None:
            self._language = language
            logger.info(f"🌐 Language locked to '{language}' for session {self.session_id}")
        elif language != "unknown":
            self._language = language

        if not text:
            logger.info(f"Empty transcription for session {self.session_id}")
            self.state = SessionState.IDLE
            return

        logger.info(f"🎤 Transcribed: '{text[:80]}' (lang={language}, dur={duration:.1f}s)")

        # Send transcription to client
        await self._send({
            "type": "transcript_final",
            "text": text,
            "language": language,
            "duration": duration,
        })

        # ── Step 2: Execute turn as background task ────
        # This allows new audio to arrive and trigger barge-in
        self._active_turn_task = asyncio.create_task(
            self._execute_turn(text, language, duration)
        )

    async def _execute_turn(self, user_text: str, language: str, audio_duration: float):
        """
        Execute one conversational turn: LLM → Sentence-Level Streaming TTS.

        VoIP-grade approach:
        - Stream LLM tokens
        - Detect sentence boundaries (. ! ? । newline)
        - Synthesize TTS for each sentence immediately
        - Client plays sentences back-to-back from audio queue
        - Much lower perceived latency vs waiting for full response
        """
        start_time = time.monotonic()

        self.state = SessionState.THINKING
        await self._send({"type": "status", "stage": "thinking"})

        self._turn_count += 1
        complete_text = ""

        # Sentence boundary config from YAML (Software Factory)
        streaming_cfg = self._realtime_cfg.get("streaming", {})
        sentence_terminals = set(streaming_cfg.get("sentence_terminals", [".", "!", "?", "।", "\n"]))
        min_sentence_len = streaming_cfg.get("min_sentence_length_chars", 15)
        tts_cfg = self._realtime_cfg.get("tts", {})
        auto_reply = tts_cfg.get("auto_reply", True)

        try:
            async with self._db_factory() as db:
                from app.services.chat.engine import ChatEngine

                # Stream the response token-by-token
                stream = await ChatEngine.execute(
                    db, self.tenant,
                    prompt=user_text,
                    conversation_id=self._conversation_id,
                    voice_profile_id=self.voice_profile_id,
                    stream=True,
                )

                full_response = []
                sentence_buffer = []
                sentence_index = 0
                first_token = True
                tts_tasks = []  # Track in-flight TTS tasks

                async for token in stream:
                    if first_token:
                        ttft = (time.monotonic() - start_time) * 1000
                        logger.info(f"⚡ [WhisperFlow] TTFT: {ttft:.0f}ms")
                        first_token = False

                    full_response.append(token)
                    sentence_buffer.append(token)

                    # Send each token to client for live text display
                    await self._send({"type": "ai_text", "content": token, "delta": True})

                    # ── Sentence boundary detection ──────────
                    current_sentence = "".join(sentence_buffer).strip()
                    last_char = token.strip()[-1] if token.strip() else ""

                    if (
                        last_char in sentence_terminals
                        and len(current_sentence) >= min_sentence_len
                        and auto_reply
                    ):
                        # Sentence complete → spawn TTS immediately (don't wait)
                        self.state = SessionState.SPEAKING
                        if sentence_index == 0:
                            await self._send({"type": "status", "stage": "speaking"})

                        # Spawn TTS as background task for parallelism
                        idx = sentence_index
                        text_to_speak = current_sentence
                        tts_task = asyncio.create_task(
                            self._generate_and_send_tts_sentence(text_to_speak, language, idx)
                        )
                        tts_tasks.append(tts_task)

                        sentence_buffer = []
                        sentence_index += 1

                complete_text = "".join(full_response)

                # ── Handle remaining sentence buffer ──────
                remaining = "".join(sentence_buffer).strip()
                if remaining and len(remaining) >= 3 and auto_reply:
                    self.state = SessionState.SPEAKING
                    if sentence_index == 0:
                        await self._send({"type": "status", "stage": "speaking"})

                    tts_task = asyncio.create_task(
                        self._generate_and_send_tts_sentence(remaining, language, sentence_index)
                    )
                    tts_tasks.append(tts_task)
                    sentence_index += 1

                # Wait for all TTS tasks to complete
                if tts_tasks:
                    await asyncio.gather(*tts_tasks, return_exceptions=True)

                # Signal client: all TTS sentences have been sent
                await self._send({"type": "tts_complete", "total_sentences": sentence_index})

                # Persist to database + teach brain
                if self._realtime_cfg.get("persist_conversations", True) and self._conversation_id:
                    await self._persist_turn(db, user_text, complete_text, language, audio_duration)

        except asyncio.CancelledError:
            logger.info(f"Turn {self._turn_count} cancelled (barge-in) for session {self.session_id}")
            return
        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            await self._send({"type": "error", "message": f"AI processing failed: {str(e)[:200]}"})
            self.state = SessionState.IDLE
            return

        # ── Turn complete ──────────────────────────────
        total_dur = (time.monotonic() - start_time) * 1000
        logger.info(f"✅ Turn {self._turn_count} complete: {total_dur:.0f}ms, {sentence_index} sentences TTS'd")

        self.state = SessionState.IDLE
        await self._send({
            "type": "turn_complete",
            "turn": self._turn_count,
            "conversation_id": self._conversation_id,
        })

    async def _generate_and_send_tts_sentence(
        self, text: str, language: str, sentence_index: int = 0
    ):
        """
        Generate TTS for a single sentence and send via WebSocket.

        Called per-sentence during streaming — enables back-to-back playback.
        Each sentence gets its own audio message with is_final=True so the
        client can enqueue and play them sequentially.
        """
        if self.state == SessionState.CLOSED:
            return

        from app.services.voice.voice_service import simplify_for_voice, text_to_speech

        # 1. Clean markdown/emojis for natural speech output
        clean_text = await simplify_for_voice(text)

        if not clean_text or len(clean_text) < 2:
            logger.debug(f"TTS: sentence {sentence_index} too short after cleaning, skipping")
            return

        # 2. Resolve voice from SmartVoiceRouter (language-aware)
        from app.services.voice.router import get_voice_router
        router = get_voice_router()
        routing = router.route(clean_text)
        voice_id = routing.get("voice_id")

        # 3. Generate TTS audio for this sentence
        try:
            audio_bytes = await text_to_speech(clean_text, voice_id)
        except Exception as e:
            logger.warning(f"TTS failed for sentence {sentence_index}: {e}")
            return

        if audio_bytes:
            # Send as base64 (single message per sentence — client enqueues)
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            # Chunk large audio for smooth WebSocket transport
            chunk_size = self._realtime_cfg.get("streaming", {}).get("audio_chunk_size_bytes", 32768)
            total_chunks = (len(audio_b64) + chunk_size - 1) // chunk_size

            for i in range(total_chunks):
                chunk = audio_b64[i * chunk_size : (i + 1) * chunk_size]
                await self._send({
                    "type": "ai_audio",
                    "data": chunk,
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "format": "mp3",
                    "is_final": i == total_chunks - 1,
                    "sentence_index": sentence_index,
                })

            logger.info(
                f"🔊 TTS sentence {sentence_index}: {len(audio_bytes)} bytes, "
                f"{total_chunks} chunks, voice={voice_id}, text='{clean_text[:40]}...'"
            )
        else:
            logger.warning(f"TTS returned empty audio for sentence {sentence_index}")

    async def _persist_turn(
        self,
        db: AsyncSession,
        user_text: str,
        ai_text: str,
        language: str,
        audio_duration: float,
    ):
        """Save the conversation turn to database AND teach the brain."""
        # 1. Persist to ai_tasks table (queries/analytics)
        try:
            from app.models.ai_task import AiTask

            task = AiTask(
                tenant_id=self.tenant.id,
                conversation_id=self._conversation_id,
                agent_type=self.agent_type,
                prompt=user_text,
                result={"response": ai_text},
                status="completed",
                options={
                    "channel": "whisper_flow",
                    "language": language,
                    "audio_duration": audio_duration,
                    "turn": self._turn_count,
                    "input_type": "voice",
                },
            )
            db.add(task)
            await db.commit()
        except Exception as e:
            logger.warning(f"Failed to persist turn: {e}")

        # 2. Teach the brain — store in ChromaDB for future recall (RAG)
        try:
            from app.services.intelligence.memory import get_memory
            memory = get_memory()

            # Agent memory: store Q&A pair so similar future queries get context
            await memory.remember(
                agent_type=self.agent_type,
                prompt=user_text,
                response=ai_text,
                quality_score=0.75,  # voice turns are slightly lower quality than curated data
            )

            # Training data: append JSONL for future LoRA fine-tuning
            memory.save_training_example(
                agent_type=self.agent_type,
                prompt=user_text,
                response=ai_text,
                quality=0.75,
            )

            logger.info(f"🧠 Brain learned from turn {self._turn_count}: '{user_text[:50]}...'")
        except Exception as e:
            logger.warning(f"Brain learning failed (non-fatal): {e}")

