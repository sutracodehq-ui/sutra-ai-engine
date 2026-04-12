"""
Voice WebSocket — Real-time Whisper Flow endpoint.

Software Factory: Config-driven, reuses existing services.

Security: Token-Based Direct Connect (Option B)
  1. Browser POSTs to Laravel → gets HMAC-signed token
  2. Browser connects here with token in query param
  3. Engine validates HMAC signature + expiry → extracts tenant_id
  4. API key never touches the browser

WebSocket: /v1/voice/ws/{session_id}?token=xxx

Protocol:
    Client → Server:
        {"type": "audio", "data": "<base64 audio>", "format": "webm"}
        {"type": "config", "language": "hi", "agent_type": "personal_assistant"}
        {"type": "end_turn"}
        {"type": "interrupt"}
        {"type": "ping"}

    Server → Client:
        {"type": "session_start", "session_id": "...", "provider": "groq"}
        {"type": "transcript_final", "text": "...", "language": "hi"}
        {"type": "status", "stage": "thinking" | "speaking" | "transcribing"}
        {"type": "ai_text", "content": "...", "delta": true}
        {"type": "ai_audio", "data": "<base64 mp3>", "chunk_index": 0}
        {"type": "turn_complete", "turn": 1}
        {"type": "session_end", "turns": 5, "duration_s": 120}
        {"type": "error", "message": "..."}
        {"type": "pong"}
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional

from app.config import get_settings
from app.services.voice.config import get_voice_config

logger = logging.getLogger(__name__)
router = APIRouter(tags=["voice-realtime"])


# ─── Connection Manager ────────────────────────────────────

class VoiceConnectionManager:
    """Track active WhisperFlow WebSocket sessions."""

    def __init__(self):
        self._sessions: dict[str, WebSocket] = {}

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self._sessions[session_id] = websocket
        logger.info(f"🎙️ Voice WS connected: {session_id} (active={self.active_count})")

    def disconnect(self, session_id: str):
        self._sessions.pop(session_id, None)
        logger.info(f"🔇 Voice WS disconnected: {session_id} (active={self.active_count})")

    async def send(self, session_id: str, data: dict):
        ws = self._sessions.get(session_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception as e:
                logger.warning(f"Send failed for {session_id}: {e}")


_manager = VoiceConnectionManager()


# ─── Token Validation (HMAC-signed from Laravel) ──────────

def _parse_token_payload(token: str) -> Optional[dict]:
    """
    Parse the unsigned payload from an HMAC token WITHOUT validating.
    Used to extract tenant_id for key lookup.

    Token format: base64(json_payload).base64(hmac_signature)
    """
    if not token:
        return None

    parts = token.split(".")
    if len(parts) != 2:
        return None

    try:
        payload_json = base64.b64decode(parts[0] + "==").decode("utf-8")
        return json.loads(payload_json)
    except Exception:
        return None


def _verify_hmac(token: str, secret: str) -> bool:
    """Verify HMAC-SHA256 signature of a token using the given secret."""
    parts = token.split(".")
    if len(parts) != 2:
        return False

    payload_b64, signature_b64 = parts

    expected_sig = hmac.new(
        secret.encode(),
        payload_b64.encode(),
        hashlib.sha256,
    ).digest()

    try:
        actual_sig = base64.b64decode(signature_b64 + "==")
    except Exception:
        return False

    return hmac.compare_digest(expected_sig, actual_sig)


async def _validate_token_with_tenant_key(token: str) -> tuple[Optional[dict], Optional["Tenant"]]:
    """
    Validate an HMAC-signed token using the tenant's own API key.

    Since the Laravel token doesn't include tenant_id (Laravel doesn't know it),
    the engine tries all active API keys to find the signing key.
    This is the standard webhook-style validation pattern (e.g. Stripe).

    Flow:
      1. Parse unsigned payload (for expiry check + metadata)
      2. Iterate active API keys → try HMAC validation with each
      3. On match → resolve tenant from the key's tenant_id
      4. Check expiry

    Returns (payload, tenant) if valid, (None, None) otherwise.
    """
    # Step 1: Parse unsigned payload
    payload = _parse_token_payload(token)
    if not payload:
        logger.warning("Token validation failed: cannot parse payload")
        return None, None

    # Step 2: Check expiry early (before DB lookup)
    exp = payload.get("exp", 0)
    if time.time() > exp:
        logger.warning(f"Token validation failed: expired (exp={exp})")
        return None, None

    # Step 3: Find the API key that validates this HMAC
    from app.dependencies import _get_engine, _session_factory

    settings = get_settings()
    _get_engine(settings)

    async with _session_factory() as db:
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        from app.models.tenant import Tenant
        from app.models.api_key import ApiKey

        # Get all active API keys with raw_key available
        key_result = await db.execute(
            select(ApiKey).where(
                ApiKey.is_active.is_(True),
                ApiKey.raw_key.isnot(None),
            )
        )
        api_keys = key_result.scalars().all()

        matched_key = None
        for api_key in api_keys:
            if _verify_hmac(token, api_key.raw_key):
                matched_key = api_key
                break

        if not matched_key:
            logger.warning("Token validation failed: no matching API key found")
            return None, None

        # Step 4: Resolve the tenant from the matched key
        result = await db.execute(
            select(Tenant).where(Tenant.id == matched_key.tenant_id, Tenant.is_active.is_(True))
        )
        tenant = result.scalar_one_or_none()

        if not tenant:
            logger.warning(f"Token validation failed: tenant {matched_key.tenant_id} not found/inactive")
            return None, None

        logger.info(
            f"🔐 Token valid: tenant={tenant.id}, user={payload.get('user_id')}, "
            f"key={matched_key.key_prefix}"
        )
        return payload, tenant


async def _resolve_tenant_from_key(api_key: str):
    """
    Fallback: Resolve a Tenant from raw API key (for backwards compat / dev).
    """
    from app.dependencies import _get_engine, _session_factory

    settings = get_settings()
    _get_engine(settings)

    async with _session_factory() as db:
        if api_key == settings.master_api_key:
            from sqlalchemy import select
            from app.models.tenant import Tenant
            result = await db.execute(
                select(Tenant).where(Tenant.is_active.is_(True)).order_by(Tenant.id).limit(1)
            )
            return result.scalar_one_or_none()

        from app.services.tenant_service import TenantService
        tenant, env, api_key_record = await TenantService.resolve_by_api_key(db, api_key)
        return tenant if tenant and tenant.is_active else None


# ─── WebSocket Endpoint ────────────────────────────────────

@router.websocket("/v1/voice/ws/{session_id}")
async def voice_realtime_ws(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = Query(default=None, description="HMAC-signed session token from Laravel"),
    api_key: Optional[str] = Query(default=None, description="(Deprecated) Raw API key"),
    voice_profile_id: Optional[int] = Query(default=None),
    agent_type: str = Query(default="personal_assistant"),
):
    """
    Real-time voice conversation via WebSocket.

    Auth priority:
    1. token param (HMAC-signed from Laravel) — production
    2. api_key param (raw key) — development / backwards compat
    """
    # Check if realtime is enabled
    voice_cfg = get_voice_config()
    realtime_cfg = voice_cfg.get("realtime", {})
    if not realtime_cfg.get("enabled", True):
        await websocket.accept()
        await websocket.close(code=1008, reason="Real-time voice is disabled")
        return

    # Check concurrent session limit
    max_sessions = realtime_cfg.get("max_concurrent_sessions", 50)
    if _manager.active_count >= max_sessions:
        await websocket.accept()
        await websocket.close(code=1013, reason="Too many concurrent voice sessions")
        return

    # ── Authentication ──────────────────────────────────
    tenant = None
    token_payload = None

    if token:
        # Option B: HMAC-signed token from Laravel (production)
        # Validates using the tenant's own API key (not master key)
        try:
            token_payload, tenant = await _validate_token_with_tenant_key(token)
        except Exception as e:
            logger.error(f"Token validation failed: {e}", exc_info=True)
            await websocket.accept()
            await websocket.close(code=1008, reason="Authentication error")
            return

        if not token_payload or not tenant:
            await websocket.accept()
            await websocket.close(code=1008, reason="Invalid or expired session token")
            return

        # Extract agent_type and voice_profile_id from token
        agent_type = token_payload.get("agent_type", agent_type)
        voice_profile_id = token_payload.get("voice_profile_id", voice_profile_id)

    elif api_key:
        # Fallback: Raw API key (development / backwards compat)
        logger.warning("⚠️ Using raw API key auth (deprecated) — switch to token-based auth")
        try:
            tenant = await _resolve_tenant_from_key(api_key)
        except Exception as e:
            logger.error(f"API key auth failed: {e}", exc_info=True)
            await websocket.accept()
            await websocket.close(code=1008, reason="Authentication error")
            return
    else:
        await websocket.accept()
        await websocket.close(code=1008, reason="Missing authentication (token or api_key required)")
        return

    if not tenant:
        await websocket.accept()
        await websocket.close(code=1008, reason="Invalid credentials")
        return

    # Accept and track connection
    await _manager.connect(session_id, websocket)

    # Create send function bound to this session
    async def send_fn(data: dict):
        await _manager.send(session_id, data)

    # DB session factory for the WhisperFlow session
    from app.dependencies import _session_factory

    # Get defaults from config (Software Factory)
    realtime_defaults = realtime_cfg.get("defaults", {})
    agent_type = agent_type or realtime_defaults.get("agent_type", "personal_assistant")

    # Create WhisperFlow session
    from app.services.voice.whisper_flow import WhisperFlowSession

    session = WhisperFlowSession(
        session_id=session_id,
        tenant=tenant,
        send_fn=send_fn,
        db_factory=_session_factory,
        voice_profile_id=voice_profile_id,
        agent_type=agent_type,
    )

    try:
        await session.start()
    except Exception as e:
        logger.error(f"Session start failed: {e}", exc_info=True)
        await send_fn({"type": "error", "message": f"Session start failed: {e}"})
        _manager.disconnect(session_id)
        return

    # Background task: monitor idle timeout + max duration
    async def _monitor_session():
        while session.state.value != "closed":
            await asyncio.sleep(5)
            if session.is_expired:
                await send_fn({"type": "error", "message": "Session time limit reached"})
                await session.close()
                try:
                    await websocket.close(code=1000, reason="Session expired")
                except Exception:
                    pass
                return
            if session.is_idle_timeout:
                await send_fn({"type": "error", "message": "Session idle timeout"})
                await session.close()
                try:
                    await websocket.close(code=1000, reason="Idle timeout")
                except Exception:
                    pass
                return

    monitor_task = asyncio.create_task(_monitor_session())

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await send_fn({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = data.get("type", "audio")

            if msg_type == "audio":
                audio_b64 = data.get("data", "")
                if not audio_b64:
                    continue

                try:
                    audio_bytes = base64.b64decode(audio_b64)
                except Exception:
                    await send_fn({"type": "error", "message": "Invalid base64 audio"})
                    continue

                audio_format = data.get("format", "audio/webm")
                if not audio_format.startswith("audio/"):
                    audio_format = f"audio/{audio_format}"

                await session.handle_audio_chunk(audio_bytes, audio_format)

            elif msg_type == "end_turn":
                await session.handle_end_turn(data.get("format", "audio/webm"))

            elif msg_type == "interrupt":
                await session.handle_interrupt()

            elif msg_type == "config":
                await session.handle_config(data)

            elif msg_type == "ping":
                await send_fn({"type": "pong"})

            else:
                await send_fn({"type": "error", "message": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info(f"🔇 Voice WS disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Voice WS error for {session_id}: {e}", exc_info=True)
    finally:
        monitor_task.cancel()
        try:
            await session.close()
        except Exception:
            pass
        _manager.disconnect(session_id)
