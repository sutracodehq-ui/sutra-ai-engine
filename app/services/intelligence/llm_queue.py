"""
LLM Request Queue — software-level parallel inference with status events.

Provides:
1. Semaphore-based concurrency control (matches OLLAMA_NUM_PARALLEL)
2. Thinking/calculating SSE status events (no raw queue positions)
3. Client disconnect detection → cancel stale requests → free slots
4. Thread-safe singleton

SSE events emitted:
    {"type": "status", "stage": "thinking"}      — queued/waiting
    {"type": "status", "stage": "calculating"}   — slot acquired, LLM starting
    {"type": "token", "content": "..."}          — normal streaming (unchanged)
"""

import asyncio
import json
import logging
import threading
import time
from typing import AsyncGenerator, Callable, Awaitable

from starlette.requests import Request

from app.config import get_settings

logger = logging.getLogger(__name__)


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


class LlmQueue:
    """
    Async queue with semaphore-based concurrency control.

    Limits concurrent LLM calls to `max_parallel` (should match OLLAMA_NUM_PARALLEL).
    Overflow requests wait in FIFO order, emitting "thinking" status.
    When client disconnects, the request is cancelled and the slot freed.
    """

    def __init__(self, max_parallel: int | None = None):
        settings = get_settings()
        self._max_parallel = max_parallel or settings.llm_max_parallel
        self._semaphore = asyncio.Semaphore(self._max_parallel)
        self._active = 0
        self._waiting = 0
        logger.info(f"LlmQueue: initialized with {self._max_parallel} parallel slots")

    @property
    def stats(self) -> dict:
        return {
            "max_parallel": self._max_parallel,
            "active": self._active,
            "waiting": self._waiting,
        }

    async def stream(
        self,
        generator_fn: Callable[[], AsyncGenerator[str, None]],
        request: Request | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Queue a streaming LLM request.

        Yields SSE-formatted events:
        1. {"type": "status", "stage": "thinking"}   — while waiting for slot
        2. {"type": "status", "stage": "calculating"} — slot acquired, LLM starting
        3. ... tokens from generator_fn ...

        Args:
            generator_fn: Zero-arg callable that returns an async generator of SSE strings
            request: Starlette request for disconnect detection (optional)
        """
        self._waiting += 1

        # Emit "thinking" status while waiting for a slot
        yield _sse({"type": "status", "stage": "thinking"})

        # Wait for a slot — with periodic disconnect polling
        while not self._semaphore._value > 0:
            if request and await request.is_disconnected():
                self._waiting -= 1
                logger.info("LlmQueue: client disconnected while waiting in queue")
                return
            try:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=2.0)
                break
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                self._waiting -= 1
                return
        else:
            # Semaphore has a slot available — acquire immediately
            await self._semaphore.acquire()

        self._waiting -= 1
        self._active += 1
        cancelled = False

        try:
            # Slot acquired — emit "calculating" status
            yield _sse({"type": "status", "stage": "calculating"})

            # Stream tokens with batched disconnect detection (every 5 chunks)
            gen = generator_fn()
            chunk_count = 0
            try:
                async for chunk in gen:
                    chunk_count += 1
                    # Check disconnect every 5 chunks to reduce async overhead
                    if request and chunk_count % 5 == 0 and await request.is_disconnected():
                        logger.info("LlmQueue: client disconnected mid-stream, freeing slot")
                        cancelled = True
                        break
                    yield chunk
            finally:
                if cancelled:
                    await gen.aclose()

        except asyncio.CancelledError:
            logger.info("LlmQueue: request cancelled")
        except Exception as e:
            logger.error(f"LlmQueue: stream error: {e}")
            yield _sse({"type": "error", "message": str(e)})
        finally:
            self._active -= 1
            self._semaphore.release()


# ─── Singleton ──────────────────────────────────────────────────

_queue: LlmQueue | None = None
_queue_lock = threading.Lock()


def get_llm_queue() -> LlmQueue:
    """Get or create the singleton LlmQueue."""
    global _queue
    if _queue is None:
        with _queue_lock:
            if _queue is None:
                _queue = LlmQueue()
    return _queue
