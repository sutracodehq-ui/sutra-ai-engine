"""Chat routes — primary entry point for AI interactions."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import DbSession, get_current_tenant
from app.models.tenant import Tenant
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat.engine import ChatEngine

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat_completion(
    body: ChatRequest,
    db: DbSession,
    tenant: Annotated[Tenant, Depends(get_current_tenant)],
):
    """
    Synchronous chat completion.
    Uses the high-performance pipeline (Parallel Context + Caching + Smart Routing).
    """
    result = await ChatEngine.execute(
        db,
        tenant,
        prompt=body.prompt,
        conversation_id=body.conversation_id,
        voice_profile_id=body.voice_profile_id,
        voice_profile_name=body.voice_profile_name,
        stream=False,
        **body.options or {}
    )

    return ChatResponse(
        id=getattr(result, "id", "unknown"),
        content=result.content if hasattr(result, "content") else result.get("content", ""),
        driver=result.driver if hasattr(result, "driver") else result.get("driver"),
        model=result.model if hasattr(result, "model") else result.get("model"),
        usage=result.metadata if hasattr(result, "metadata") else result.get("usage", {}),
        metadata=result.metadata if hasattr(result, "metadata") else result.get("metadata", {}),
    )


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    db: DbSession,
    tenant: Annotated[Tenant, Depends(get_current_tenant)],
):
    """
    Streaming chat completion (SSE).
    Uses the high-performance pipeline with Zero-buffer passthrough.
    """

    async def _llm_generator():
        """Inner generator: streams tokens from the LLM."""
        full_response = []
        stream = await ChatEngine.execute(
            db,
            tenant,
            prompt=body.prompt,
            conversation_id=body.conversation_id,
            voice_profile_id=body.voice_profile_id,
            voice_profile_name=body.voice_profile_name,
            stream=True,
            **body.options or {}
        )

        async for chunk in stream:
            full_response.append(chunk)
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

        # Post-stream: extract suggestions
        complete_text = "".join(full_response)
        suggestions = []

        try:
            from app.services.intelligence.response_filter import get_response_filter
            engine = get_response_filter()
            filtered = engine.filter(complete_text)
            suggestions = filtered.suggestions
        except Exception:
            pass

        if suggestions:
            yield f"data: {json.dumps({'type': 'suggestions', 'items': suggestions})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    # Queue the request — emits thinking/calculating status, handles concurrency
    from app.services.intelligence.llm_queue import get_llm_queue
    queue = get_llm_queue()

    return StreamingResponse(
        queue.stream(_llm_generator, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
