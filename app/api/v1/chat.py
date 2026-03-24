"""Chat routes — primary entry point for AI interactions."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Header
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
        metadata=result.metadata if hasattr(result, "metadata") else result.get("metadata", {})
    )


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    db: DbSession,
    tenant: Annotated[Tenant, Depends(get_current_tenant)],
):
    """
    Streaming chat completion (SSE).
    Uses the high-performance pipeline with Zero-buffer passthrough.
    """
    
    async def event_generator():
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

        # Phase 1: Collect all tokens
        async for chunk in stream:
            full_response.append(chunk)

        # Phase 2: Post-process through Response Filter
        complete_text = "".join(full_response)
        content_markdown = ""
        suggestions = []

        try:
            from app.services.intelligence.response_filter import get_response_filter
            engine = get_response_filter()
            filtered = engine.filter(complete_text)

            if filtered.parsed and isinstance(filtered.data, dict):
                content_markdown = (
                    filtered.data.get("response")
                    or filtered.data.get("content")
                    or filtered.data.get("advice")
                    or filtered.data.get("answer")
                    or ""
                )
                if isinstance(content_markdown, dict):
                    content_markdown = content_markdown.get("content") or json.dumps(content_markdown, indent=2)
            else:
                content_markdown = complete_text

            suggestions = filtered.suggestions
        except Exception:
            content_markdown = complete_text

        # Phase 3: Emit typed SSE events
        if content_markdown:
            yield f"data: {json.dumps({'type': 'token', 'content': content_markdown})}\n\n"
        if suggestions:
            yield f"data: {json.dumps({'type': 'suggestions', 'items': suggestions})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
