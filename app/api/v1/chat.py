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
        id=result.get("id", "unknown"),
        content=result.get("content", ""),
        driver=result.get("driver"),
        model=result.get("model"),
        usage=result.get("usage", {}),
        metadata=result.get("metadata", {})
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
            # Yield as SSE data
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
