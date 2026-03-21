"""Chat API routes — single completion + streaming."""

import json
import logging

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from app.dependencies import CurrentTenant, DbSession
from app.models.ai_task import AiTask
from app.models.voice_profile import VoiceProfile
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.agents.hub import get_agent_hub
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, tenant: CurrentTenant, db: DbSession):
    """Single-turn agent completion (synchronous)."""
    hub = get_agent_hub()

    # Resolve voice profile if provided
    voice = None
    if body.voice_profile_id:
        result = await db.execute(
            select(VoiceProfile).where(
                VoiceProfile.id == body.voice_profile_id,
                VoiceProfile.tenant_id == tenant.id,
            )
        )
        voice = result.scalar_one_or_none()

    # Build context
    context = body.metadata or {}
    context["tenant_slug"] = tenant.slug

    # Create task record
    task = AiTask(
        tenant_id=tenant.id,
        agent_type=body.agent_type,
        status="processing",
        prompt=body.prompt,
        external_user_id=body.external_user_id,
        options=body.options,
    )
    db.add(task)
    await db.flush()

    try:
        # Execute agent
        response = await hub.run(body.agent_type, body.prompt, context, **body.options or {})

        # Update task with results
        task.status = "completed"
        task.result = {"content": response.content}
        task.tokens_used = response.total_tokens
        task.driver_used = response.driver
        task.model_used = response.model

        return ChatResponse(
            task_id=task.id,
            status="completed",
            agent_type=body.agent_type,
            result={"content": response.content},
            tokens_used=response.total_tokens,
            driver_used=response.driver,
            model_used=response.model,
        )
    except Exception as e:
        task.status = "failed"
        task.error = str(e)
        logger.error(f"Chat failed: {e}")
        raise


@router.post("/stream")
async def chat_stream(body: ChatRequest, tenant: CurrentTenant, db: DbSession):
    """SSE streaming agent completion."""
    hub = get_agent_hub()
    agent = hub.get(body.agent_type)
    context = body.metadata or {}
    context["tenant_slug"] = tenant.slug

    messages = agent.build_messages(body.prompt, context=context)

    from app.services.llm_service import get_llm_service
    llm = get_llm_service()

    async def event_generator():
        try:
            async for chunk in llm.stream(messages):
                yield {"event": "message", "data": json.dumps({"content": chunk})}
            yield {"event": "done", "data": json.dumps({"status": "completed"})}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_generator())
