"""Conversation API routes — multi-turn chat threads."""

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.dependencies import CurrentTenant, DbSession
from app.models.ai_conversation import AiConversation
from app.models.ai_task import AiTask
from app.schemas.chat import ChatResponse, ConversationMessageRequest
from app.services.agents.hub import get_agent_hub

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    tenant: CurrentTenant,
    db: DbSession,
    agent_type: str | None = None,
    external_user_id: str | None = None,
):
    """Create a new conversation thread. Defaults to personal assistant if no agent specified."""
    # Resolve default from YAML config
    if not agent_type:
        from app.services.intelligence.config_loader import get_intelligence_config
        pa_config = get_intelligence_config().get("personal_assistant_config", {})
        agent_type = pa_config.get("default_agent", "personal_assistant")

    hub = get_agent_hub()
    hub.get(agent_type)  # Validate agent exists

    conv = AiConversation(
        tenant_id=tenant.id,
        agent_type=agent_type,
        external_user_id=external_user_id,
    )
    db.add(conv)
    await db.flush()

    return {"id": conv.id, "agent_type": agent_type, "created_at": conv.created_at.isoformat()}


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: int, tenant: CurrentTenant, db: DbSession):
    """Get a conversation with all messages."""
    result = await db.execute(
        select(AiConversation).where(
            AiConversation.id == conversation_id,
            AiConversation.tenant_id == tenant.id,
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    messages = []
    for task in conv.tasks:
        messages.append({
            "id": task.id,
            "role": "user",
            "content": task.prompt,
            "created_at": task.created_at.isoformat(),
        })
        if task.result:
            messages.append({
                "id": task.id,
                "role": "assistant",
                "content": task.result.get("content", ""),
                "created_at": task.updated_at.isoformat(),
            })

    return {
        "id": conv.id,
        "agent_type": conv.agent_type,
        "messages": messages,
        "created_at": conv.created_at.isoformat(),
    }


@router.post("/{conversation_id}/messages", response_model=ChatResponse)
async def send_message(
    conversation_id: int,
    body: ConversationMessageRequest,
    tenant: CurrentTenant,
    db: DbSession,
):
    """Send a message in a conversation."""
    result = await db.execute(
        select(AiConversation).where(
            AiConversation.id == conversation_id,
            AiConversation.tenant_id == tenant.id,
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    # Build history from previous tasks
    history = []
    for task in conv.tasks:
        history.append({"role": "user", "content": task.prompt})
        if task.result:
            history.append({"role": "assistant", "content": task.result.get("content", "")})

    # Create task
    task = AiTask(
        tenant_id=tenant.id,
        conversation_id=conv.id,
        agent_type=conv.agent_type,
        status="processing",
        prompt=body.prompt,
    )
    db.add(task)
    await db.flush()

    hub = get_agent_hub()
    context = {"tenant_slug": tenant.slug}

    try:
        response = await hub.run_in_conversation(conv.agent_type, body.prompt, history, context, db=db)
        task.status = "completed"
        task.result = {"content": response.content}
        task.tokens_used = response.total_tokens
        task.driver_used = response.driver
        task.model_used = response.model
        
        # Attribution for A/B testing
        if response.metadata and "agent_optimization_id" in response.metadata:
            task.agent_optimization_id = response.metadata["agent_optimization_id"]

        return ChatResponse(
            task_id=task.id,
            status="completed",
            agent_type=conv.agent_type,
            result={"content": response.content},
            tokens_used=response.total_tokens,
            driver_used=response.driver,
            model_used=response.model,
        )
    except Exception as e:
        task.status = "failed"
        task.error = str(e)
        raise
