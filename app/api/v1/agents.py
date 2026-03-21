"""Agent API routes — listing + execution."""

from fastapi import APIRouter, Depends

from app.dependencies import CurrentTenant, DbSession
from app.models.ai_task import AiTask
from app.schemas.agent import AgentInfo, AgentRunRequest, BatchRunRequest
from app.schemas.chat import ChatResponse
from app.services.agents.hub import get_agent_hub

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_model=list[AgentInfo])
async def list_agents(tenant: CurrentTenant):
    """List all available agents with their capabilities."""
    hub = get_agent_hub()
    return hub.agent_info()


@router.post("/{agent_type}/run", response_model=ChatResponse)
async def run_agent(agent_type: str, body: AgentRunRequest, tenant: CurrentTenant, db: DbSession):
    """Run a single agent task."""
    hub = get_agent_hub()

    context = body.metadata or {}
    context["tenant_slug"] = tenant.slug

    task = AiTask(
        tenant_id=tenant.id,
        agent_type=agent_type,
        status="processing",
        prompt=body.prompt,
        external_user_id=body.external_user_id,
        options=body.options,
    )
    db.add(task)
    await db.flush()

    try:
        response = await hub.run(agent_type, body.prompt, context, db=db, **body.options or {})
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
            agent_type=agent_type,
            result={"content": response.content},
            tokens_used=response.total_tokens,
            driver_used=response.driver,
            model_used=response.model,
        )
    except Exception as e:
        task.status = "failed"
        task.error = str(e)
        raise


@router.post("/batch")
async def batch_run(body: BatchRunRequest, tenant: CurrentTenant, db: DbSession):
    """Run multiple agents in parallel on the same prompt."""
    hub = get_agent_hub()

    context = {}
    context["tenant_slug"] = tenant.slug

    results = await hub.batch(body.prompt, body.agent_types, context, db=db, **body.options or {})

    output = {}
    for agent_type, response in results.items():
        task = AiTask(
            tenant_id=tenant.id,
            agent_type=agent_type,
            status="completed",
            prompt=body.prompt,
            result={"content": response.content},
            tokens_used=response.total_tokens,
            driver_used=response.driver,
            model_used=response.model,
            agent_optimization_id=response.metadata.get("agent_optimization_id") if response.metadata else None
        )
        db.add(task)
        output[agent_type] = {
            "content": response.content,
            "tokens_used": response.total_tokens,
            "driver_used": response.driver,
            "model_used": response.model,
        }

    return {"results": output, "agents_run": list(output.keys())}
