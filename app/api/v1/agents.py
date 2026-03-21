"""Agent API routes — listing + execution.

Each registered agent gets its own explicit Swagger entry via dynamic route generation.
"""

from fastapi import APIRouter, Depends

from app.dependencies import CurrentTenant, DbSession
from app.models.ai_task import AiTask
from app.schemas.agent import AgentInfo, AgentRunRequest, BatchRunRequest
from app.schemas.chat import ChatResponse
from app.services.agents.hub import get_agent_hub

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_model=list[AgentInfo], summary="List All AI Agents")
async def list_agents(tenant: CurrentTenant):
    """
    Returns metadata for all 23 registered agents.

    Each agent entry includes:
    - **identifier**: The agent_type slug to use in `/agents/{type}/run`
    - **domain**: What the agent specializes in
    - **capabilities**: List of things the agent can do
    - **response_schema**: Expected output fields
    """
    hub = get_agent_hub()
    return hub.agent_info()


# ─── Core agent execution logic (shared by all routes) ──────────

async def _execute_agent(agent_type: str, body: AgentRunRequest, tenant, db):
    """Shared agent execution logic used by all agent-specific routes."""
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


# ─── Dynamic per-agent route generation ─────────────────────────
# Each agent gets its own explicit endpoint in Swagger, auto-generated
# from the hub registry. This follows the Software Factory pattern:
# add a YAML config + Python class → route appears automatically.

def _register_agent_routes():
    """Generate explicit Swagger endpoints for each registered agent."""
    hub = get_agent_hub()

    for agent_info in hub.agent_info():
        agent_type = agent_info["type"]
        agent_name = agent_info.get("name", agent_type.replace("_", " ").title())
        agent_desc = agent_info.get("description", "")
        capabilities = agent_info.get("capabilities", [])

        # Build a rich docstring from the YAML config
        caps_list = "\n    ".join(f"- {c}" for c in capabilities) if capabilities else "- General AI assistance"
        docstring = f"""
    Run the **{agent_name}** agent.

    *{agent_desc}*

    **Capabilities:**
    {caps_list}

    Send a prompt and receive AI-generated content tailored to this agent's specialization.
    """

        # Create a closure to capture agent_type
        def make_handler(at: str):
            async def handler(body: AgentRunRequest, tenant: CurrentTenant, db: DbSession):
                return await _execute_agent(at, body, tenant, db)
            handler.__doc__ = docstring
            handler.__name__ = f"run_{at}"
            return handler

        router.post(
            f"/{agent_type}/run",
            response_model=ChatResponse,
            summary=f"Run {agent_name}",
        )(make_handler(agent_type))


# Register all agent routes at import time
_register_agent_routes()


# ─── Batch endpoint ──────────────────────────────────────────────

@router.post("/batch", summary="Batch Run Multiple Agents")
async def batch_run(body: BatchRunRequest, tenant: CurrentTenant, db: DbSession):
    """
    Run multiple agents in parallel on the same prompt.

    Example: Send `{"prompt": "Launch our new feature", "agent_types": ["copywriter", "social", "email_campaign"]}`
    to get headline copy, a social post, AND an email draft — all in one request.

    Returns results keyed by agent type.
    """
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

