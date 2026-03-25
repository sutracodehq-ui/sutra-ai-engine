"""Agent API routes — listing + execution + streaming.

Each registered agent gets its own explicit Swagger entry via dynamic route generation.
Streaming endpoints provide SSE (Server-Sent Events) for ChatGPT-style token-by-token responses.
"""

import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

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

    # Prevent keyword argument conflict for 'context' if it exists in options
    options = (body.options or {}).copy()
    if "context" in options:
        extra_context = options.pop("context")
        if isinstance(extra_context, dict):
            context.update(extra_context)

    try:
        response = await hub.run(agent_type, body.prompt, db=db, context=context, **options)
        task.status = "completed"
        task.result = {"content": response.content}
        task.tokens_used = response.total_tokens
        task.driver_used = response.driver
        task.model_used = response.model
        
        # Attribution for A/B testing
        if response.metadata and "agent_optimization_id" in response.metadata:
            task.agent_optimization_id = response.metadata["agent_optimization_id"]

        # Use Response Filtration Engine output (attached by BaseAgent._filter_response)
        filtered = (response.metadata or {}).get("filtered_result", {})
        result_data = filtered.get("data", {"content": response.content})
        suggestions = filtered.get("suggestions", [])

        task.result = result_data

        return ChatResponse(
            task_id=task.id,
            status="completed",
            agent_type=agent_type,
            result=result_data,
            suggestions=suggestions,
            tokens_used=response.total_tokens,
            driver_used=response.driver,
            model_used=response.model,
        )
    except Exception as e:
        task.status = "failed"
        task.error = str(e)
        raise


# ─── SSE Streaming execution logic ──────────────────────────────

async def _stream_agent(agent_type: str, body: AgentRunRequest, tenant, db, request: Request | None = None):
    """SSE streaming agent execution — yields tokens as they arrive."""
    hub = get_agent_hub()

    context = body.metadata or {}
    context["tenant_slug"] = tenant.slug

    # Create task record for tracking
    task = AiTask(
        tenant_id=tenant.id,
        agent_type=agent_type,
        status="streaming",
        prompt=body.prompt,
        external_user_id=body.external_user_id,
        options=body.options,
    )
    db.add(task)
    await db.flush()
    task_id = task.id

    # Prevent keyword argument conflict for 'context'
    options = (body.options or {}).copy()
    if "context" in options:
        extra_context = options.pop("context")
        if isinstance(extra_context, dict):
            context.update(extra_context)

    async def _llm_generator():
        """Inner generator: streams tokens from the LLM."""
        full_response = []
        try:
            async for token in hub.run_stream(agent_type, body.prompt, db=db, context=context, **options):
                full_response.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # Post-stream: extract suggestions
            complete_text = "".join(full_response)
            suggestions = []

            try:
                from app.services.intelligence.brain import get_brain
                brain = get_brain()
                filtered = brain.filter_response(complete_text)
                suggestions = filtered.suggestions
                result_data = filtered.data if filtered.parsed else {"content": complete_text}
            except Exception:
                result_data = {"content": complete_text}

            if suggestions:
                yield f"data: {json.dumps({'type': 'suggestions', 'items': suggestions})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'task_id': task_id, 'agent': agent_type})}\n\n"

            task.status = "completed"
            task.result = result_data
            await db.commit()

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'task_id': task_id, 'agent': agent_type})}\n\n"
            task.status = "failed"
            task.error = str(e)
            await db.commit()

    # Queue the request — emits thinking/calculating status, handles concurrency
    from app.services.intelligence.brain import get_brain
    brain = get_brain()

    return StreamingResponse(
        brain.queue.stream(_llm_generator, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Dynamic per-agent route generation ─────────────────────────
# Each agent gets its own explicit endpoint in Swagger, auto-generated
# from the hub registry. This follows the Software Factory pattern:
# add a YAML config + Python class → route appears automatically.

def _register_agent_routes():
    """Generate explicit Swagger endpoints for each registered agent."""
    hub = get_agent_hub()

    for agent_info in hub.agent_info():
        agent_type = agent_info["identifier"]
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

        stream_docstring = f"""
    Stream the **{agent_name}** agent (SSE).

    *{agent_desc}*

    Returns Server-Sent Events with tokens as they are generated.
    Format: `data: {{"token": "...", "agent": "{agent_type}"}}`
    Final event: `data: [DONE]`
    """

        # Create closures to capture agent_type
        def make_handler(at: str):
            async def handler(body: AgentRunRequest, tenant: CurrentTenant, db: DbSession):
                return await _execute_agent(at, body, tenant, db)
            handler.__doc__ = docstring
            handler.__name__ = f"run_{at}"
            return handler

        def make_stream_handler(at: str):
            async def handler(body: AgentRunRequest, request: Request, tenant: CurrentTenant, db: DbSession):
                return await _stream_agent(at, body, tenant, db, request)
            handler.__doc__ = stream_docstring
            handler.__name__ = f"stream_{at}"
            return handler

        # REST endpoint (existing)
        router.post(
            f"/{agent_type}/run",
            response_model=ChatResponse,
            summary=f"Run {agent_name}",
        )(make_handler(agent_type))

        # SSE streaming endpoint (new)
        router.post(
            f"/{agent_type}/stream",
            summary=f"Stream {agent_name} (SSE)",
        )(make_stream_handler(agent_type))


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

    # Prevent keyword argument conflict for 'context'
    options = (body.options or {}).copy()
    if "context" in options:
        extra_context = options.pop("context")
        if isinstance(extra_context, dict):
            context.update(extra_context)

    results = await hub.batch(body.prompt, body.agent_types, db=db, context=context, **options)

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

