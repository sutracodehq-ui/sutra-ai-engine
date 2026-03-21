"""
Edit Diff Job — Celery task to analyze user edits.
"""

import logging
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.workers.celery_app import celery_app
from app.dependencies import get_db_context
from app.models.agent_feedback import AgentFeedback
from app.services.learning.edit_analyzer import EditAnalyzer

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.edit_diff_job.edit_diff_analyze")
async def edit_diff_analyze():
    """
    Periodic job that analyzes 'edited' feedback signals.
    """
    async with get_db_context() as db:
        # 1. Fetch recent feedback with edits
        stmt = (
            select(AgentFeedback)
            .where(AgentFeedback.signal == "edited")
            .where(AgentFeedback.user_edit.isnot(None))
            .order_by(AgentFeedback.created_at.desc())
            .limit(100)
        )
        result = await db.execute(stmt)
        all_feedback = result.scalars().all()
        
        # 2. Group by agent_type
        by_agent = {}
        for f in all_feedback:
            if f.agent_type not in by_agent:
                by_agent[f.agent_type] = []
            
            # The user_edit col is a JSON dict: { "field_name": { "original": "...", "edited": "..." } }
            # We flatten it for the analyzer
            for field, delta in f.user_edit.items():
                if isinstance(delta, dict) and "original" in delta and "edited" in delta:
                    by_agent[f.agent_type].append(delta)

        # 3. Analyze each agent
        for agent_type, edits in by_agent.items():
            if len(edits) < 5:  # Need at least a small batch
                continue
                
            logger.info(f"Learn Loop: Analyzing {len(edits)} edits for '{agent_type}'")
            rules = await EditAnalyzer.analyze_edits(agent_type, edits)
            
            if rules:
                # For now, we just log it. 
                # In the next step, we'll store these 'Learned Rules' to be used by OPRO.
                logger.info(f"Learn Loop: Discovered rules for '{agent_type}':\n{rules}")
                
                # TODO: Store in an 'AgentLearnedRules' table
