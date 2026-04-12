"""Celery worker jobs for Click Shield self-learning."""

from celery import shared_task
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_redis
from app.services.learning.click_learning import ClickLearningService
from app.models.tenant import Tenant
from sqlalchemy import select

@shared_task(name="click_shield.optimization_job")
def click_optimization_job():
    """
    Background job that runs the Click Shield learning cycle 
    for all active tenants.
    """
    return asyncio.run(_run_optimization())

async def _run_optimization():
    async for db in get_db():
        redis_client = await get_redis()
        learning_service = ClickLearningService(redis_client)
        
        # Fetch all active tenants
        stmt = select(Tenant).where(Tenant.is_active == True)
        result = await db.execute(stmt)
        tenants = result.scalars().all()
        
        for tenant in tenants:
            try:
                await learning_service.run_optimization_cycle(db, tenant.id)
            except Exception as e:
                # Log error but continue with other tenants
                print(f"ClickLearn: Failed for tenant {tenant.id}: {e}")
        
        break # Exit after one session
