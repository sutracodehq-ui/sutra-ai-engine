"""
Cross-Teaching Job — Scheduled background worker for collective intelligence.

Runs daily via Celery Beat and executes the full ACI flywheel:
1. Cross-Teaching: Extract insights from top agents → teach alliance peers
2. Domain Evolution: Run domain-specific benchmarks & self-teaching
3. Skill Transfer: Create skill packs from top performers → distribute

This is the automated engine that makes agents continuously improve
through collective knowledge sharing.
"""

import asyncio
import logging

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name="app.workers.tasks.cross_teach_job.run_collective_intelligence",
    bind=True,
    max_retries=1,
    default_retry_delay=300,
)
def run_collective_intelligence(self):
    """
    Full Agent Collective Intelligence cycle.

    Scheduled daily via Celery Beat. Runs three phases:
    1. Cross-Teaching — agents share insights with alliance peers
    2. Domain Evolution — domain-specific benchmarks & self-teaching
    3. Skill Transfer — top performers' expertise distilled into packs
    """
    async def _run():
        results = {
            "cross_teaching": None,
            "domain_evolution": None,
            "skill_transfer": None,
        }

        # ── Phase 1: Cross-Teaching ──
        try:
            from app.services.intelligence.cross_teacher import get_cross_teacher
            teacher = get_cross_teacher()
            results["cross_teaching"] = await teacher.run_teaching_cycle()
            logger.info(
                f"ACI Phase 1 (Cross-Teaching): "
                f"insights={results['cross_teaching'].get('insights_extracted', 0)}, "
                f"teachings={results['cross_teaching'].get('teachings_delivered', 0)}"
            )
        except Exception as e:
            logger.error(f"ACI Phase 1 (Cross-Teaching) failed: {e}")
            results["cross_teaching"] = {"error": str(e)}

        # ── Phase 2: Domain Evolution ──
        try:
            from app.services.intelligence.domain_evolution import get_domain_evolution
            domain_engine = get_domain_evolution()
            results["domain_evolution"] = await domain_engine.evolve_all_domains()
            logger.info(
                f"ACI Phase 2 (Domain Evolution): "
                f"domains={len(results['domain_evolution'].get('domains', {}))}"
            )
        except Exception as e:
            logger.error(f"ACI Phase 2 (Domain Evolution) failed: {e}")
            results["domain_evolution"] = {"error": str(e)}

        # ── Phase 3: Skill Transfer ──
        try:
            from app.services.intelligence.skill_transfer import get_skill_transfer
            transfer = get_skill_transfer()
            results["skill_transfer"] = await transfer.run_transfer_cycle()
            logger.info(
                f"ACI Phase 3 (Skill Transfer): "
                f"packs={results['skill_transfer'].get('packs_created', 0)}, "
                f"applied={results['skill_transfer'].get('packs_applied', 0)}"
            )
        except Exception as e:
            logger.error(f"ACI Phase 3 (Skill Transfer) failed: {e}")
            results["skill_transfer"] = {"error": str(e)}

        logger.info("ACI: Full collective intelligence cycle complete ✓")
        return results

    # Run async in sync Celery context
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()


@celery_app.task(name="app.workers.tasks.cross_teach_job.run_cross_teaching_only")
def run_cross_teaching_only():
    """Run only the cross-teaching phase (lighter, can run more frequently)."""
    async def _run():
        from app.services.intelligence.cross_teacher import get_cross_teacher
        teacher = get_cross_teacher()
        return await teacher.run_teaching_cycle()

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()


@celery_app.task(name="app.workers.tasks.cross_teach_job.run_domain_evolution_only")
def run_domain_evolution_only(domain: str | None = None):
    """Run domain evolution for a specific domain or all domains."""
    async def _run():
        from app.services.intelligence.domain_evolution import get_domain_evolution
        engine = get_domain_evolution()
        if domain:
            return await engine.evolve_domain(domain)
        return await engine.evolve_all_domains()

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()


@celery_app.task(name="app.workers.tasks.cross_teach_job.run_skill_transfer_only")
def run_skill_transfer_only():
    """Run only the skill transfer phase."""
    async def _run():
        from app.services.intelligence.skill_transfer import get_skill_transfer
        transfer = get_skill_transfer()
        return await transfer.run_transfer_cycle()

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()
