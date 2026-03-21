"""
Celery Application — task queue + scheduled jobs.

Software Factory: cron jobs are defined declaratively in beat_schedule.
Adding a new learning pipeline = one entry + one task function.
"""

from celery import Celery
from celery.schedules import crontab

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "sutra_ai",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# ─── Beat Schedule (Software Factory: declarative cron) ──────────

celery_app.conf.beat_schedule = {
    # Daily 2 AM — OPRO: LLM analyzes accepted vs rejected → improved instructions
    "meta-prompt-optimize": {
        "task": "app.workers.meta_prompt_job.meta_prompt_optimize",
        "schedule": crontab(hour=2, minute=0),
    },
    # Daily 3 AM — TextGrad: Reverse-engineers user edit patterns
    "edit-diff-analyze": {
        "task": "app.workers.edit_diff_job.edit_diff_analyze",
        "schedule": crontab(hour=3, minute=0),
    },
    # Daily 3:30 AM — A/B testing: Scores prompt variants, promotes winners
    "prompt-evolution": {
        "task": "app.workers.evolution_job.run_prompt_evolution",
        "schedule": crontab(hour=3, minute=30),
    },
    # Daily 4 AM — Persist feedback data to R2
    "training-data-sync": {
        "task": "app.workers.training_sync_job.training_data_sync",
        "schedule": crontab(hour=4, minute=0),
    },
    # Weekly Monday 5 AM — Regenerate fine-tuned Ollama Modelfiles
    "ollama-fine-tune": {
        "task": "app.workers.ollama_finetune_job.ollama_fine_tune",
        "schedule": crontab(hour=5, minute=0, day_of_week=1),
    },
    # Every 30 min — Aggregate token usage for billing
    "token-usage-aggregate": {
        "task": "app.workers.token_aggregator_job.token_usage_aggregate",
        "schedule": crontab(minute="*/30"),
    },
}

# Auto-discover tasks
celery_app.autodiscover_tasks([
    "app.workers",
])
