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
    # Daily 1 AM — Evolution: Discover, benchmark, upgrade, self-teach
    "evolve-ai": {
        "task": "evolve_ai",
        "schedule": crontab(hour=1, minute=0),
    },
    # Daily 2 AM — Meta-Prompt: Analyze failures → generate improved prompts
    "meta-prompt-optimize": {
        "task": "optimize_prompts",
        "schedule": crontab(hour=2, minute=0),
    },
    # Daily 2:30 AM — Prompt Engine: Evaluate and promote winning candidates
    "evaluate-prompt-promotions": {
        "task": "evaluate_prompt_promotions",
        "schedule": crontab(hour=2, minute=30),
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
    # Weekly Sunday 4 AM — Export feedback as JSONL training data
    "export-training-data": {
        "task": "export_training_data",
        "schedule": crontab(hour=4, minute=0, day_of_week=0),
        "kwargs": {"days_back": 7},
    },
    # Weekly Sunday 5 AM — LoRA fine-tune: export feedback + retrain model
    "lora-fine-tune": {
        "task": "fine_tune_model",
        "schedule": crontab(hour=5, minute=0, day_of_week=0),
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
    # Every hour — Scan web for AI trends, stocks, crypto
    "scan-web-intelligence": {
        "task": "scan_web_intelligence",
        "schedule": crontab(minute=0),  # Top of every hour
    },
    # Daily 6 AM — VoIP call analytics and sentiment
    "voip-analytics": {
        "task": "voip_analytics",
        "schedule": crontab(hour=6, minute=0),
    },
}

# Auto-discover tasks
celery_app.autodiscover_tasks([
    "app.workers",
    "app.workers.tasks",
])
