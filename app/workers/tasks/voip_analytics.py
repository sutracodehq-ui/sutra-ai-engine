"""
VoIP Analytics — Daily call metrics and sentiment analysis.

Runs daily at 6 AM IST:
1. Aggregate call metrics (volume, duration, language distribution)
2. Run sentiment analysis on call transcriptions
3. Quality score per persona
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="voip_analytics", max_retries=1)
def voip_analytics(self):
    """Daily VoIP call analytics aggregation."""
    import asyncio
    asyncio.run(_analyze_calls())


async def _analyze_calls():
    """Process yesterday's calls and generate insights."""
    log_path = Path("training/call_logs/call_log.jsonl")
    if not log_path.exists():
        logger.info("VoIP Analytics: no call logs found")
        return

    # Read yesterday's calls
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")

    calls = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("timestamp", "").startswith(yesterday_str):
                    calls.append(entry)

    if not calls:
        logger.info(f"VoIP Analytics: no calls found for {yesterday_str}")
        return

    # ─── Aggregate Metrics ──────────────────────────────────
    total_calls = len(calls)
    total_duration = sum(c.get("duration", 0) for c in calls)

    # Language distribution
    lang_dist = {}
    for c in calls:
        lang = c.get("language", "unknown")
        lang_dist[lang] = lang_dist.get(lang, 0) + 1

    # Persona distribution
    persona_dist = {}
    for c in calls:
        persona = c.get("persona", "unknown")
        persona_dist[persona] = persona_dist.get(persona, 0) + 1

    # ─── Sentiment Analysis ────────────────────────────────
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    for c in calls:
        text = c.get("caller_text", "")
        # Simple keyword-based sentiment (upgrade to LLM later)
        text_lower = text.lower()
        if any(w in text_lower for w in ["thanks", "great", "perfect", "dhanyavaad", "bahut accha"]):
            sentiments["positive"] += 1
        elif any(w in text_lower for w in ["problem", "issue", "angry", "kharab", "galat"]):
            sentiments["negative"] += 1
        else:
            sentiments["neutral"] += 1

    # ─── Save Report ───────────────────────────────────────
    report = {
        "date": yesterday_str,
        "total_calls": total_calls,
        "total_duration_seconds": round(total_duration, 1),
        "avg_duration_seconds": round(total_duration / max(total_calls, 1), 1),
        "language_distribution": lang_dist,
        "persona_distribution": persona_dist,
        "sentiment": sentiments,
        "satisfaction_rate": round(
            sentiments["positive"] / max(total_calls, 1) * 100, 1
        ),
    }

    report_path = Path("training/call_logs") / f"report_{yesterday_str}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(
        f"📞 VoIP Analytics: {total_calls} calls, "
        f"top lang={max(lang_dist, key=lang_dist.get, default='?')}, "
        f"satisfaction={report['satisfaction_rate']}%"
    )
