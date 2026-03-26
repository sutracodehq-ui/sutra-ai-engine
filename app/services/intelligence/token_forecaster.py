"""
Token Forecaster — Provides cost analysis and usage predictions.

Optimization-AI: Analyzes historical token usage to help tenants 
stay within budget and understand their ROI.
"""

import logging
from datetime import datetime, date, timedelta
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.token_usage_log import TokenUsageLog

logger = logging.getLogger(__name__)


class TokenForecaster:
    """Service to analyze and forecast AI token costs."""

    @classmethod
    async def get_tenant_summary(cls, db: AsyncSession, tenant_id: int) -> dict:
        """Get current month summary and forecast for a tenant."""
        today = date.today()
        first_day = today.replace(day=1)
        
        # 1. Total cost this month
        stmt = (
            select(func.sum(TokenUsageLog.cost_usd), func.sum(TokenUsageLog.prompt_tokens), func.sum(TokenUsageLog.completion_tokens))
            .where(TokenUsageLog.tenant_id == tenant_id)
            .where(TokenUsageLog.log_date >= first_day)
        )
        result = await db.execute(stmt)
        total_cost, prompt_t, completion_t = result.fetchone()
        
        total_cost = float(total_cost or 0.0)
        days_elapsed = today.day
        
        # 2. Daily Average
        daily_avg = total_cost / days_elapsed if days_elapsed > 0 else 0.0
        
        # 3. Forecast for EOM (End of Month)
        import calendar
        _, last_day_num = calendar.monthrange(today.year, today.month)
        forecast_eom = daily_avg * last_day_num

        # 4. Usage by Model
        model_stmt = (
            select(TokenUsageLog.model, func.sum(TokenUsageLog.cost_usd))
            .where(TokenUsageLog.tenant_id == tenant_id)
            .where(TokenUsageLog.log_date >= first_day)
            .group_by(TokenUsageLog.model)
        )
        model_result = await db.execute(model_stmt)
        usage_by_model = {row[0]: float(row[1]) for row in model_result.all()}

        return {
            "tenant_id": tenant_id,
            "period": today.strftime("%B %Y"),
            "current_cost_usd": round(total_cost, 4),
            "daily_average_usd": round(daily_avg, 4),
            "forecast_eom_usd": round(forecast_eom, 4),
            "tokens": {
                "prompt": int(prompt_t or 0),
                "completion": int(completion_t or 0),
                "total": int((prompt_t or 0) + (completion_t or 0))
            },
            "usage_by_model": usage_by_model
        }
