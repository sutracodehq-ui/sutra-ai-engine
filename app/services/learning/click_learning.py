"""
ClickLearningService — The self-learning brain of Click Shield.

Analyzes historical ClickLogs and ClickFeedback to:
1. Discover new bot IP clusters.
2. Adjust rule weights based on false positives/negatives.
3. Train an Isolation Forest model for anomaly detection.
"""

import logging
import numpy as np
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sklearn.ensemble import IsolationForest
import pickle
import redis

from app.models.click_log import ClickLog
from app.models.click_feedback import ClickFeedback

logger = logging.getLogger(__name__)

class ClickLearningService:
    """Service for autonomous fraud detection optimization."""

    def __init__(self, redis_client):
        self._redis = redis_client

    async def run_optimization_cycle(self, db: AsyncSession, tenant_id: int):
        """
        Main learning loop:
        1. Fetch last 5000 clicks + any feedback.
        2. Identify 'High Certainty' bot clusters.
        3. Push confirmed bot IPs to Redis dynamic blacklist.
        4. (Optional) Train/Update Isolation Forest model.
        """
        logger.info(f"ClickLearn: Starting optimization cycle for tenant {tenant_id}")
        
        # 1. Fetch data
        clicks = await self._fetch_training_data(db, tenant_id)
        if len(clicks) < 100:
            logger.info("ClickLearn: Not enough data for learning yet.")
            return

        # 2. Discover IP Clusters with high fraud rate
        await self._discover_bad_ips(db, tenant_id)

        # 3. Train Isolation Forest
        await self._train_anomaly_detector(clicks, tenant_id)

        logger.info(f"ClickLearn: Completed optimization for tenant {tenant_id}")

    async def _fetch_training_data(self, db: AsyncSession, tenant_id: int, limit: int = 5000):
        """Fetch click logs for ML training."""
        stmt = (
            select(ClickLog)
            .where(ClickLog.tenant_id == tenant_id)
            .order_by(ClickLog.created_at.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        return result.scalars().all()

    async def _discover_bad_ips(self, db: AsyncSession, tenant_id: int):
        """Find IPs with suspiciously high click counts/fraud scores."""
        stmt = (
            select(
                ClickLog.ip_address,
                func.count(ClickLog.id).label("total"),
                func.avg(ClickLog.fraud_score).label("avg_score")
            )
            .where(ClickLog.tenant_id == tenant_id)
            .group_by(ClickLog.ip_address)
            .having(func.count(ClickLog.id) > 20)
            .order_by(func.avg(ClickLog.fraud_score).desc())
            .limit(50)
        )
        result = await db.execute(stmt)
        bad_ips = result.all()

        for ip, count, avg_score in bad_ips:
            if avg_score > 70:
                # Add to dynamic blacklist in Redis for 24 hours
                key = f"shield:blacklist:{tenant_id}:{ip}"
                await self._redis.setex(key, 86400, "bot_cluster")
                logger.info(f"ClickLearn: Blacklisted suspicious IP {ip} (score={avg_score:.1f})")

    async def _train_anomaly_detector(self, clicks: list[ClickLog], tenant_id: int):
        """
        Train an Isolation Forest on numeric behavioral features.
        Features: mouse_moves, scroll_depth, touch_events, time_on_page_ms
        """
        features = []
        for c in clicks:
            signals = c.client_data.get("signals", {})
            features.append([
                signals.get("mouse_moves", 0),
                signals.get("scroll_depth", 0),
                signals.get("touch_events", 0),
                signals.get("time_on_page_ms", 0)
            ])

        X = np.array(features)
        
        # Fit Isolation Forest
        # contamination=0.1 assumes 10% of traffic is typically anomalous
        clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        clf.fit(X)

        # Store model in Redis (or S3 in production)
        # In a real app, we'd use a more robust model registry
        model_data = pickle.dumps(clf)
        await self._redis.set(f"shield:model:{tenant_id}:isolation_forest", model_data)
        logger.info(f"ClickLearn: Updated Isolation Forest model for tenant {tenant_id}")
