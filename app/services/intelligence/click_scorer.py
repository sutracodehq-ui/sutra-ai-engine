"""
ClickScorerService — Core logic for fraud detection.

Shield-AI: Real-time scoring using Redis-based rules, 
behavioral signals, and (future) ML anomaly detection.
"""

import logging
import hashlib
import time
from typing import List, Dict, Any, Tuple

from app.schemas.click_shield import ClickTrackRequest, ClickScoreReason
from app.config import get_settings

logger = logging.getLogger(__name__)

class ClickScorerService:
    """Service to evaluate click quality and detect fraud."""

    # Risk thresholds
    THRESHOLD_BLOCK = 80
    THRESHOLD_FLAG = 40

    # Rule weights
    WEIGHT_IP_RATE = 40
    WEIGHT_FP_RATE = 50
    WEIGHT_UA_SPOOF = 30
    WEIGHT_BEHAVIOR_ZERO = 60

    def __init__(self, redis):
        self._redis = redis
        self.settings = get_settings()

    async def score(self, request: ClickTrackRequest) -> Tuple[int, str, List[ClickScoreReason]]:
        """
        Calculate fraud score for a click.
        Returns: (score, action, reasons)
        """
        reasons = []
        base_score = 0

        # 0. Check Dynamic Blacklist (Self-Learning Result)
        blacklist_score, blacklist_reasons = await self._check_dynamic_blacklist(request)
        base_score += blacklist_score
        reasons.extend(blacklist_reasons)

        # 1. Real-time Rate Limiting (Redis Rules)
        rate_score, rate_reasons = await self._check_rate_limits(request)
        base_score += rate_score
        reasons.extend(rate_reasons)

        # 2. ML Anomaly Detection (Self-Learning Model)
        ml_score, ml_reasons = await self._check_ml_anomaly(request)
        base_score += ml_score
        reasons.extend(ml_reasons)

        # 3. Behavioral Scoring
        behavior_score, behavior_reasons = self._check_behavior(request)
        base_score += behavior_score
        reasons.extend(behavior_reasons)

        # 3. Environment/Fingerprint Analysis
        env_score, env_reasons = self._check_environment(request)
        base_score += env_score
        reasons.extend(env_reasons)

        # Cap score at 100
        total_score = min(max(base_score, 0), 100)

        # Determine action
        if total_score >= self.THRESHOLD_BLOCK:
            action = "BLOCK"
        elif total_score >= self.THRESHOLD_FLAG:
            action = "FLAG"
        else:
            action = "ALLOW"

        return total_score, action, reasons

    async def _check_rate_limits(self, request: ClickTrackRequest) -> Tuple[int, List[ClickScoreReason]]:
        """Check IP and Fingerprint rate limits in Redis."""
        reasons = []
        score = 0
        
        ip_hash = hashlib.sha256((request.ip or "unknown").encode()).hexdigest()[:12]
        fp_hash = request.client_data.fingerprint or "unknown"
        
        # Window: 1 minute
        current_window = int(time.time() / 60)
        
        # IP Rate Key
        ip_key = f"shield:ip:{request.tenant_id}:{ip_hash}:{current_window}"
        # Fingerprint Rate Key
        fp_key = f"shield:fp:{request.tenant_id}:{fp_hash}:{current_window}"
        
        try:
            pipe = self._redis.pipeline()
            pipe.incr(ip_key)
            pipe.expire(ip_key, 120)  # Keep for 2 mins
            pipe.incr(fp_key)
            pipe.expire(fp_key, 120)
            results = await pipe.execute()
            
            ip_count = results[0]
            fp_count = results[2]
            
            if ip_count > 5:
                score += self.WEIGHT_IP_RATE
                reasons.append(ClickScoreReason(
                    rule="IP_VELOCITY_HIGH",
                    score_impact=self.WEIGHT_IP_RATE,
                    description=f"IP address generated {ip_count} clicks in 60 seconds."
                ))
            
            if fp_count > 3:
                score += self.WEIGHT_FP_RATE
                reasons.append(ClickScoreReason(
                    rule="FP_VELOCITY_HIGH",
                    score_impact=self.WEIGHT_FP_RATE,
                    description=f"Device fingerprint generated {fp_count} clicks in 60 seconds."
                ))
                
        except Exception as e:
            logger.warning(f"ClickShield: Redis tracking failed: {e}")

        return score, reasons

    async def _check_dynamic_blacklist(self, request: ClickTrackRequest) -> Tuple[int, List[ClickScoreReason]]:
        """Check if IP was blacklisted by the Self-Learning engine."""
        key = f"shield:blacklist:{request.tenant_id}:{request.ip}"
        is_bad = await self._redis.get(key)
        
        if is_bad:
            return 100, [ClickScoreReason(
                rule="DYNAMIC_BLACKLIST",
                score_impact=100,
                description="IP address was blacklisted by the self-learning engine due to suspicious historical behavior."
            )]
        return 0, []

    async def _check_ml_anomaly(self, request: ClickTrackRequest) -> Tuple[int, List[ClickScoreReason]]:
        """Run the click through the tenant's latest Isolation Forest model."""
        import pickle
        import numpy as np
        
        model_data = await self._redis.get(f"shield:model:{request.tenant_id}:isolation_forest")
        if not model_data:
            return 0, []

        try:
            clf = pickle.loads(model_data)
            signals = request.client_data.signals
            X = np.array([[
                signals.mouse_moves,
                signals.scroll_depth,
                signals.touch_events,
                signals.time_on_page_ms
            ]])
            
            # predict returns -1 for outliers, 1 for inliers
            prediction = clf.predict(X)
            
            if prediction[0] == -1:
                return 40, [ClickScoreReason(
                    rule="ML_ANOMALY_DETECTED",
                    score_impact=40,
                    description="The behavioral pattern of this click is anomalous compared to normal human traffic for your ads."
                )]
        except Exception as e:
            logger.warning(f"ClickShield: ML scoring failed: {e}")

        return 0, []

    def _check_behavior(self, request: ClickTrackRequest) -> Tuple[int, List[ClickScoreReason]]:
        """Analyze client-side behavior signals."""
        reasons = []
        score = 0
        signals = request.client_data.signals

        # No mouse moves and no touch events = highly suspicious for non-redirect clicks
        if signals.mouse_moves == 0 and signals.touch_events == 0:
            score += self.WEIGHT_BEHAVIOR_ZERO
            reasons.append(ClickScoreReason(
                rule="ZERO_INTERACTION",
                score_impact=self.WEIGHT_BEHAVIOR_ZERO,
                description="No mouse or touch interaction detected on the landing page."
            ))

        # Very low time on page (e.g., bot just hitting and leaving)
        if signals.time_on_page_ms < 500:
            score += 20
            reasons.append(ClickScoreReason(
                rule="LOW_ENGAGEMENT_TIME",
                score_impact=20,
                description=f"User spent only {signals.time_on_page_ms}ms on the page."
            ))

        return score, reasons

    def _check_environment(self, request: ClickTrackRequest) -> Tuple[int, List[ClickScoreReason]]:
        """Analyze user agent and other environment signals."""
        reasons = []
        score = 0
        ua = request.client_data.ua.lower()

        bot_keywords = ["headless", "phantomjs", "selenium", "webdriver", "puppeteer", "bot", "crawler"]
        if any(keyword in ua for keyword in bot_keywords):
            score += 100
            reasons.append(ClickScoreReason(
                rule="BOT_USER_AGENT",
                score_impact=100,
                description=f"User Agent contains bot signatures: {request.client_data.ua}"
            ))

        return score, reasons
