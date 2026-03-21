"""
AI Agent Hub — Software Factory registry and orchestrator.

Config-driven: agents self-register from configuration.
New agents = new YAML config + one-line class. Zero hub changes.
"""

import logging
from typing import Any

from app.services.agents.base import BaseAgent
from app.services.drivers.base import LlmResponse
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)


class AiAgentHub:
    """
    Central registry and orchestrator for all AI agents.

    Software Factory pattern:
    - Agents self-register via config-driven discovery
    - The hub routes tasks to the correct agent by identifier
    - Adding a new agent = add YAML config + class, register here
    """

    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}
        self._auto_register()

    def _auto_register(self):
        """Auto-register all built-in agents. Software Factory: config-driven assembly."""
        # ─── Core Marketing Agents ────────────────────────
        from app.services.agents.copywriter import CopywriterAgent
        from app.services.agents.seo import SeoAgent
        from app.services.agents.social import SocialAgent
        from app.services.agents.email_campaign import EmailCampaignAgent
        from app.services.agents.whatsapp import WhatsappAgent
        from app.services.agents.sms import SmsAgent
        from app.services.agents.ad_creative import AdCreativeAgent
        from app.services.agents.brand_auditor import BrandAuditorAgent
        from app.services.agents.content_repurpose import ContentRepurposerAgent
        from app.services.agents.click_shield import ClickShieldAgent

        # ─── Phase 1: Marketing Intelligence ─────────────
        from app.services.agents.persona_builder import PersonaBuilderAgent
        from app.services.agents.campaign_strategist import CampaignStrategistAgent
        from app.services.agents.ab_test_advisor import AbTestAdvisorAgent
        from app.services.agents.competitor_analyst import CompetitorAnalystAgent
        from app.services.agents.url_analyzer import UrlAnalyzerAgent

        # ─── Phase 2: Analytics & Insights ────────────────
        from app.services.agents.performance_reporter import PerformanceReporterAgent
        from app.services.agents.budget_optimizer import BudgetOptimizerAgent
        from app.services.agents.anomaly_alerter import AnomalyAlerterAgent

        # ─── Phase 3: Creative & Media ────────────────────
        from app.services.agents.visual_designer import VisualDesignerAgent
        from app.services.agents.video_scriptwriter import VideoScriptwriterAgent
        from app.services.agents.landing_page_builder import LandingPageBuilderAgent

        # ─── Phase 4: Autonomous Operations ───────────────
        from app.services.agents.auto_publisher import AutoPublisherAgent
        from app.services.agents.lead_scorer import LeadScorerAgent
        from app.services.agents.chatbot_trainer import ChatbotTrainerAgent

        # ─── Phase 5: Reputation & Growth ─────────────────
        from app.services.agents.review_reputation import ReviewReputationAgent
        from app.services.agents.trend_spotter import TrendSpotterAgent
        from app.services.agents.funnel_analyzer import FunnelAnalyzerAgent
        from app.services.agents.influencer_matcher import InfluencerMatcherAgent
        from app.services.agents.journey_mapper import JourneyMapperAgent

        # ─── Phase 6: Smart Automation ────────────────────
        from app.services.agents.auto_scheduler import AutoSchedulerAgent
        from app.services.agents.audience_segmenter import AudienceSegmenterAgent
        from app.services.agents.churn_predictor import ChurnPredictorAgent

        # ─── Phase 7: Analytics & Intelligence ────────────
        from app.services.agents.roi_calculator import RoiCalculatorAgent
        from app.services.agents.content_grader import ContentGraderAgent
        from app.services.agents.attribution_analyst import AttributionAnalystAgent
        from app.services.agents.pricing_strategist import PricingStrategistAgent

        # ─── Phase 8: Platform-Specific ───────────────────
        from app.services.agents.google_ads_optimizer import GoogleAdsOptimizerAgent
        from app.services.agents.meta_ads_optimizer import MetaAdsOptimizerAgent
        from app.services.agents.linkedin_growth import LinkedinGrowthAgent

        llm = get_llm_service()
        for agent_cls in [
            # Core
            CopywriterAgent, SeoAgent, SocialAgent, EmailCampaignAgent,
            WhatsappAgent, SmsAgent, AdCreativeAgent,
            BrandAuditorAgent, ContentRepurposerAgent, ClickShieldAgent,
            # Phase 1: Marketing Intelligence
            PersonaBuilderAgent, CampaignStrategistAgent,
            AbTestAdvisorAgent, CompetitorAnalystAgent, UrlAnalyzerAgent,
            # Phase 2: Analytics & Insights
            PerformanceReporterAgent, BudgetOptimizerAgent, AnomalyAlerterAgent,
            # Phase 3: Creative & Media
            VisualDesignerAgent, VideoScriptwriterAgent, LandingPageBuilderAgent,
            # Phase 4: Autonomous Operations
            AutoPublisherAgent, LeadScorerAgent, ChatbotTrainerAgent,
            # Phase 5: Reputation & Growth
            ReviewReputationAgent, TrendSpotterAgent, FunnelAnalyzerAgent,
            InfluencerMatcherAgent, JourneyMapperAgent,
            # Phase 6: Smart Automation
            AutoSchedulerAgent, AudienceSegmenterAgent, ChurnPredictorAgent,
            # Phase 7: Analytics & Intelligence
            RoiCalculatorAgent, ContentGraderAgent, AttributionAnalystAgent,
            PricingStrategistAgent,
            # Phase 8: Platform-Specific
            GoogleAdsOptimizerAgent, MetaAdsOptimizerAgent, LinkedinGrowthAgent,
        ]:
            agent = agent_cls(llm)
            self.register(agent)

    def register(self, agent: BaseAgent) -> None:
        """Register an agent in the hub."""
        self._agents[agent.identifier] = agent
        logger.info(f"AiAgentHub: registered agent '{agent.identifier}'")

    def get(self, identifier: str) -> BaseAgent:
        """Resolve an agent by identifier."""
        agent = self._agents.get(identifier)
        if not agent:
            raise ValueError(f"Agent '{identifier}' not registered. Available: {list(self._agents.keys())}")
        return agent

    def available_agents(self) -> list[str]:
        """List all registered agent identifiers."""
        return list(self._agents.keys())

    def agent_info(self) -> list[dict]:
        """Get metadata for all registered agents."""
        return [agent.info() for agent in self._agents.values()]

    async def run(
        self, 
        agent_type: str, 
        prompt: str, 
        db: Any | None = None,
        context: dict | None = None, 
        **options
    ) -> LlmResponse:
        """Dispatch a task to the appropriate agent."""
        agent = self.get(agent_type)
        logger.info(f"AiAgentHub: running agent '{agent_type}'")
        return await agent.execute(prompt, db=db, context=context, **options)

    async def run_in_conversation(
        self,
        agent_type: str,
        prompt: str,
        history: list[dict],
        db: Any | None = None,
        context: dict | None = None,
        **options,
    ) -> LlmResponse:
        """Run a task within a conversation with full history."""
        agent = self.get(agent_type)
        return await agent.execute_in_conversation(prompt, history, db=db, context=context, **options)

    async def batch(
        self, 
        prompt: str, 
        agent_types: list[str], 
        db: Any | None = None,
        context: dict | None = None, 
        **options
    ) -> dict[str, LlmResponse]:
        """Run multiple agents in parallel on the same prompt."""
        import asyncio

        async def _run(agent_type: str):
            return agent_type, await self.run(agent_type, prompt, db=db, context=context, **options)

        results = await asyncio.gather(*[_run(t) for t in agent_types], return_exceptions=True)

        output = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"AiAgentHub: batch agent failed: {result}")
            else:
                agent_type, response = result
                output[agent_type] = response

        return output


# ─── Singleton ──────────────────────────────────────────────────

_hub: AiAgentHub | None = None


def get_agent_hub() -> AiAgentHub:
    global _hub
    if _hub is None:
        _hub = AiAgentHub()
    return _hub
