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

        # ─── Phase 9: Voice & Calling ─────────────────────
        from app.services.agents.cold_call_scripter import ColdCallScripterAgent
        from app.services.agents.call_sentiment_analyzer import CallSentimentAnalyzerAgent
        from app.services.agents.whatsapp_bot_builder import WhatsappBotBuilderAgent
        from app.services.agents.call_summarizer import CallSummarizerAgent
        from app.services.agents.ivr_designer import IvrDesignerAgent

        # ─── Phase 10: Video Intelligence ─────────────────
        from app.services.agents.youtube_analyzer import YoutubeAnalyzerAgent
        from app.services.agents.video_summarizer import VideoSummarizerAgent
        from app.services.agents.caption_generator import CaptionGeneratorAgent
        from app.services.agents.audio_dubber import AudioDubberAgent
        from app.services.agents.social_clip_maker import SocialClipMakerAgent

        # ─── Phase 11: EdTech Intelligence ────────────────
        from app.services.agents.note_generator import NoteGeneratorAgent
        from app.services.agents.key_points_extractor import KeyPointsExtractorAgent
        from app.services.agents.quiz_generator import QuizGeneratorAgent
        from app.services.agents.flashcard_creator import FlashcardCreatorAgent
        from app.services.agents.lecture_planner import LecturePlannerAgent

        # ─── Phase 12: Market & Finance Intelligence ──────
        from app.services.agents.stock_analyzer import StockAnalyzerAgent
        from app.services.agents.stock_predictor import StockPredictorAgent
        from app.services.agents.market_trend_analyzer import MarketTrendAnalyzerAgent
        from app.services.agents.ai_trend_tracker import AiTrendTrackerAgent
        from app.services.agents.crypto_analyzer import CryptoAnalyzerAgent

        # ─── Phase 13: Data & ML Intelligence ─────────────
        from app.services.agents.data_curator import DataCuratorAgent
        from app.services.agents.dataset_optimizer import DatasetOptimizerAgent
        from app.services.agents.ml_pipeline import MlPipelineAgent

        # ─── Phase 14: Health Intelligence ─────────────────
        from app.services.agents.lab_report_interpreter import LabReportInterpreterAgent
        from app.services.agents.symptom_triage import SymptomTriageAgent
        from app.services.agents.diet_planner import DietPlannerAgent
        from app.services.agents.mental_health_companion import MentalHealthCompanionAgent
        from app.services.agents.medicine_info import MedicineInfoAgent
        from app.services.agents.patient_followup import PatientFollowupAgent
        from app.services.agents.ayurveda_advisor import AyurvedaAdvisorAgent

        # ─── Phase 15: Legal & Compliance ──────────────────
        from app.services.agents.contract_analyzer import ContractAnalyzerAgent
        from app.services.agents.rti_drafter import RtiDrafterAgent
        from app.services.agents.gst_compliance import GstComplianceAgent
        from app.services.agents.legal_document_writer import LegalDocumentWriterAgent

        # ─── Phase 16: HR & Recruitment ────────────────────
        from app.services.agents.resume_screener import ResumeScreenerAgent
        from app.services.agents.interview_q_generator import InterviewQGeneratorAgent
        from app.services.agents.jd_writer import JdWriterAgent
        from app.services.agents.salary_benchmarker import SalaryBenchmarkerAgent
        from app.services.agents.onboarding_guide import OnboardingGuideAgent

        # ─── Phase 17: E-Commerce ──────────────────────────
        from app.services.agents.product_description_writer import ProductDescriptionWriterAgent
        from app.services.agents.review_analyzer import ReviewAnalyzerAgent
        from app.services.agents.dynamic_pricing import DynamicPricingAgent
        from app.services.agents.returns_predictor import ReturnsPredictorAgent
        from app.services.agents.catalog_enricher import CatalogEnricherAgent

        # ─── Phase 18: Real Estate ─────────────────────────
        from app.services.agents.property_valuator import PropertyValuatorAgent
        from app.services.agents.rental_yield_calculator import RentalYieldCalculatorAgent
        from app.services.agents.rera_compliance import ReraComplianceAgent
        from app.services.agents.area_comparator import AreaComparatorAgent

        # ─── Phase 19: Agriculture ─────────────────────────
        from app.services.agents.crop_advisor import CropAdvisorAgent
        from app.services.agents.soil_report_interpreter import SoilReportInterpreterAgent
        from app.services.agents.weather_planting import WeatherPlantingAgent
        from app.services.agents.msp_tracker import MspTrackerAgent
        from app.services.agents.subsidy_finder import SubsidyFinderAgent

        # ─── Phase 20: Personal Finance ────────────────────
        from app.services.agents.tax_planner import TaxPlannerAgent
        from app.services.agents.loan_comparator import LoanComparatorAgent
        from app.services.agents.insurance_advisor import InsuranceAdvisorAgent
        from app.services.agents.sip_calculator import SipCalculatorAgent
        from app.services.agents.retirement_planner import RetirementPlannerAgent

        # ─── Phase 21: Travel & Tourism ────────────────────
        from app.services.agents.trip_planner import TripPlannerAgent
        from app.services.agents.visa_guide import VisaGuideAgent
        from app.services.agents.travel_budget_optimizer import TravelBudgetOptimizerAgent
        from app.services.agents.cultural_advisor import CulturalAdvisorAgent
        from app.services.agents.itinerary_generator import ItineraryGeneratorAgent

        # ─── Phase 22: Logistics ───────────────────────────
        from app.services.agents.route_optimizer import RouteOptimizerAgent
        from app.services.agents.shipment_tracker import ShipmentTrackerAgent
        from app.services.agents.warehouse_planner import WarehousePlannerAgent
        from app.services.agents.last_mile_optimizer import LastMileOptimizerAgent

        # ─── Phase 23: Government Services ─────────────────
        from app.services.agents.scheme_eligibility import SchemeEligibilityAgent
        from app.services.agents.complaint_drafter import ComplaintDrafterAgent
        from app.services.agents.document_translator import DocumentTranslatorAgent
        from app.services.agents.form_filler import FormFillerAgent

        # ─── Phase 24: Customer Success ────────────────────
        from app.services.agents.nps_analyzer import NpsAnalyzerAgent
        from app.services.agents.retention_strategist import RetentionStrategistAgent
        from app.services.agents.feedback_synthesizer import FeedbackSynthesizerAgent
        from app.services.agents.churn_reversal import ChurnReversalAgent
        from app.services.agents.upsell_advisor import UpsellAdvisorAgent

        # ─── Phase 25: Daily Productivity ──────────────────
        from app.services.agents.email_summarizer import EmailSummarizerAgent
        from app.services.agents.meeting_notes import MeetingNotesAgent
        from app.services.agents.invoice_generator import InvoiceGeneratorAgent
        from app.services.agents.expense_tracker import ExpenseTrackerAgent
        from app.services.agents.daily_briefing import DailyBriefingAgent
        from app.services.agents.reminder_agent import ReminderAgent

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
            # Phase 9: Voice & Calling
            ColdCallScripterAgent, CallSentimentAnalyzerAgent,
            WhatsappBotBuilderAgent, CallSummarizerAgent, IvrDesignerAgent,
            # Phase 10: Video Intelligence
            YoutubeAnalyzerAgent, VideoSummarizerAgent, CaptionGeneratorAgent,
            AudioDubberAgent, SocialClipMakerAgent,
            # Phase 11: EdTech Intelligence
            NoteGeneratorAgent, KeyPointsExtractorAgent, QuizGeneratorAgent,
            FlashcardCreatorAgent, LecturePlannerAgent,
            # Phase 12: Market & Finance Intelligence
            StockAnalyzerAgent, StockPredictorAgent, MarketTrendAnalyzerAgent,
            AiTrendTrackerAgent, CryptoAnalyzerAgent,
            # Phase 13: Data & ML Intelligence
            DataCuratorAgent, DatasetOptimizerAgent, MlPipelineAgent,
            # Phase 14: Health Intelligence
            LabReportInterpreterAgent, SymptomTriageAgent, DietPlannerAgent,
            MentalHealthCompanionAgent, MedicineInfoAgent, PatientFollowupAgent,
            AyurvedaAdvisorAgent,
            # Phase 15: Legal & Compliance
            ContractAnalyzerAgent, RtiDrafterAgent, GstComplianceAgent,
            LegalDocumentWriterAgent,
            # Phase 16: HR & Recruitment
            ResumeScreenerAgent, InterviewQGeneratorAgent, JdWriterAgent,
            SalaryBenchmarkerAgent, OnboardingGuideAgent,
            # Phase 17: E-Commerce
            ProductDescriptionWriterAgent, ReviewAnalyzerAgent, DynamicPricingAgent,
            ReturnsPredictorAgent, CatalogEnricherAgent,
            # Phase 18: Real Estate
            PropertyValuatorAgent, RentalYieldCalculatorAgent,
            ReraComplianceAgent, AreaComparatorAgent,
            # Phase 19: Agriculture
            CropAdvisorAgent, SoilReportInterpreterAgent, WeatherPlantingAgent,
            MspTrackerAgent, SubsidyFinderAgent,
            # Phase 20: Personal Finance
            TaxPlannerAgent, LoanComparatorAgent, InsuranceAdvisorAgent,
            SipCalculatorAgent, RetirementPlannerAgent,
            # Phase 21: Travel & Tourism
            TripPlannerAgent, VisaGuideAgent, TravelBudgetOptimizerAgent,
            CulturalAdvisorAgent, ItineraryGeneratorAgent,
            # Phase 22: Logistics
            RouteOptimizerAgent, ShipmentTrackerAgent,
            WarehousePlannerAgent, LastMileOptimizerAgent,
            # Phase 23: Government Services
            SchemeEligibilityAgent, ComplaintDrafterAgent,
            DocumentTranslatorAgent, FormFillerAgent,
            # Phase 24: Customer Success
            NpsAnalyzerAgent, RetentionStrategistAgent, FeedbackSynthesizerAgent,
            ChurnReversalAgent, UpsellAdvisorAgent,
            # Phase 25: Daily Productivity
            EmailSummarizerAgent, MeetingNotesAgent, InvoiceGeneratorAgent,
            ExpenseTrackerAgent, DailyBriefingAgent, ReminderAgent,
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

    # ─── Safe Inter-Agent Delegation ──────────────────────────

    MAX_DELEGATION_DEPTH = 3     # Max hops: A → B → C → D (stops)
    DELEGATION_TIMEOUT = 30      # Seconds before a delegation times out

    async def delegate(
        self,
        from_agent: str,
        to_agent: str,
        prompt: str,
        context: dict | None = None,
        db: Any | None = None,
        _chain: list[str] | None = None,
        _depth: int = 0,
    ) -> dict:
        """
        Safe inter-agent delegation with:
        1. Max depth (3 hops) — prevents infinite delegation chains
        2. Cycle detection — agent A→B→A stops immediately
        3. Timeout (30s) — agent never waits forever
        4. Fallback — if delegation fails, returns best-effort response

        Usage (from within any agent):
            hub = get_agent_hub()
            result = await hub.delegate(
                from_agent="trip_planner",
                to_agent="visa_guide",
                prompt="What's the visa process for Thailand?",
            )
        """
        import asyncio

        chain = _chain or [from_agent]

        # ── Guard: max depth ──
        if _depth >= self.MAX_DELEGATION_DEPTH:
            logger.warning(
                f"Delegation depth limit ({self.MAX_DELEGATION_DEPTH}) reached: "
                f"{'→'.join(chain)}→{to_agent}. Returning fallback."
            )
            return {
                "status": "fallback",
                "reason": "max_depth_reached",
                "chain": chain,
                "response": f"I consulted with {to_agent} but couldn't get a detailed answer in time. "
                            f"Here's what I know based on my own expertise.",
            }

        # ── Guard: cycle detection ──
        if to_agent in chain:
            logger.warning(
                f"Delegation cycle detected: {'→'.join(chain)}→{to_agent}. Breaking cycle."
            )
            return {
                "status": "fallback",
                "reason": "cycle_detected",
                "chain": chain,
                "response": f"I've already consulted with {to_agent} in this chain. "
                            f"Proceeding with available information.",
            }

        # ── Guard: agent exists ──
        if to_agent not in self._agents:
            return {
                "status": "fallback",
                "reason": "agent_not_found",
                "chain": chain,
                "response": f"The specialist '{to_agent}' is not available right now.",
            }

        # ── Execute with timeout ──
        chain_ext = chain + [to_agent]
        logger.info(f"Delegation: {'→'.join(chain_ext)} (depth={_depth + 1})")

        try:
            result = await asyncio.wait_for(
                self.run(to_agent, prompt, db=db, context={
                    **(context or {}),
                    "_delegation_chain": chain_ext,
                    "_delegation_depth": _depth + 1,
                }),
                timeout=self.DELEGATION_TIMEOUT,
            )
            return {
                "status": "success",
                "chain": chain_ext,
                "response": result.content if hasattr(result, "content") else str(result),
                "agent": to_agent,
            }
        except asyncio.TimeoutError:
            logger.warning(f"Delegation to '{to_agent}' timed out after {self.DELEGATION_TIMEOUT}s")
            return {
                "status": "fallback",
                "reason": "timeout",
                "chain": chain_ext,
                "response": f"The {to_agent} specialist is taking too long. "
                            f"I'll provide my best answer based on what I know.",
            }
        except Exception as e:
            logger.error(f"Delegation to '{to_agent}' failed: {e}")
            return {
                "status": "fallback",
                "reason": "error",
                "chain": chain_ext,
                "response": f"Couldn't reach the {to_agent} specialist. "
                            f"Providing my best answer instead.",
            }

    async def multi_delegate(
        self,
        from_agent: str,
        to_agents: list[str],
        prompt: str,
        context: dict | None = None,
        db: Any | None = None,
    ) -> dict[str, dict]:
        """
        Delegate to multiple agents in parallel.
        All run concurrently — if one is slow/fails, others still return.

        Example: trip_planner delegates to [visa_guide, cultural_advisor, travel_budget_optimizer]
        """
        import asyncio

        async def _delegate_one(target: str):
            return target, await self.delegate(
                from_agent=from_agent,
                to_agent=target,
                prompt=prompt,
                context=context,
                db=db,
            )

        results = await asyncio.gather(
            *[_delegate_one(t) for t in to_agents],
            return_exceptions=True,
        )

        output = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Multi-delegation failed: {result}")
            else:
                agent_id, response = result
                output[agent_id] = response

        return output


# ─── Singleton ──────────────────────────────────────────────────

_hub: AiAgentHub | None = None


def get_agent_hub() -> AiAgentHub:
    global _hub
    if _hub is None:
        _hub = AiAgentHub()
    return _hub
