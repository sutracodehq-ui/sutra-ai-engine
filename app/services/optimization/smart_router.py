"""
Smart Router — Auto-detects the best agent for a prompt.

Engine Optimization: Users don't need to know which agent to call.
Just send a prompt, and the router picks the right specialist.

Flow:
    "What's my tax liability?" → router → tax_planner
    "Write a product description" → router → product_description_writer
    "My tomato crop has yellow leaves" → router → crop_advisor
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    """Routing decision."""
    agent_id: str
    confidence: float       # 0.0 → 1.0
    reasoning: str
    alternatives: list[str] # Other possible agents


# ─── Keyword → Agent Routing Rules ──────────────────────────

ROUTING_RULES: list[dict] = [
    # Health
    {"keywords": ["blood test", "lab report", "cbc", "thyroid", "hemoglobin", "creatinine"], "agent": "lab_report_interpreter", "domain": "health"},
    {"keywords": ["symptom", "fever", "cough", "headache", "pain", "sick", "doctor"], "agent": "symptom_triage", "domain": "health"},
    {"keywords": ["diet", "nutrition", "meal plan", "calories", "weight loss", "keto"], "agent": "diet_planner", "domain": "health"},
    {"keywords": ["anxiety", "depression", "stress", "mental health", "counseling", "lonely"], "agent": "mental_health_companion", "domain": "health"},
    {"keywords": ["medicine", "drug", "tablet", "dosage", "side effect", "interaction"], "agent": "medicine_info", "domain": "health"},
    {"keywords": ["ayurveda", "dosha", "yoga", "pranayama", "herbal"], "agent": "ayurveda_advisor", "domain": "health"},

    # Legal
    {"keywords": ["contract", "agreement", "clause", "terms", "legal review"], "agent": "contract_analyzer", "domain": "legal"},
    {"keywords": ["rti", "right to information", "transparency"], "agent": "rti_drafter", "domain": "legal"},
    {"keywords": ["gst", "tax return", "hsn", "igst", "cgst", "invoice tax"], "agent": "gst_compliance", "domain": "legal"},
    {"keywords": ["nda", "mou", "partnership deed", "legal notice", "affidavit"], "agent": "legal_document_writer", "domain": "legal"},

    # HR
    {"keywords": ["resume", "cv", "candidate", "hiring", "screen"], "agent": "resume_screener", "domain": "hr"},
    {"keywords": ["interview question", "ask candidate"], "agent": "interview_q_generator", "domain": "hr"},
    {"keywords": ["job description", "jd", "open position", "hiring for"], "agent": "jd_writer", "domain": "hr"},
    {"keywords": ["salary", "ctc", "compensation", "pay scale", "benchmark"], "agent": "salary_benchmarker", "domain": "hr"},
    {"keywords": ["onboarding", "new employee", "first day", "joining"], "agent": "onboarding_guide", "domain": "hr"},

    # E-Commerce
    {"keywords": ["product description", "listing", "amazon", "flipkart"], "agent": "product_description_writer", "domain": "ecommerce"},
    {"keywords": ["review analysis", "customer feedback", "star rating", "sentiment"], "agent": "review_analyzer", "domain": "ecommerce"},
    {"keywords": ["pricing strategy", "price optimization", "discount"], "agent": "dynamic_pricing", "domain": "ecommerce"},
    {"keywords": ["returns", "refund", "rto", "return rate"], "agent": "returns_predictor", "domain": "ecommerce"},

    # Real Estate
    {"keywords": ["property value", "flat price", "apartment cost", "sq ft rate"], "agent": "property_valuator", "domain": "real_estate"},
    {"keywords": ["rental yield", "rent vs buy", "emi vs rent"], "agent": "rental_yield_calculator", "domain": "real_estate"},
    {"keywords": ["rera", "builder compliance", "carpet area"], "agent": "rera_compliance", "domain": "real_estate"},
    {"keywords": ["locality", "area comparison", "which area", "best location"], "agent": "area_comparator", "domain": "real_estate"},

    # Agriculture
    {"keywords": ["crop", "farming", "kharif", "rabi", "harvest", "seeds"], "agent": "crop_advisor", "domain": "agriculture"},
    {"keywords": ["soil", "ph level", "npk", "fertilizer"], "agent": "soil_report_interpreter", "domain": "agriculture"},
    {"keywords": ["monsoon", "weather", "planting season", "rainfall"], "agent": "weather_planting", "domain": "agriculture"},
    {"keywords": ["msp", "mandi", "market price", "sell crop"], "agent": "msp_tracker", "domain": "agriculture"},
    {"keywords": ["subsidy", "pm kisan", "government scheme", "farmer scheme"], "agent": "subsidy_finder", "domain": "agriculture"},

    # Finance
    {"keywords": ["income tax", "80c", "80d", "itr", "tax saving", "old regime", "new regime"], "agent": "tax_planner", "domain": "finance"},
    {"keywords": ["home loan", "personal loan", "car loan", "emi", "interest rate"], "agent": "loan_comparator", "domain": "finance"},
    {"keywords": ["insurance", "health cover", "term plan", "mediclaim", "claim"], "agent": "insurance_advisor", "domain": "finance"},
    {"keywords": ["sip", "mutual fund", "elss", "nav", "fund"], "agent": "sip_calculator", "domain": "finance"},
    {"keywords": ["retirement", "pension", "nps", "ppf", "epf"], "agent": "retirement_planner", "domain": "finance"},

    # Travel
    {"keywords": ["trip", "travel plan", "itinerary", "vacation", "holiday"], "agent": "trip_planner", "domain": "travel"},
    {"keywords": ["visa", "passport", "embassy", "travel document"], "agent": "visa_guide", "domain": "travel"},
    {"keywords": ["travel budget", "cheap flight", "affordable hotel"], "agent": "travel_budget_optimizer", "domain": "travel"},

    # Logistics
    {"keywords": ["delivery route", "route plan", "shortest path", "logistics"], "agent": "route_optimizer", "domain": "logistics"},
    {"keywords": ["shipment", "tracking", "courier", "delivery status"], "agent": "shipment_tracker", "domain": "logistics"},
    {"keywords": ["warehouse", "inventory layout", "storage"], "agent": "warehouse_planner", "domain": "logistics"},

    # Government
    {"keywords": ["scheme eligibility", "am i eligible", "government benefit"], "agent": "scheme_eligibility", "domain": "government"},
    {"keywords": ["complaint", "grievance", "cpgrams", "consumer forum"], "agent": "complaint_drafter", "domain": "government"},
    {"keywords": ["translate", "hindi to english", "english to hindi"], "agent": "document_translator", "domain": "government"},
    {"keywords": ["form fill", "passport form", "pan card", "aadhaar update", "voter id"], "agent": "form_filler", "domain": "government"},

    # Marketing
    {"keywords": ["write copy", "tagline", "headline", "ad copy", "slogan"], "agent": "copywriter", "domain": "marketing"},
    {"keywords": ["seo", "keywords", "meta description", "serp", "backlink"], "agent": "seo", "domain": "marketing"},
    {"keywords": ["social media post", "instagram", "twitter", "linkedin post"], "agent": "social", "domain": "marketing"},
    {"keywords": ["email campaign", "newsletter", "drip", "email sequence"], "agent": "email_campaign", "domain": "marketing"},

    # Productivity
    {"keywords": ["summarize email", "email summary", "inbox"], "agent": "email_summarizer", "domain": "productivity"},
    {"keywords": ["meeting notes", "meeting summary", "action items", "minutes"], "agent": "meeting_notes", "domain": "productivity"},
    {"keywords": ["invoice", "bill", "gst invoice", "proforma"], "agent": "invoice_generator", "domain": "productivity"},
    {"keywords": ["expense", "spending", "budget track", "money spent"], "agent": "expense_tracker", "domain": "productivity"},
]


class SmartRouter:
    """
    Auto-routes prompts to the best agent.
    
    Algorithm:
    1. Keyword matching with scoring
    2. Multiple keyword hits increase confidence
    3. Returns top match + alternatives
    """

    def __init__(self):
        self._rules = ROUTING_RULES

    def route(self, prompt: str) -> RouteResult:
        """
        Determine the best agent for a prompt.
        
        Returns RouteResult with agent_id, confidence, and alternatives.
        """
        prompt_lower = prompt.lower()
        scores: dict[str, float] = {}
        reasons: dict[str, list[str]] = {}

        for rule in self._rules:
            agent = rule["agent"]
            matched_keywords = [kw for kw in rule["keywords"] if kw in prompt_lower]

            if matched_keywords:
                # Score = proportion of keywords matched × weight
                score = len(matched_keywords) / len(rule["keywords"])
                # Boost for exact phrase matches
                for kw in matched_keywords:
                    if f" {kw} " in f" {prompt_lower} ":
                        score += 0.1

                scores[agent] = max(scores.get(agent, 0), min(score, 1.0))
                reasons[agent] = matched_keywords

        if not scores:
            # Default to copywriter (most general-purpose)
            return RouteResult(
                agent_id="copywriter",
                confidence=0.2,
                reasoning="No specific domain detected, using general-purpose agent.",
                alternatives=[],
            )

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_agent, best_score = ranked[0]
        alternatives = [a for a, _ in ranked[1:4]]

        return RouteResult(
            agent_id=best_agent,
            confidence=best_score,
            reasoning=f"Matched keywords: {reasons.get(best_agent, [])}",
            alternatives=alternatives,
        )

    def route_with_fallback(self, prompt: str, min_confidence: float = 0.3) -> RouteResult:
        """
        Route with fallback: if confidence is too low, return a more general agent.
        """
        result = self.route(prompt)

        if result.confidence < min_confidence:
            logger.info(
                f"SmartRouter: low confidence ({result.confidence:.1f}) for '{result.agent_id}', "
                f"suggesting user confirm agent selection."
            )
            result.reasoning += f" (Low confidence — consider specifying agent directly)"

        return result


# ─── Singleton ──────────────────────────────────────────────

_router: SmartRouter | None = None


def get_smart_router() -> SmartRouter:
    global _router
    if _router is None:
        _router = SmartRouter()
    return _router
