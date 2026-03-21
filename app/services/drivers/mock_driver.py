"""Mock LLM Driver — returns predictable responses for development/testing."""

import json
from typing import AsyncGenerator

from app.services.drivers.base import LlmDriver, LlmResponse


# ─── Mock response templates per agent type ──────────────────

MOCK_RESPONSES: dict[str, dict] = {
    "copywriter": {
        "headline": "Transform Your Business with AI-Powered Solutions",
        "body": "In today's competitive landscape, staying ahead means embracing innovation. Our cutting-edge platform helps you create compelling content that resonates with your audience and drives measurable results.",
        "cta": "Start Your Free Trial →",
        "suggested_keywords": ["AI marketing", "content automation", "digital growth"],
    },
    "seo": {
        "meta_title": "AI Marketing Platform | Boost Your Digital Presence",
        "meta_description": "Transform your marketing with AI-powered content generation, SEO optimization, and multi-channel distribution. Start free today.",
        "keywords": ["AI marketing", "content generation", "SEO tools", "digital marketing automation"],
        "content_outline": [
            "1. Introduction to AI-Powered Marketing",
            "2. Key Features and Benefits",
            "3. How It Works",
            "4. Case Studies",
            "5. Getting Started",
        ],
        "suggestions": ["Add schema markup", "Optimize images with alt text", "Internal linking strategy"],
    },
    "social_media": {
        "post_text": "🚀 Ready to revolutionize your marketing game? Our AI platform creates scroll-stopping content in seconds. No more creative blocks — just results.\n\n💡 What would you create with unlimited AI power?",
        "hashtags": ["#AIMarketing", "#DigitalMarketing", "#ContentCreation", "#MarTech", "#GrowthHacking"],
        "best_time_to_post": "Tuesday 10:00 AM or Thursday 2:00 PM",
        "platform_tips": "Use a carousel format for 3x engagement. Lead with a bold hook.",
        "character_count": 198,
        "image_prompt": "Futuristic holographic dashboard showing marketing analytics with vibrant purple and blue gradients, minimalist style",
    },
    "email_campaign": {
        "subject_line": "Your marketing just got an upgrade ✨",
        "preview_text": "See how AI can transform your content strategy...",
        "email_body": "Hi there,\n\nImagine creating a week's worth of marketing content in minutes, not days.\n\nWith our AI-powered platform, you can:\n• Generate on-brand social posts instantly\n• Craft SEO-optimized blog content\n• Design email campaigns that convert\n\nReady to see it in action?\n\nBest,\nThe Team",
        "cta": "Try It For Free",
        "subject_variants": [
            "The future of marketing is here 🔮",
            "We just saved 40 hours of content creation",
            "Your competitors are already using this...",
        ],
        "tips": ["Personalize the greeting", "A/B test subject lines", "Send Tuesday 10 AM for best open rates"],
    },
    "whatsapp": {
        "template_body": "Hi {{1}}! 👋\n\nWe have exciting news — our new AI marketing tools are now live!\n\nCreate stunning content, optimize your SEO, and manage campaigns all in one place.\n\nWant to learn more?",
        "header_text": "New Feature Alert! 🎉",
        "footer_text": "Reply STOP to unsubscribe",
        "buttons": [{"type": "QUICK_REPLY", "text": "Tell me more"}, {"type": "QUICK_REPLY", "text": "Not now"}],
        "compliance_notes": "Template must be pre-approved by Meta. Include opt-out option.",
        "engagement_tips": "Use personalization tokens. Keep messages under 1024 characters.",
    },
    "sms": {
        "message_body": "🚀 Your marketing AI is ready! Create content 10x faster. Try free: https://link.co/start Reply STOP to opt out",
        "character_count": 108,
        "variants": [
            "AI-powered marketing is here! Generate posts, emails & ads instantly. Start free: https://link.co/go",
            "Stop spending hours on content. Let AI do it in seconds. Try now: https://link.co/try",
        ],
        "cta": "Start Free",
        "best_send_time": "Weekdays 11 AM - 1 PM",
        "tips": ["Keep under 160 chars for single SMS", "Include clear CTA", "Add opt-out language"],
    },
    "ad_creative": {
        "headline": "Create Marketing Content 10x Faster",
        "body": "AI-powered platform for social posts, emails, ads, and SEO. Start your free trial.",
        "cta": "Start Free Trial",
        "suggested_keywords": ["AI marketing tool", "content automation", "marketing AI"],
    },
}


def _detect_agent(system_prompt: str) -> str:
    """Detect agent type from system prompt keywords."""
    lower = system_prompt.lower()
    if "seo" in lower:
        return "seo"
    if "social media" in lower or "social_media" in lower:
        return "social_media"
    if "email" in lower:
        return "email_campaign"
    if "whatsapp" in lower:
        return "whatsapp"
    if "sms" in lower:
        return "sms"
    if "ad" in lower and "creative" in lower:
        return "ad_creative"
    return "copywriter"


class MockDriver(LlmDriver):
    """Returns predictable mock responses — zero API cost, instant response."""

    def name(self) -> str:
        return "mock"

    async def complete(self, system_prompt: str, user_prompt: str, **options) -> LlmResponse:
        agent_type = _detect_agent(system_prompt)
        response_data = MOCK_RESPONSES.get(agent_type, MOCK_RESPONSES["copywriter"])
        content = json.dumps(response_data, indent=2)

        return LlmResponse(
            content=content,
            raw_response=content,
            prompt_tokens=len(system_prompt.split()) + len(user_prompt.split()),
            completion_tokens=len(content.split()),
            total_tokens=len(system_prompt.split()) + len(user_prompt.split()) + len(content.split()),
            model="mock",
            driver="mock",
        )

    async def chat(self, messages: list[dict], **options) -> LlmResponse:
        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return await self.complete(system_prompt, user_prompt, **options)

    async def stream(self, messages: list[dict], **options) -> AsyncGenerator[str, None]:
        response = await self.chat(messages, **options)
        # Simulate streaming by yielding chunks
        words = response.content.split(" ")
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i : i + 3])
            yield chunk + " "
