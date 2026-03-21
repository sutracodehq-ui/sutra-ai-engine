"""
Voice Profile Seeder — creates default voice profiles per tenant.

Idempotent: skips profiles that already exist.
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tenant import Tenant
from app.models.voice_profile import VoiceProfile
from app.seeders.base import BaseSeeder

logger = logging.getLogger(__name__)

VOICE_PROFILES = [
    {
        "name": "Professional",
        "tone": "professional",
        "style": "formal",
        "instructions": "Use a corporate, polished tone. Avoid slang and casual language. Be authoritative and data-driven.",
        "is_default": True,
    },
    {
        "name": "Conversational",
        "tone": "friendly",
        "style": "casual",
        "instructions": "Write like you're talking to a friend. Use contractions, simple language, and a warm tone. Be approachable.",
        "is_default": False,
    },
    {
        "name": "Bold & Edgy",
        "tone": "bold",
        "style": "provocative",
        "instructions": "Be punchy and direct. Use power words, short sentences, and strong opinions. Challenge the reader.",
        "is_default": False,
    },
]


class VoiceProfileSeeder(BaseSeeder):
    name = "VoiceProfileSeeder"

    async def run(self, db: AsyncSession) -> None:
        # Get all tenants
        result = await db.execute(select(Tenant))
        tenants = result.scalars().all()

        if not tenants:
            logger.warning("  ⚠️  No tenants found — run TenantSeeder first")
            return

        for tenant in tenants:
            for profile_data in VOICE_PROFILES:
                # Check if profile exists for this tenant
                existing = await db.execute(
                    select(VoiceProfile).where(
                        VoiceProfile.tenant_id == tenant.id,
                        VoiceProfile.name == profile_data["name"],
                    )
                )
                if existing.scalar_one_or_none():
                    continue

                profile = VoiceProfile(
                    tenant_id=tenant.id,
                    name=profile_data["name"],
                    tone=profile_data["tone"],
                    style=profile_data["style"],
                    instructions=profile_data["instructions"],
                    is_default=profile_data["is_default"],
                )
                db.add(profile)
                logger.info(f"  ✅ Created voice profile '{profile_data['name']}' for {tenant.slug}")

        await db.flush()
