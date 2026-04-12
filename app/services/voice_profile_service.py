"""Voice Profile Service — stub for missing module."""
import logging
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

class VoiceProfileService:
    @staticmethod
    async def create(
        db: AsyncSession,
        tenant_id: int,
        name: str,
        slug: str,
        description: str | None = None,
        is_default: bool = False
    ):
        """Stub for creating a voice profile."""
        logger.warning(f"VoiceProfileService.create STUB called for tenant {tenant_id}")
        return None
