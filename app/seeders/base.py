"""
Base Seeder — abstract seeder contract.

Software Factory: each seeder is a self-contained unit that can run independently.
Like Laravel's Seeder class — implement `run()` with your seed logic.
"""

import logging
from abc import ABC, abstractmethod

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class BaseSeeder(ABC):
    """Abstract base seeder — all seeders implement `run()`."""

    name: str = "base"

    @abstractmethod
    async def run(self, db: AsyncSession) -> None:
        """Execute the seeder. Override in subclass."""
        ...

    async def execute(self, db: AsyncSession) -> None:
        """Run the seeder with logging and error handling."""
        logger.info(f"🌱 Seeding: {self.name}...")
        try:
            await self.run(db)
            logger.info(f"✅ Seeded: {self.name}")
        except Exception as e:
            logger.error(f"❌ Seeder failed: {self.name} — {e}")
            raise
