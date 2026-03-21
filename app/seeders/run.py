"""
Seeder CLI — run seeders from the command line.

Usage:
    python -m app.seeders.run              # Run all seeders
    python -m app.seeders.run tenant       # Run only TenantSeeder
    python -m app.seeders.run voice        # Run only VoiceProfileSeeder

Like: php artisan db:seed
      php artisan db:seed --class=TenantSeeder
"""

import asyncio
import logging
import sys

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.seeders.database_seeder import DatabaseSeeder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


async def main():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as db:
        try:
            if len(sys.argv) > 1:
                # Run specific seeder: python -m app.seeders.run tenant
                seeder_name = sys.argv[1]
                await DatabaseSeeder.run_one(db, seeder_name)
            else:
                # Run all: python -m app.seeders.run
                await DatabaseSeeder.run_all(db)

            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Seeding failed: {e}")
            sys.exit(1)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
