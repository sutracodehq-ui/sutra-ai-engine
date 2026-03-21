"""
CLI Tool — rotate API keys for a tenant.

Software Factory: use this for manual key rotation from the terminal.
Usage: podman exec sutra-ai-api python -m app.scripts.rotate_key <slug> <live|test>
"""

import asyncio
import sys
import argparse

from app.db.session import async_session_factory
from app.services.tenant_service import TenantService

async def rotate(slug: str, environment: str):
    async with async_session_factory() as db:
        tenant = await TenantService.get_by_slug(db, slug)
        if not tenant:
            print(f"Error: Tenant with slug '{slug}' not found.")
            sys.exit(1)

        print(f"Rotating {environment} key for '{tenant.name}'...")
        
        if environment == "live":
            new_key = await TenantService.rotate_live_key(db, tenant)
        else:
            new_key = await TenantService.rotate_test_key(db, tenant)
            
        await db.commit()
        
        print("-" * 50)
        print(f"SUCCESS: New {environment.upper()} key generated.")
        print(f"Tenant: {tenant.name}")
        print(f"Raw Key: {new_key}")
        print("-" * 50)
        print("IMPORTANT: This key is valid immediately. Store it safely.")

def main():
    parser = argparse.ArgumentParser(description="Rotate SutraAI Tenant API Keys")
    parser.add_argument("slug", help="The tenant slug (e.g. tryambaka)")
    parser.add_argument("env", choices=["live", "test"], default="live", help="Which key to rotate")
    
    args = parser.parse_args()
    asyncio.run(rotate(args.slug, args.env))

if __name__ == "__main__":
    main()
