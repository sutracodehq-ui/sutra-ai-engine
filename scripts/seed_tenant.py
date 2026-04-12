"""
Seed Script — Auto-provision default tenant.

Run at container startup (idempotent — safe to run every time).
Creates a tenant if none exists, using the API key from env or generating one.

Usage:
    python scripts/seed_tenant.py              # Uses SEED_API_KEY from env
    python scripts/seed_tenant.py --key sk_live_xxx  # Explicit key
"""

import argparse
import hashlib
import os
import secrets
import sys

import psycopg2


def generate_api_key(prefix: str = "sk_live") -> str:
    return f"{prefix}_{secrets.token_urlsafe(32)}"


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def key_prefix(raw_key: str) -> str:
    parts = raw_key.split("_", 2)
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2][:8]}..."
    return raw_key[:16] + "..."


def get_sync_dsn() -> str:
    """Convert async DATABASE_URL to sync psycopg2 DSN."""
    url = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/sutra_ai")
    # Strip the SQLAlchemy driver prefix
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    url = url.replace("postgresql+psycopg2://", "postgresql://")
    return url


def seed(explicit_key: str | None = None):
    dsn = get_sync_dsn()
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor()

    # Check if any tenant exists
    cur.execute("SELECT COUNT(*) FROM tenants")
    count = cur.fetchone()[0]

    if count > 0:
        cur.execute("SELECT id, slug, live_key_prefix FROM tenants LIMIT 3")
        rows = cur.fetchall()
        print(f"✅ {count} tenant(s) already exist:")
        for row in rows:
            print(f"   id={row[0]} slug={row[1]} key={row[2]}")
        cur.close()
        conn.close()
        return

    # Determine the API key to use
    raw_live_key = explicit_key or os.getenv("SEED_API_KEY") or generate_api_key("sk_live")
    raw_test_key = generate_api_key("sk_test")

    tenant_name = os.getenv("SEED_TENANT_NAME", "SutraCode Platform")
    tenant_slug = os.getenv("SEED_TENANT_SLUG", "sutracode")
    tenant_email = os.getenv("SEED_TENANT_EMAIL", "admin@sutracode.in")

    cur.execute(
        """
        INSERT INTO tenants (name, slug, live_key_hash, live_key_prefix,
                             test_key_hash, test_key_prefix, is_active,
                             contact_email, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, true, %s, NOW(), NOW())
        RETURNING id
        """,
        (
            tenant_name,
            tenant_slug,
            hash_key(raw_live_key),
            key_prefix(raw_live_key),
            hash_key(raw_test_key),
            key_prefix(raw_test_key),
            tenant_email,
        ),
    )

    tenant_id = cur.fetchone()[0]
    cur.close()
    conn.close()

    print(f"✅ Tenant seeded: id={tenant_id} slug={tenant_slug}")
    print(f"   Live key: {raw_live_key}")
    print(f"   Test key: {raw_test_key}")
    print()
    print("   ⚠️  Save these keys — they cannot be retrieved later.")
    print(f"   Set SUTRA_AI_API_KEY={raw_live_key} in your Laravel .env")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed default tenant")
    parser.add_argument("--key", help="Explicit live API key to use")
    args = parser.parse_args()
    seed(explicit_key=args.key)
