#!/bin/sh
# ─────────────────────────────────────────────────────────────
# SutraAI Engine — Container Entrypoint
#
# Runs on every container start (idempotent):
#   1. Wait for PostgreSQL
#   2. Run Alembic migrations
#   3. Seed default tenant (if none exists)
#   4. Start uvicorn
# ─────────────────────────────────────────────────────────────
set -e

echo "🧠 SutraAI Engine — Starting up..."

# ─── 1. Wait for PostgreSQL ─────────────────────────────────
# Extract host:port from DATABASE_URL — pure socket check, no psycopg2 needed.
DB_HOST=$(python -c "
import os, re
url = os.getenv('DATABASE_URL', '')
m = re.search(r'@([^:/]+):?(\d+)?/', url)
print(f\"{m.group(1)}:{m.group(2) or '5432'}\" if m else 'sutra-ai-postgres:5432')
")
echo "⏳ Waiting for PostgreSQL at ${DB_HOST}..."

MAX_RETRIES=30
RETRY=0
until python -c "
import socket, sys
host, port = '${DB_HOST}'.rsplit(':', 1)
try:
    s = socket.create_connection((host, int(port)), timeout=2)
    s.close()
    print('connected')
except Exception as e:
    print(f'  → {e}', file=sys.stderr)
    sys.exit(1)
"; do
    RETRY=$((RETRY + 1))
    if [ "$RETRY" -ge "$MAX_RETRIES" ]; then
        echo "❌ PostgreSQL not reachable after ${MAX_RETRIES} retries. Exiting."
        exit 1
    fi
    echo "   Retry $RETRY/$MAX_RETRIES..."
    sleep 2
done
echo "✅ PostgreSQL is ready"

# ─── 2. Run Alembic Migrations ──────────────────────────────
echo "📦 Running database migrations..."
alembic upgrade head
echo "✅ Migrations complete"

# ─── 3. Seed Default Tenant ─────────────────────────────────
echo "🌱 Checking tenant seed..."
python scripts/seed_tenant.py ${SEED_API_KEY:+--key "$SEED_API_KEY"}
echo "✅ Tenant check complete"

# ─── 4. Start Application ───────────────────────────────────
echo "🚀 Starting uvicorn..."
exec "$@"
