"""
V1 API Router — aggregates all v1 route modules.

Route include order matches the Swagger tag order defined in config/openapi.yaml:
  🔧 System → 🔐 Admin & Auth → 🤖 Core AI → 🛠️ Capabilities → 📊 Data
"""

from fastapi import APIRouter

from app.config import get_settings

# ─── Import all route modules ──────────────────────────────────
# Admin & Auth
from app.api.v1.tenants import router as tenants_router
from app.api.v1.provision import router as provision_router

# Core AI
from app.api.v1.agents import router as agents_router
from app.api.v1.chat import router as chat_router
from app.api.v1.intelligence import router as intelligence_router

# Capabilities
from app.api.v1.content import router as content_router
from app.api.v1.voice import router as voice_router
from app.api.v1.url_analyzer import router as url_analyzer_router
from app.api.v1.clicks import router as clicks_router
from app.api.v1.rag import router as rag_router

# Data & History
from app.api.v1.conversations import router as conversations_router
from app.api.v1.tasks import router as tasks_router
from app.api.v1.billing import router as billing_router

# ─── Assemble Router ──────────────────────────────────────────

router = APIRouter()
settings = get_settings()
v1_prefix = settings.api_v1_prefix

# Order matches Swagger sections (config/openapi.yaml tag order)
# 🔐 Admin & Auth
router.include_router(tenants_router, prefix=v1_prefix)      # tenants + api-keys
router.include_router(provision_router, prefix=v1_prefix)     # provisioning

# 🤖 Core AI
router.include_router(agents_router, prefix=v1_prefix)        # agents
router.include_router(chat_router, prefix=v1_prefix)          # chat
router.include_router(intelligence_router, prefix=v1_prefix)  # intelligence

# 🛠️ Capabilities
router.include_router(content_router, prefix=v1_prefix)       # content
router.include_router(voice_router, prefix=v1_prefix)         # voice
router.include_router(url_analyzer_router, prefix=v1_prefix)  # url-analyzer
router.include_router(clicks_router, prefix=v1_prefix)        # click-shield
router.include_router(rag_router, prefix=v1_prefix)           # rag

# 📊 Data & History
router.include_router(conversations_router, prefix=v1_prefix) # conversations
router.include_router(tasks_router, prefix=v1_prefix)         # tasks
router.include_router(billing_router, prefix=v1_prefix)       # billing
