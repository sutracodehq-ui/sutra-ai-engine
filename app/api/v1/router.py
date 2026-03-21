"""V1 API Router — aggregates all v1 route modules."""

from fastapi import APIRouter

from app.api.v1.chat import router as chat_router
from app.api.v1.agents import router as agents_router
from app.api.v1.clicks import router as clicks_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.tasks import router as tasks_router
from app.api.v1.tenants import router as tenants_router
from app.api.v1.intelligence import router as intelligence_router
from app.api.v1.rag import router as rag_router
from app.api.v1.billing import router as billing_router
# from app.api.v1.auth import router as auth_router
from app.api.v1.voice import router as voice_router

from app.api.v1.provision import router as provision_router
from app.api.v1.url_analyzer import router as url_analyzer_router
from app.api.v1.content import router as content_router

router = APIRouter()
# router.include_router(auth_router, prefix="/v1")
router.include_router(chat_router, prefix="/v1")
router.include_router(conversations_router, prefix="/v1")
router.include_router(agents_router, prefix="/v1")
router.include_router(clicks_router, prefix="/v1")
router.include_router(voice_router, prefix="/v1")
router.include_router(intelligence_router, prefix="/v1")
router.include_router(rag_router, prefix="/v1")
router.include_router(billing_router, prefix="/v1")
router.include_router(provision_router)
router.include_router(url_analyzer_router)
router.include_router(content_router)
