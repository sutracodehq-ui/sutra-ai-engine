"""V1 API Router — aggregates all v1 route modules."""

from fastapi import APIRouter

from app.api.v1.chat import router as chat_router
from app.api.v1.agents import router as agents_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.tasks import router as tasks_router
from app.api.v1.tenants import router as tenants_router
from app.api.v1.intelligence import router as intelligence_router
from app.api.v1.rag import router as rag_router
from app.api.v1.billing import router as billing_router
from app.api.v1.auth import router as auth_router
from app.api.v1.voice import router as voice_router

router = APIRouter()
router.include_router(auth_router)
router.include_router(chat_router)
router.include_router(conversations_router)
router.include_router(agents_router)
router.include_router(voice_router)
router.include_router(intelligence_router)
router.include_router(rag_router)
router.include_router(billing_router)
