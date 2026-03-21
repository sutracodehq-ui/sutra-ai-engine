"""V1 API Router — aggregates all v1 route modules."""

from fastapi import APIRouter

from app.api.v1.chat import router as chat_router
from app.api.v1.agents import router as agents_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.tasks import router as tasks_router
from app.api.v1.tenants import router as tenants_router
from app.api.v1.intelligence import router as intelligence_router

router = APIRouter(prefix="/v1")
router.include_router(chat_router)
router.include_router(agents_router)
router.include_router(conversations_router)
router.include_router(tasks_router)
router.include_router(tenants_router)
router.include_router(intelligence_router)
