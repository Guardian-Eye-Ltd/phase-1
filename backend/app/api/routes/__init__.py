from fastapi import APIRouter
from app.api.routes.auth import router as auth_router
from app.api.routes.camera import router as camera_router
from app.api.routes.streams import router as streams_router
from app.api.routes.alerts import router as alerts_router
from app.api.routes.incidents import router as incidents_router
from app.api.routes.health import router as health_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(camera_router)
api_router.include_router(streams_router)
api_router.include_router(alerts_router)
api_router.include_router(incidents_router)
api_router.include_router(health_router)

__all__ = ["api_router"]
