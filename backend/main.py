import sys
from pathlib import Path

# Add the backend directory to sys.path to ensure 'app' imports work seamlessly
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# pyrefly: ignore [missing-import]
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api.routes import api_router
from app.database.init_db import init_db

# Initialize system-wide logging
logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager handling startup initialization and shutdown tasks.
    """
    logger.info("Initializing GuardianEye backend services...")
    try:
        await init_db()
    except Exception as e:
        logger.warning(f"Database auto-initialization skipped or failed: {e}")
    yield
    logger.info("GuardianEye backend shutdown complete.")

# Initialize FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="GuardianEye AI-Powered Surveillance Platform API",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Configure CORS (Cross-Origin Resource Sharing) middleware
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Register v1 API Routes
app.include_router(api_router, prefix=settings.API_V1_STR)

# Basic health status API endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "database": "configured"
    }

# Uvicorn bootstrapper for local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

