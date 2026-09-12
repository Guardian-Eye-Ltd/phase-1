import os
from typing import List, Union, Any
from pydantic import field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "GuardianEye"
    API_V1_STR: str = "/api/v1"
    
    # JWT Config
    SECRET_KEY: str = "super-secret-developer-key-replace-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120

    # Streaming Config
    STREAM_RECONNECT_INITIAL_DELAY: float = 1.0
    STREAM_RECONNECT_MAX_DELAY: float = 30.0
    STREAM_TARGET_FPS: int = 15
    STREAM_IDLE_TIMEOUT: float = 60.0
    STREAM_MAX_BUFFER_SIZE: int = 2

    # Demo Streams Config
    DEMO_STREAMS_ENABLED: bool = False
    DEMO_CAMERA_001_SOURCE: str = "rtsp://127.0.0.1:8554/camera01"
    DEMO_CAMERA_002_SOURCE: str = "rtsp://127.0.0.1:8554/camera02"
    DEMO_CAMERA_003_SOURCE: str = "rtsp://127.0.0.1:8554/camera03"
    DEMO_CAMERA_004_SOURCE: str = "rtsp://127.0.0.1:8554/camera04"
    DEMO_CAMERA_005_SOURCE: str = "rtsp://127.0.0.1:8554/camera05"
    DEMO_CAMERA_006_SOURCE: str = "rtsp://127.0.0.1:8554/camera06"

    # Evidence Storage Config
    STORAGE_DIR: str = "storage/evidence"
    DERIVED_STORAGE_DIR: str = "storage/derived"
    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_EXTENSIONS: List[str] = ["mp4", "webm", "avi", "mov", "mkv"]

    # Phase 1C Semantic Intelligence Config
    VLM_ENABLED: bool = True
    VLM_MODEL_NAME: str = "Salesforce/blip-image-captioning-base"
    MAX_KEYFRAMES_FOR_VLM: int = 30
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    VECTOR_DB_TYPE: str = "chromadb"
    VECTOR_DB_PATH: str = "storage/chroma_db"
    DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32

    # PostgreSQL Database Config
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgrespassword"
    POSTGRES_DB: str = "guardianeye"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str | None = None

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: str | None, info: ValidationInfo) -> Any:
        if isinstance(v, str) and v:
            return v
        data = info.data or {}
        user = data.get("POSTGRES_USER", "postgres")
        password = data.get("POSTGRES_PASSWORD", "postgrespassword")
        server = data.get("POSTGRES_SERVER", "localhost")
        port = data.get("POSTGRES_PORT", 5432)
        db = data.get("POSTGRES_DB", "guardianeye")
        return f"postgresql+asyncpg://{user}:{password}@{server}:{port}/{db}"

    # CORS Config
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if not v.startswith("["):
                return [i.strip() for i in v.split(",") if i.strip()]
            import json
            try:
                return json.loads(v)
            except ValueError:
                return []
        elif isinstance(v, list):
            return v
        return []

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

# Instantiate settings to be imported by application components
settings = Settings()
