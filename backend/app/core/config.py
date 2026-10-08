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

    # Phase 1 Pipeline & Video Processing Config
    FRAME_SAMPLE_FPS: float = 2.0
    DETECTION_INTERVAL: int = 1
    KEYFRAME_INTERVAL: float = 10.0
    MAX_PROCESSING_RESOLUTION: int = 1280
    YOLO_MODEL_NAME: str = "yolo11n.pt"
    YOLO_POSE_MODEL_NAME: str = "yolo11n-pose.pt"
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.25
    DETECTION_IOU_THRESHOLD: float = 0.45
    # The detector runs at this floor so pre-threshold box counts are measurable;
    # the job's confidence threshold is then applied in Python.
    DETECTOR_RAW_CONFIDENCE_FLOOR: float = 0.05

    # CLIP Attribute Engine & ALPR Config
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    ENABLE_ATTRIBUTE_CLASSIFICATION: bool = True
    ENABLE_ALPR: bool = True

    # Phase 1C Semantic Intelligence & Forensic Thresholds Config
    VLM_ENABLED: bool = True
    VLM_MODEL_NAME: str = "Salesforce/blip-image-captioning-base"
    MAX_KEYFRAMES_FOR_VLM: int = 30
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_MODEL_VERSION: str = "1.0"
    INDEX_VERSION: str = "1.0"
    VECTOR_DB_TYPE: str = "chromadb"
    VECTOR_DB_PATH: str = "storage/chroma_db"
    DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32
    MIN_SEARCH_RELEVANCE: float = 0.22
    LOITERING_THRESHOLD_SECONDS: float = 12.0
    CONCEALMENT_DISTANCE_THRESHOLD: float = 0.12  # Normalized keypoint wrist-to-hip distance

    # Face recognition (InsightFace buffalo_l: SCRFD detection + ArcFace embeddings).
    # Output is always a similarity candidate, never an identification.
    FACE_RECOGNITION_ENABLED: bool = True
    FACE_MODEL_PACK: str = "buffalo_l"
    FACE_DET_SIZE: int = 640
    FACE_MAX_FRAMES_PER_TRACK: int = 3       # faces are extracted from the best N frames per person track
    FACE_MIN_SIZE_PX: int = 32               # measured in source-frame pixels, never after upscaling
    FACE_MIN_DET_SCORE: float = 0.60
    FACE_MIN_SHARPNESS: float = 10.0         # variance of Laplacian on a 112x112 face crop
    FACE_MATCH_THRESHOLD: float = 0.45       # cosine similarity; same person ~0.9, strangers ~0.0 on probe data
    FACE_QUERY_MAX_BYTES: int = 10 * 1024 * 1024
    # Fernet key for embeddings at rest. If empty, a key is derived from
    # SECRET_KEY (development only — set this explicitly in production).
    FACE_EMBEDDING_KEY: str = ""

    # Agent & LLM Provider Config
    LLM_PROVIDER: str = "ollama"  # "ollama", "openai", "groq", "heuristics"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: str = "llama3"
    GROQ_API_KEY: str = ""
    GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"
    MAX_AGENT_STEPS: int = 10
    INVESTIGATION_TIMEOUT_SECONDS: float = 60.0

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
