import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.future import select
from app.database.base import Base
from app.database.session import engine, SessionLocal
from app.models.role import Role
from app.models.user import User
from app.models.camera import Camera
from app.models.alert import Alert
from app.models.incident import Incident
from app.models.observation import VisualAttributeObservation, TrackAttributeAggregate
from app.models.face import FaceObservation
from app.authentication.password import hash_password

logger = logging.getLogger("guardianeye.init_db")

DEFAULT_ROLES = [
    {"id": 1, "role_name": "Admin"},
    {"id": 2, "role_name": "Operator"},
    {"id": 3, "role_name": "Investigator"},
    {"id": 4, "role_name": "Viewer"}
]

async def auto_migrate_schema(db_engine: AsyncEngine = engine):
    """
    Safely adds missing diagnostic counter columns to existing analysis_jobs tables if they don't exist.
    Prevents OperationalError on existing SQLite/PostgreSQL databases without manual migrations.
    """
    new_columns = [
        ("frames_sampled", "INTEGER DEFAULT 0 NOT NULL"),
        ("frames_with_detections", "INTEGER DEFAULT 0 NOT NULL"),
        ("raw_detections", "INTEGER DEFAULT 0 NOT NULL"),
        ("stored_detections", "INTEGER DEFAULT 0 NOT NULL"),
        ("unique_tracks", "INTEGER DEFAULT 0 NOT NULL"),
        ("unique_keyframes", "INTEGER DEFAULT 0 NOT NULL"),
        ("activity_intervals_count", "INTEGER DEFAULT 0 NOT NULL"),
        ("source_frames", "INTEGER DEFAULT 0 NOT NULL"),
        ("processed_frames", "INTEGER DEFAULT 0 NOT NULL"),
        ("confidence_filtered_detections", "INTEGER DEFAULT 0 NOT NULL"),
        ("stats_schema_version", "INTEGER DEFAULT 0 NOT NULL"),
        ("class_filtered_detections", "INTEGER DEFAULT 0 NOT NULL"),
        ("visual_attribute_observations", "INTEGER DEFAULT 0 NOT NULL"),
        ("vehicle_attribute_observations", "INTEGER DEFAULT 0 NOT NULL"),
        ("plate_observations", "INTEGER DEFAULT 0 NOT NULL"),
        ("track_attribute_aggregates", "INTEGER DEFAULT 0 NOT NULL"),
        ("withheld_attributes", "INTEGER DEFAULT 0 NOT NULL"),
        ("event_candidates", "INTEGER DEFAULT 0 NOT NULL"),
        ("semantic_documents", "INTEGER DEFAULT 0 NOT NULL"),
        ("face_stage", "VARCHAR(20) DEFAULT 'NOT_RUN' NOT NULL"),
        ("face_stage_detail", "TEXT"),
    ]
    # One transaction per statement: on Postgres a failed ALTER ("column exists")
    # aborts the whole transaction, which would silently skip every later column.
    for col_name, col_def in new_columns:
        try:
            async with db_engine.begin() as conn:
                await conn.execute(text(f"ALTER TABLE analysis_jobs ADD COLUMN {col_name} {col_def}"))
            logger.info(f"Schema migration: Added column '{col_name}' to analysis_jobs table.")
        except Exception:
            pass  # Column already exists


# Idempotent data-integrity migrations, applied in order on every startup.
_INTEGRITY_MIGRATIONS = [
    (
        "dedupe vlm_observations per (job, keyframe)",
        "DELETE FROM vlm_observations WHERE id NOT IN ("
        " SELECT MIN(id) FROM vlm_observations GROUP BY analysis_job_id, keyframe_id)",
    ),
    (
        "unique vlm_observations (job, keyframe)",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_vlm_job_keyframe "
        "ON vlm_observations (analysis_job_id, keyframe_id)",
    ),
    (
        "unique track_attribute_aggregates (job, track, attribute)",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_taa_job_track_attr "
        "ON track_attribute_aggregates (analysis_job_id, track_number, attribute)",
    ),
    (
        "index detections (job, track, timestamp)",
        "CREATE INDEX IF NOT EXISTS ix_det_job_track_ts "
        "ON detections (analysis_job_id, track_id, timestamp)",
    ),
    (
        # Older runs stored detection-count template text as VLM output. Mark
        # those search documents as system-generated so they stop claiming a
        # VLM saw something.
        "relabel legacy template documents that claimed VLM provenance",
        "UPDATE forensic_documents SET source_type = 'SYSTEM_GENERATED' "
        "WHERE source_type = 'VLM_OBSERVATION' AND EXISTS ("
        " SELECT 1 FROM vlm_observations v"
        " WHERE v.analysis_job_id = forensic_documents.analysis_job_id"
        " AND v.keyframe_id = forensic_documents.keyframe_id"
        " AND v.model_name = 'VLM_Heuristic_Fallback')",
    ),
]


async def apply_integrity_migrations(db_engine: AsyncEngine = engine):
    for name, sql in _INTEGRITY_MIGRATIONS:
        try:
            async with db_engine.begin() as conn:
                res = await conn.execute(text(sql))
            if res.rowcount and res.rowcount > 0:
                logger.info(f"Integrity migration '{name}': {res.rowcount} row(s) affected.")
        except Exception as e:
            logger.warning(f"Integrity migration '{name}' skipped: {e}")


async def recover_interrupted_jobs():
    """
    Analysis runs execute in-process, so at startup nothing can still be running.
    Any job left QUEUED/PROCESSING was killed by a crash or restart; mark it
    FAILED so it stops looking active. Assumes a single worker process.
    """
    from app.models.analysis import AnalysisJob, JobStatus
    from app.models.evidence import Evidence, EvidenceStatus

    async with SessionLocal() as session:
        stuck = (await session.execute(
            select(AnalysisJob).where(AnalysisJob.status.in_([JobStatus.QUEUED, JobStatus.PROCESSING]))
        )).scalars().all()
        if not stuck:
            return
        for job in stuck:
            job.status = JobStatus.FAILED
            job.error_message = "Interrupted: the server stopped before this analysis finished."

        for ev_id in {j.evidence_id for j in stuck}:
            ev = (await session.execute(select(Evidence).where(Evidence.id == ev_id))).scalars().first()
            if ev is None or ev.status != EvidenceStatus.PROCESSING:
                continue
            has_completed = (await session.execute(
                select(AnalysisJob.id).where(
                    AnalysisJob.evidence_id == ev_id,
                    AnalysisJob.status == JobStatus.COMPLETED,
                ).limit(1)
            )).first() is not None
            ev.status = EvidenceStatus.COMPLETED if has_completed else EvidenceStatus.FAILED

        await session.commit()
        logger.warning(f"Recovered {len(stuck)} interrupted analysis job(s): {[j.id for j in stuck]}")


async def create_tables(db_engine: AsyncEngine = engine):
    """
    Create database tables defined in SQLAlchemy Base metadata and apply safe schema migrations.
    """
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await auto_migrate_schema(db_engine)
    await apply_integrity_migrations(db_engine)
    logger.info("Database tables and schema migrations initialized successfully.")

async def seed_roles():
    """
    Seed initial system roles (Admin, Operator, Investigator, Viewer) if not present.
    """
    async with SessionLocal() as session:
        for role_data in DEFAULT_ROLES:
            stmt = select(Role).where(Role.role_name == role_data["role_name"])
            res = await session.execute(stmt)
            existing_role = res.scalars().first()
            if not existing_role:
                role = Role(id=role_data["id"], role_name=role_data["role_name"])
                session.add(role)
                logger.info(f"Seeded default role: {role_data['role_name']}")
        await session.commit()

async def seed_admin_user():
    """
    Seed initial super administrator user account if no admin exists.
    """
    async with SessionLocal() as session:
        stmt = select(User).where(User.username == "admin")
        res = await session.execute(stmt)
        admin_user = res.scalars().first()
        if not admin_user:
            admin_role_stmt = select(Role).where(Role.role_name == "Admin")
            admin_role_res = await session.execute(admin_role_stmt)
            admin_role = admin_role_res.scalars().first()
            
            if admin_role:
                hashed_pw = hash_password("Admin123!")
                new_admin = User(
                    full_name="System Administrator",
                    username="admin",
                    email="admin@guardianeye.io",
                    password_hash=hashed_pw,
                    role_id=admin_role.id,
                    is_active=True
                )
                session.add(new_admin)
                await session.commit()
                logger.info("Seeded initial system admin user (username: admin).")

async def seed_investigator_user():
    """
    Seed initial investigator user account if no investigator exists.
    """
    async with SessionLocal() as session:
        stmt = select(User).where(User.username == "investigator")
        res = await session.execute(stmt)
        inv_user = res.scalars().first()
        if not inv_user:
            inv_role_stmt = select(Role).where(Role.role_name == "Investigator")
            inv_role_res = await session.execute(inv_role_stmt)
            inv_role = inv_role_res.scalars().first()
            
            if inv_role:
                hashed_pw = hash_password("Investigator123!")
                new_inv = User(
                    full_name="Lead Forensic Investigator",
                    username="investigator",
                    email="investigator@guardianeye.io",
                    password_hash=hashed_pw,
                    role_id=inv_role.id,
                    is_active=True
                )
                session.add(new_inv)
                await session.commit()
                logger.info("Seeded initial investigator user (username: investigator).")

from app.core.config import settings

async def seed_mock_data():
    """
    Seed mock cameras, alerts, and incidents for the dashboard.
    """
    async with SessionLocal() as session:
        # Define the demo cameras
        demo_cameras = [
            {"camera_code": "CAM-001", "name": "Main Entrance", "location": "Front Gate", "stream_url": getattr(settings, "DEMO_CAMERA_001_SOURCE", "rtsp://demo/1")},
            {"camera_code": "CAM-002", "name": "Parking Area", "location": "North Lot", "stream_url": getattr(settings, "DEMO_CAMERA_002_SOURCE", "rtsp://demo/2")},
            {"camera_code": "CAM-003", "name": "Street Perimeter", "location": "East Wall", "stream_url": getattr(settings, "DEMO_CAMERA_003_SOURCE", "rtsp://demo/3")},
            {"camera_code": "CAM-004", "name": "Building Corridor", "location": "Level 1", "stream_url": getattr(settings, "DEMO_CAMERA_004_SOURCE", "rtsp://demo/4")},
            {"camera_code": "CAM-005", "name": "Warehouse", "location": "Storage Bay", "stream_url": getattr(settings, "DEMO_CAMERA_005_SOURCE", "rtsp://demo/5")},
            {"camera_code": "CAM-006", "name": "Rear Perimeter", "location": "Loading Dock", "stream_url": getattr(settings, "DEMO_CAMERA_006_SOURCE", "rtsp://demo/6")},
        ]

        if settings.DEMO_STREAMS_ENABLED:
            for cam_data in demo_cameras:
                stmt = select(Camera).where(Camera.camera_code == cam_data["camera_code"])
                res = await session.execute(stmt)
                existing_cam = res.scalars().first()
                
                if existing_cam:
                    existing_cam.stream_url = cam_data["stream_url"]
                    existing_cam.name = cam_data["name"]
                    existing_cam.location = cam_data["location"]
                else:
                    new_cam = Camera(
                        name=cam_data["name"],
                        camera_code=cam_data["camera_code"],
                        location=cam_data["location"],
                        stream_url=cam_data["stream_url"],
                        is_active=True,
                        status="ONLINE"
                    )
                    session.add(new_cam)
            
            await session.commit()
            logger.info("Seeded demo cameras from configuration.")
            
            # Ensure at least one alert exists for demo
            stmt_alert = select(Alert)
            res_alert = await session.execute(stmt_alert)
            if not res_alert.scalars().first():
                stmt_cam1 = select(Camera).where(Camera.camera_code == "CAM-001")
                cam1 = (await session.execute(stmt_cam1)).scalars().first()
                if cam1:
                    alert1 = Alert(camera_id=cam1.id, alert_type="MOTION_DETECTED", severity="LOW", description="Motion at front gate")
                    session.add(alert1)
                    await session.commit()

async def init_db(db_engine: AsyncEngine = engine):
    """
    Complete database initialization pipeline: tables creation and role/user seeding.
    """
    await create_tables(db_engine)
    await recover_interrupted_jobs()
    await seed_roles()
    await seed_admin_user()
    await seed_investigator_user()
    await seed_mock_data()
