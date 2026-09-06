import logging
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.future import select
from app.database.base import Base
from app.database.session import engine, SessionLocal
from app.models.role import Role
from app.models.user import User
from app.models.camera import Camera
from app.models.alert import Alert
from app.models.incident import Incident
from app.authentication.password import hash_password

logger = logging.getLogger("guardianeye.init_db")

DEFAULT_ROLES = [
    {"id": 1, "role_name": "Admin"},
    {"id": 2, "role_name": "Operator"},
    {"id": 3, "role_name": "Investigator"},
    {"id": 4, "role_name": "Viewer"}
]

async def create_tables(db_engine: AsyncEngine = engine):
    """
    Create database tables defined in SQLAlchemy Base metadata.
    """
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized successfully.")

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
    await seed_roles()
    await seed_admin_user()
    await seed_investigator_user()
    await seed_mock_data()
