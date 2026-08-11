from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from datetime import datetime, timezone
from app.models.camera import Camera
from app.schemas.camera import CameraCreate, CameraUpdate

class CameraService:
    """
    Service layer providing business logic for camera device lifecycle management,
    status updates, and database persistence.
    """

    @staticmethod
    async def get_camera(db: AsyncSession, camera_id: int) -> Camera | None:
        """Fetch camera entity by primary key ID."""
        statement = select(Camera).where(Camera.id == camera_id)
        result = await db.execute(statement)
        return result.scalars().first()

    @staticmethod
    async def get_camera_by_code(db: AsyncSession, camera_code: str) -> Camera | None:
        """Fetch camera entity by unique camera code."""
        statement = select(Camera).where(Camera.camera_code == camera_code)
        result = await db.execute(statement)
        return result.scalars().first()

    @staticmethod
    async def list_cameras(
        db: AsyncSession, 
        skip: int = 0, 
        limit: int = 100
    ) -> tuple[list[Camera], int]:
        """
        Fetch paginated camera list alongside total record count.
        """
        count_stmt = select(func.count(Camera.id))
        total_res = await db.execute(count_stmt)
        total_count = total_res.scalar_one()

        list_stmt = select(Camera).order_by(Camera.id.asc()).offset(skip).limit(limit)
        items_res = await db.execute(list_stmt)
        cameras = list(items_res.scalars().all())

        return cameras, total_count

    @staticmethod
    async def create_camera(db: AsyncSession, camera_in: CameraCreate) -> Camera:
        """
        Provision a new camera stream endpoint.
        
        Raises:
            ValueError: If camera code is already registered.
        """
        existing = await CameraService.get_camera_by_code(db, camera_in.camera_code)
        if existing:
            raise ValueError(f"Camera code '{camera_in.camera_code}' is already registered.")

        db_camera = Camera(
            name=camera_in.name,
            camera_code=camera_in.camera_code,
            location=camera_in.location,
            description=camera_in.description,
            protocol=camera_in.protocol,
            stream_url=camera_in.stream_url,
            username=camera_in.username,
            password=camera_in.password,
            is_active=True,
            status="OFFLINE"
        )
        db.add(db_camera)
        await db.commit()
        await db.refresh(db_camera)
        return db_camera

    @staticmethod
    async def update_camera(db: AsyncSession, camera_id: int, camera_in: CameraUpdate) -> Camera:
        """
        Update properties of an existing camera record.
        
        Raises:
            ValueError: If camera does not exist or duplicate camera code occurs.
        """
        camera = await CameraService.get_camera(db, camera_id)
        if not camera:
            raise ValueError(f"Camera with ID {camera_id} not found.")

        if camera_in.camera_code and camera_in.camera_code != camera.camera_code:
            existing = await CameraService.get_camera_by_code(db, camera_in.camera_code)
            if existing:
                raise ValueError(f"Camera code '{camera_in.camera_code}' is already taken.")

        update_data = camera_in.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(camera, key, value)

        await db.commit()
        await db.refresh(camera)
        return camera

    @staticmethod
    async def delete_camera(db: AsyncSession, camera_id: int) -> bool:
        """
        Delete a camera record from the database.
        
        Raises:
            ValueError: If camera ID is not found.
        """
        camera = await CameraService.get_camera(db, camera_id)
        if not camera:
            raise ValueError(f"Camera with ID {camera_id} not found.")

        await db.delete(camera)
        await db.commit()
        return True

    @staticmethod
    async def activate_camera(db: AsyncSession, camera_id: int) -> Camera:
        """Enable an inactive camera stream."""
        camera = await CameraService.get_camera(db, camera_id)
        if not camera:
            raise ValueError(f"Camera with ID {camera_id} not found.")

        camera.is_active = True
        camera.status = "CONNECTING"
        await db.commit()
        await db.refresh(camera)
        return camera

    @staticmethod
    async def deactivate_camera(db: AsyncSession, camera_id: int) -> Camera:
        """Disable a camera stream."""
        camera = await CameraService.get_camera(db, camera_id)
        if not camera:
            raise ValueError(f"Camera with ID {camera_id} not found.")

        camera.is_active = False
        camera.status = "OFFLINE"
        await db.commit()
        await db.refresh(camera)
        return camera

    @staticmethod
    async def update_camera_status(
        db: AsyncSession, 
        camera_id: int, 
        status: str, 
        last_seen_at: datetime | None = None
    ) -> Camera:
        """Update operational status and last_seen timestamp."""
        camera = await CameraService.get_camera(db, camera_id)
        if not camera:
            raise ValueError(f"Camera with ID {camera_id} not found.")

        camera.status = status
        if last_seen_at:
            camera.last_seen_at = last_seen_at
        elif status == "ONLINE":
            camera.last_seen_at = datetime.now(timezone.utc)

        await db.commit()
        await db.refresh(camera)
        return camera
