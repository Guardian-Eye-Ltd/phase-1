import asyncio
from app.database.session import SessionLocal
from sqlalchemy import select
from app.models.camera import Camera

async def update_camera():
    async with SessionLocal() as session:
        stmt = select(Camera).where(Camera.id == 1)
        res = await session.execute(stmt)
        camera = res.scalar_one_or_none()
        if camera:
            camera.stream_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4"
            await session.commit()
            print("Updated camera 1 stream URL to TearsOfSteel.mp4")
        else:
            print("Camera 1 not found")

if __name__ == "__main__":
    asyncio.run(update_camera())
