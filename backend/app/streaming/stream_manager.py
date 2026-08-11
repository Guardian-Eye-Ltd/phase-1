import asyncio
import logging
from typing import Dict
from app.streaming.camera_worker import CameraWorker
from app.core.config import settings

logger = logging.getLogger(__name__)

class StreamManager:
    def __init__(self):
        self._workers: Dict[int, CameraWorker] = {}
        self._lock = asyncio.Lock()
        
    async def get_or_create_worker(self, camera_id: int, stream_url: str) -> CameraWorker:
        async with self._lock:
            if camera_id not in self._workers:
                logger.info(f"Creating new CameraWorker for camera {camera_id}")
                worker = CameraWorker(
                    camera_id=camera_id, 
                    stream_url=stream_url, 
                    target_fps=settings.STREAM_TARGET_FPS
                )
                self._workers[camera_id] = worker
                await worker.start()
            return self._workers[camera_id]
            
    async def remove_worker_if_idle(self, camera_id: int):
        async with self._lock:
            worker = self._workers.get(camera_id)
            if worker and len(worker.clients) == 0:
                logger.info(f"No more clients for camera {camera_id}, scheduling shutdown")
                asyncio.create_task(self._delayed_shutdown(camera_id))
                
    async def _delayed_shutdown(self, camera_id: int):
        await asyncio.sleep(settings.STREAM_IDLE_TIMEOUT)
        async with self._lock:
            worker = self._workers.get(camera_id)
            if worker and len(worker.clients) == 0:
                logger.info(f"Idle timeout reached for camera {camera_id}, stopping worker")
                await worker.stop()
                del self._workers[camera_id]
                
    async def shutdown_all(self):
        async with self._lock:
            for worker in self._workers.values():
                await worker.stop()
            self._workers.clear()

stream_manager = StreamManager()
