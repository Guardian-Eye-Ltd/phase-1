import asyncio
import time
import av
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class CameraWorker:
    def __init__(self, camera_id: int, stream_url: str, target_fps: int = 15):
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.target_fps = target_fps
        self.is_running = False
        self.current_frame: Optional[bytes] = None
        self.status = "CONNECTING"
        self.fps = 0.0
        
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        self.frame_count = 0
        self._last_fps_time = time.time()
        self._frames_since_fps_calc = 0
        
        # Clients listening for frames
        # Use asyncio.Queue for broadcasting
        self.clients = set()
        
    async def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.status = "CONNECTING"
        self._task = asyncio.create_task(self._run())
        
    async def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.status = "OFFLINE"
        
    async def _run(self):
        reconnect_delay = 1.0
        max_delay = 30.0
        
        while self.is_running:
            try:
                self.status = "CONNECTING"
                # Using PyAV for decoding
                await self._decode_stream()
                
                # If we exit gracefully but still running, it means EOF or disconnect
                if self.is_running:
                    self.status = "RECONNECTING"
                    logger.warning(f"Camera {self.camera_id} stream ended. Reconnecting in {reconnect_delay}s")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(max_delay, reconnect_delay * 2)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.is_running:
                    self.status = "RECONNECTING"
                    logger.error(f"Camera {self.camera_id} stream error: {e}. Reconnecting in {reconnect_delay}s")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(max_delay, reconnect_delay * 2)

    async def _decode_stream(self):
        # PyAV needs to run in a thread to avoid blocking asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._decode_blocking)

    def _decode_blocking(self):
        options = {
            'rtsp_transport': 'tcp', 
            'stimeout': '5000000', # 5 seconds
            'max_delay': '500000',
            'flags': 'low_delay'
        }
        
        container = None
        try:
            logger.info(f"Camera {self.camera_id} connecting to {self.stream_url}")
            container = av.open(self.stream_url, options=options)
            video_stream = container.streams.video[0]
            video_stream.thread_type = 'AUTO'
            
            self.status = "LIVE"
            # Reset FPS calc
            self._last_fps_time = time.time()
            self._frames_since_fps_calc = 0
            
            # target frame time
            frame_time = 1.0 / self.target_fps if self.target_fps > 0 else 0
            last_emit = 0
            
            for frame in container.decode(video_stream):
                if not self.is_running:
                    break
                    
                now = time.time()
                
                # Skip frames if we are decoding faster than target FPS (unlikely for live, but good for throttling)
                if now - last_emit >= frame_time:
                    # Convert to JPEG
                    img = frame.to_image()
                    import io
                    buf = io.BytesIO()
                    # Quality can be adjusted to save bandwidth
                    img.save(buf, format='JPEG', quality=60)
                    jpeg_bytes = buf.getvalue()
                    
                    self.current_frame = jpeg_bytes
                    self.frame_count += 1
                    self._frames_since_fps_calc += 1
                    
                    # Update FPS every second
                    if now - self._last_fps_time >= 1.0:
                        self.fps = self._frames_since_fps_calc / (now - self._last_fps_time)
                        self._last_fps_time = now
                        self._frames_since_fps_calc = 0
                        
                    last_emit = now
                    
                    # Notify clients
                    self._notify_clients(jpeg_bytes)
                    
        except Exception as e:
            logger.error(f"Decoder error for camera {self.camera_id}: {e}")
            raise
        finally:
            if container:
                container.close()

    def _notify_clients(self, frame_bytes: bytes):
        for q in list(self.clients):
            try:
                # If queue is full, drop the oldest frame to maintain low latency
                if q.full():
                    q.get_nowait()
                q.put_nowait(frame_bytes)
            except Exception:
                pass

    def add_client(self, queue: asyncio.Queue):
        self.clients.add(queue)
        
    def remove_client(self, queue: asyncio.Queue):
        if queue in self.clients:
            self.clients.remove(queue)
