# GuardianEye Demo Cameras Setup

This guide explains how to set up and run the local demonstration environment to simulate a multi-camera RTSP CCTV network using standard MP4 video files.

## Architecture Overview

To faithfully recreate a real CCTV environment without risking hardcoded production ingestion, GuardianEye uses **MediaMTX** (a zero-dependency RTSP server) and **FFmpeg**.

1. **FFmpeg** reads dummy MP4 files in an infinite loop (`-stream_loop -1`).
2. It pushes the streams to **MediaMTX** over RTSP (`rtsp://127.0.0.1:8554/camera01`).
3. The GuardianEye **FastAPI PyAV Worker** ingests the RTSP streams from MediaMTX.
4. FastAPI transcodes frames to JPEGs and pipes them via **WebSocket** to the **Next.js Dashboard**.

*Note: For the local demo, streams are ingested over standard RTSP. GuardianEye natively supports RTSPS/TLS when configured for production.*

## Prerequisites

1. **FFmpeg**: Must be installed and accessible in your system `PATH`.
2. **Demo Videos**: You need 6 standard MP4 files (H.264 encoded).

## Setup Instructions

### 1. Place the Video Files
Create a folder called `demo_streams` in the `backend` directory.
Place 6 video files inside it, named exactly:
- `backend/demo_streams/camera01.mp4`
- `backend/demo_streams/camera02.mp4`
- `backend/demo_streams/camera03.mp4`
- `backend/demo_streams/camera04.mp4`
- `backend/demo_streams/camera05.mp4`
- `backend/demo_streams/camera06.mp4`

### 2. Enable Demo Config
Ensure your `backend/.env` file has the following flag set:
```env
DEMO_STREAMS_ENABLED=true
```

### 3. Start the Demo Cameras
Run the provided PowerShell script. It will automatically download MediaMTX, verify your files, and start all streams.
```powershell
cd backend/scripts
./start_demo_cameras.ps1
```

The script will output the RTSP URLs, e.g.: `rtsp://127.0.0.1:8554/camera01`

### 4. Restart Backend
Because the database seeds on application startup, restart the FastAPI server:
```bash
python -m uvicorn main:app --reload
```

## Testing Reconnection Logic

The architecture is highly resilient to camera drop-outs. To test:
1. Open the Live Monitoring Dashboard (`http://localhost:3000/realtime-monitoring`).
2. Identify the FFmpeg process running `camera03.mp4` (or any specific camera).
3. Kill that specific FFmpeg process.
4. Watch the Next.js UI transition from `LIVE` to `RECONNECTING Attempt N`.
5. Start the stream again (or re-run the script). The system will automatically recover to `LIVE` without needing a FastAPI restart!

## Troubleshooting

- **Black Video Card / Stuck CONNECTING**: Ensure MediaMTX is running. Check the FastAPI logs to see if PyAV throws a decoder error. If so, your MP4 file might be encoded with a codec other than H.264.
- **Script Fails**: Ensure `ffmpeg` is actually in your `PATH` by running `ffmpeg -version` in a new terminal.
- **Adding Another Camera**: 
  1. Add `camera07.mp4`
  2. Update the `start_demo_cameras.ps1` loop condition (`$i -le 7`).
  3. Update `.env` with `DEMO_CAMERA_007_SOURCE`.
  4. Update `init_db.py` to seed `CAM-007`.
