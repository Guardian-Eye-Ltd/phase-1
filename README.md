# GuardianEye

GuardianEye is an AI-powered video surveillance, investigation, and digital-forensics platform. It ingests CCTV/video evidence, runs a computer-vision pipeline over it (detection, tracking, visual attributes, number plates, faces, keyframes), indexes the results, and lets investigators query the evidence in natural language.

The system reports **observable patterns, never conclusions**: every answer is traced to a specific analysis run, carries a confidence and an evidence status, and says "not determined" (with the reason) instead of guessing.

## Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Setup](#setup)
- [Configuration](#configuration)
- [Default Users](#default-users)
- [Using the Platform](#using-the-platform)
- [API Overview](#api-overview)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Known Limitations](#known-limitations)

---

## Features

### Evidence handling
- **Secure ingestion** — upload with SHA-256 hashing, duplicate detection, and chain-of-custody timestamps.
- **Audit log** — uploads, searches, face searches, and resets are all recorded.
- **Sealed manifests** — each analysis run writes a hashed manifest that is re-verified every time it is read.
- **Role-based access** — Admin, Operator, Investigator, and Viewer roles.

### Computer-vision pipeline
- **Detection and tracking** — frame sampling, motion filtering, YOLO11 object and pose detection, and multi-object tracking.
- **Visual attributes** — garment colour and type, headwear, carried items, vehicle colour and body style. Each attribute is voted across frames per track; contested or weak results are withheld rather than asserted.
- **Number plates (ALPR)** — per-vehicle OCR with EasyOCR, format-guided correction, and character-by-character voting. A plate is reported as `READ`, `UNCONFIRMED` (with the disputed characters), or `NOT_READ` (with the reason).
- **Face search by photo** — upload a photo and get possible-match candidates with timestamps and crops. Face embeddings are encrypted at rest and the query photo is never stored.
- **Keyframes, interactions, and events** — track entry/exit keyframes, spatial interactions, and rule-based event synthesis.

### Search and investigation
- **Semantic search** — natural-language queries such as *"find a person carrying a backpack near a vehicle"*, combining structured lookups with vector search over BLIP captions.
- **Structured answers** — plate and attribute questions (*"what is the vehicle number of the white car"*, *"find people wearing black"*) are answered from per-track consensus, not embeddings.
- **Investigation workspace** — intent-first queries (count, frame count, attribute search, plate lookup, event search, summary) answered by deterministic database queries and report templates. No LLM is involved in producing results.

### Operations
- **Live monitoring** — RTSP camera grid streamed over WebSocket.
- **Re-analysis** — evidence can be analysed many times; each run is isolated and selectable.
- **Admin reset** — wipe one evidence item's analysis, or all evidence, behind a typed confirmation.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14 (App Router), TypeScript, TailwindCSS, Lucide, Axios, React Hook Form, Recharts |
| Backend | FastAPI, Uvicorn, async SQLAlchemy 2.x |
| Database | SQLite (`aiosqlite`) for local development, PostgreSQL (`asyncpg`) in Docker |
| Auth | JWT (PyJWT), bcrypt |
| Detection | Ultralytics YOLO11n and YOLO11n-pose, OpenCV, PyAV |
| Attributes | CLIP (`openai/clip-vit-base-patch32`) plus deterministic HSV colour naming |
| Plates | EasyOCR |
| Faces | InsightFace `buffalo_l` (detection and recognition only) |
| Captions | BLIP (`Salesforce/blip-image-captioning-base`) |
| Search | ChromaDB, `sentence-transformers` (`all-MiniLM-L6-v2`) |

---

## Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Several GB of free disk space for model weights

### 1. Backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
```

Create `backend/.env` before the first run. Without `DATABASE_URL`, the backend tries to connect to a local PostgreSQL server; this line selects SQLite instead:

```env
DATABASE_URL=sqlite+aiosqlite:///./guardianeye.db
SECRET_KEY=replace-with-a-long-random-string
```

Start the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# or
python main.py
```

- API: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`

The database tables, roles, and default users are created on startup.

> **First-run downloads.** Model weights are fetched on first use: BLIP (~1 GB), InsightFace `buffalo_l` (~280 MB), EasyOCR (~100 MB), plus CLIP, the sentence-transformer, and the YOLO weights. The first analysis is therefore much slower than later ones.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI runs at `http://localhost:3000`.

### 3. Docker (optional)

Runs PostgreSQL, the backend, and the frontend together:

```bash
docker compose up --build
```

### 4. Demo cameras (optional, Windows)

To try live monitoring without real cameras, set `DEMO_STREAMS_ENABLED=true` in `backend/.env`, then serve the demo RTSP streams:

```powershell
cd backend/scripts
./mediamtx.exe
./start_demo_cameras.ps1
```

---

## Configuration

Settings are read from `backend/.env`; defaults live in [backend/app/core/config.py](backend/app/core/config.py). The most commonly changed keys:

| Key | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | PostgreSQL on localhost | Database connection string |
| `SECRET_KEY` | development value | JWT signing key; change it outside development |
| `FACE_EMBEDDING_KEY` | derived from `SECRET_KEY` | Fernet key for stored face embeddings |
| `FRAME_SAMPLE_FPS` | `2.0` | Frames per second sampled for analysis |
| `DETECTION_CONFIDENCE_THRESHOLD` | `0.25` | Minimum detection confidence kept |
| `ENABLE_ATTRIBUTE_CLASSIFICATION` | `true` | Garment and vehicle attributes |
| `ENABLE_ALPR` | `true` | Number-plate OCR |
| `FACE_RECOGNITION_ENABLED` | `true` | Face extraction and search |
| `FACE_MATCH_THRESHOLD` | `0.45` | Cosine similarity for a possible face match |
| `VLM_ENABLED` | `true` | BLIP keyframe captions |
| `MAX_UPLOAD_SIZE_MB` | `500` | Upload size limit |
| `DEMO_STREAMS_ENABLED` | `false` | Seed demo cameras CAM-001 to CAM-006 |
| `BACKEND_CORS_ORIGINS` | `localhost:3000` | Allowed frontend origins |

If you change `SECRET_KEY` without setting `FACE_EMBEDDING_KEY`, previously stored face embeddings can no longer be decrypted. Never commit `.env`.

---

## Default Users

Two accounts are created when the database is initialised:

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `Admin123!` |
| Investigator | `investigator` | `Investigator123!` |

These are for local testing. Change them before exposing the system to anyone else.

---

## Using the Platform

1. **Log in** at `http://localhost:3000` with one of the default accounts.
2. **Upload evidence** from the Evidence page. Analysis starts automatically; progress and per-stage timings are shown on the evidence page.
3. **Review the results** on the evidence detail page: video playback with overlays, keyframes, timeline, the *People & Vehicles* tab (every tracked entity with its attributes, confidence, and the reason for anything undetermined), and the *Face Search* tab.
4. **Search** from the Forensic workspace using natural language, or run a structured investigation in the Investigation workspace.
5. **Re-run analysis** from the evidence page when needed. Analyses run one at a time; later requests wait in a queue.

Admins can reset all evidence under **Settings → Danger Zone**. Users, roles, cameras, and the audit log are kept.

---

## API Overview

All routes are under `/api/v1`. The full reference is at `http://localhost:8000/docs`.

| Area | Examples |
|---|---|
| Auth | `POST /auth/login`, `GET /auth/me` |
| Evidence | `POST /evidence`, `GET /evidence`, download, delete |
| Analysis | `POST /evidence/{id}/analysis`, `POST /evidence/{id}/reprocess`, `GET /evidence/{id}/analysis-jobs`, `/detections`, `/tracks`, `/keyframes`, `/timeline`, `/statistics`, `/manifest`, `/entities` |
| Semantic | `POST /semantic/search` |
| Investigation | `POST /investigation/query` |
| Faces | `POST /evidence/{id}/faces/search`, `GET /evidence/{id}/faces` (Investigator and above) |
| Admin | `GET /admin/reset/preview`, `POST /admin/reset`, `POST /evidence/{id}/reset-analysis` (Admin only) |
| Other | audit, cameras, streams, alerts, incidents, health |

Evidence-scoped read routes accept an optional `?job_id=` to select a specific analysis run; without it, the latest completed run is used.

---

## Testing

```bash
cd backend
pytest
```

Tests run against a temporary database and storage directory, so they never touch your local data. Eleven legacy tests in `test_audit_scenarios.py` and `test_phase1c_semantic.py` are known to fail; they cover an unused older code path.

---

## Project Structure

```
phase-1/
├── backend/
│   ├── main.py                   # FastAPI entrypoint
│   ├── requirements.txt
│   ├── alembic/                  # Database migrations
│   ├── scripts/                  # mediamtx RTSP server + demo camera launcher
│   ├── storage/                  # Evidence, derived data, ChromaDB (created at runtime)
│   ├── tests/                    # pytest suites
│   └── app/
│       ├── core/                 # Settings, logging, permissions
│       ├── database/             # Engine, sessions, startup init + migrations
│       ├── authentication/       # JWT and password hashing
│       ├── models/               # SQLAlchemy ORM models
│       ├── schemas/              # Pydantic request/response schemas
│       ├── api/routes/           # REST endpoints
│       ├── services/
│       │   ├── pipeline/         # CV pipeline: detection, tracking, attributes, ALPR, faces
│       │   ├── semantic/         # Captioning, indexing, hybrid search
│       │   ├── agents/           # Investigation intent parsing, tools, reports
│       │   └── face/             # Face engine, embedding encryption, search
│       ├── streaming/            # RTSP camera workers
│       └── websocket/            # Live feed endpoints
├── frontend/
│   ├── app/                      # Next.js pages
│   ├── components/               # React components
│   ├── contexts/                 # Auth context
│   ├── services/                 # Axios API clients
│   └── types/                    # TypeScript interfaces
├── docker-compose.yml            # PostgreSQL + backend + frontend
├── CLAUDE.md                     # Detailed architecture reference
└── README.md
```

[CLAUDE.md](CLAUDE.md) holds the detailed architecture notes, design decisions, and change log.

---

## Known Limitations

- **Vehicle make and model are not available.** Zero-shot recognition was tested and was wrong on every vehicle, so it is disabled.
- **Colour, body style, plate OCR, and captions are best-effort.** They depend on resolution and viewing angle; small or distant vehicles produce no plate read, and captions are scene context only.
- **Track counts tend to run high.** At 2 fps the tracker can split one object into several tracks.
- **Face matches are candidates, not identifications.** The `buffalo_l` weights are licensed for non-commercial research only.
- **Single backend worker only.** Analysis runs in-process; running multiple workers would break job recovery and queueing.
- **Keyframe images are served without authentication.** This is an open security issue; do not expose the backend to an untrusted network.
- **Runs analysed on older versions** lack attributes, plates, and faces until the evidence is re-analysed.
