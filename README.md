# GuardianEye — Intelligent Video Digital Forensics Platform (Phase 1 Foundation)

**GuardianEye** is an enterprise-grade, intelligent video digital forensics platform designed for cyber security investigators, law enforcement, and forensic analysts to ingest, verify, stream, and analyze CCTV video evidence.

---

## 🏛️ Phase 1 Architecture Overview

Phase 1 establishes a modular, secure software foundation designed specifically to support future computer vision and AI video analysis engines (YOLO, DeepSORT/Re-ID, Qwen-VL/LLaVA, ChromaDB vector search) without requiring architectural redesigns.

```
                  ┌───────────────────────────────────────────────────────────┐
                  │                 Next.js 14 Frontend UI                    │
                  │   (Dark Cybersecurity Forensics Theme, Vault, Player)     │
                  └─────────────────────────────┬─────────────────────────────┘
                                                │ REST API / HTTP Range
                                                ▼
                  ┌───────────────────────────────────────────────────────────┐
                  │                 FastAPI Backend Engine                    │
                  │   - JWT Auth & Role-Based Access Control (RBAC)           │
                  │   - Evidence Ingestion & SHA-256 Integrity Service        │
                  │   - HTTP Range Video Streaming Engine                     │
                  │   - Background Pipeline Queue & Audit Logging             │
                  └──────┬──────────────────────┬──────────────────────┬──────┘
                         │                      │                      │
                         ▼                      ▼                      ▼
           ┌────────────────────────┐┌──────────────────────┐┌───────────────────────┐
           │ PostgreSQL Database    ││ Local Storage Vault  ││ Pluggable AI Modules  │
           │ (Evidence & Audit Logs)││ (Immutable Originals)││ (Abstract Interfaces) │
           └────────────────────────┘└──────────────────────┘└───────────────────────┘
```

---

## 🔒 Key Forensics Features & Design Principles

1. **SHA-256 Chain of Custody Integrity**:
   - Every uploaded CCTV video is treated as an immutable evidence original.
   - SHA-256 cryptographic hash is calculated via streaming chunks upon ingestion and saved to the database.
   - Automatic duplicate detection alerts investigators when identical evidence files are uploaded.

2. **Secure Video Streaming**:
   - Supports HTTP Range requests (`bytes=0-`) to enable smooth seeking and frame scrubbing in high-definition CCTV footage without full memory loading.

3. **Pluggable AI Module Architecture**:
   - Defines standard abstract interfaces (`BaseDetectionService`, `BaseTrackingService`, `BaseTimelineService`, `BaseVLMService`, `BaseSearchService`) allowing future AI models to be plugged in seamlessly.

4. **Security & Path Traversal Defense**:
   - Strict UUID file renaming prevents directory traversal (`../`) and file overwrite attacks.
   - Strict extension whitelist (`.mp4`, `.webm`, `.avi`, `.mov`, `.mkv`).

---

## 📂 System Project Structure

```
phase-1/
├── backend/
│   ├── app/
│   │   ├── api/routes/          # REST API endpoints (auth, evidence, audit, users)
│   │   ├── core/                # JWT security, password hashing, environment config
│   │   ├── database/            # SQLAlchemy async session & table seeders
│   │   ├── models/              # DB Models (User, Role, Evidence, AuditLog)
│   │   ├── schemas/             # Pydantic validation schemas
│   │   └── services/            # Storage, Metadata, Audit, Evidence, and AI Interfaces
│   ├── storage/                 # Local filesystem evidence vault (gitignored)
│   ├── tests/                   # Pytest automated test suite
│   ├── main.py                  # FastAPI application entry point
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app/                     # Next.js App Router (login, dashboard, evidence, audit)
│   ├── components/              # UI components & Dashboard Layout
│   ├── contexts/                # AuthContext (JWT state)
│   ├── services/                # API client & Evidence services
│   ├── types/                   # TypeScript evidence & audit interfaces
│   └── Dockerfile
└── docker-compose.yml           # Multi-container orchestration
```

---

## 🚀 Quickstart Guide

### Prerequisites
- Node.js v18+ & npm
- Python 3.10+
- PostgreSQL 15 (or Docker)

### Option A: Local Development Setup

#### 1. Database Setup
Ensure PostgreSQL is running locally on port 5432 and create a database named `guardianeye`:
```bash
createdb -U postgres guardianeye
```

#### 2. Backend Setup
```bash
cd backend
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
*The database tables and default investigator accounts will be seeded automatically on startup.*

#### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

### Option B: Docker Compose Deployment

Run the entire full-stack application (PostgreSQL + FastAPI + Next.js) with a single command:
```bash
docker-compose up --build
```
Access the application at:
- **Frontend Dashboard**: `http://localhost:3000`
- **FastAPI OpenAPI Docs**: `http://localhost:8000/docs`

---

## 🔐 Default Demo Credentials

| Role | Username | Password | Email |
| :--- | :--- | :--- | :--- |
| **Investigator** | `investigator` | `Investigator123!` | `investigator@guardianeye.io` |
| **Administrator** | `admin` | `Admin123!` | `admin@guardianeye.io` |

---

## 🧪 Running Automated Tests

Run the backend Pytest test suite covering authentication, file upload integrity, duplicate hash detection, invalid format rejection, and audit logs:
```bash
cd backend
python -m pytest -v tests/test_evidence_and_auth.py
```

---

## 🛰️ Key API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/auth/login` | Authenticate investigator and issue JWT token |
| `POST` | `/api/v1/evidence/upload` | Upload CCTV video file & calculate SHA-256 hash |
| `GET` | `/api/v1/evidence` | List evidence vault files with search and filters |
| `GET` | `/api/v1/evidence/{id}` | Get detailed evidence metadata |
| `GET` | `/api/v1/evidence/{id}/stream`| HTTP Range video streaming endpoint |
| `POST` | `/api/v1/evidence/{id}/analysis`| Enqueue 8-stage post-event video analysis job |
| `GET` | `/api/v1/evidence/analysis/{job_id}`| Real-time pipeline stage & percentage progress |
| `POST` | `/api/v1/evidence/analysis/{job_id}/cancel`| Cancel active video analysis job |
| `GET` | `/api/v1/evidence/{id}/detections`| List object/person detections & bounding boxes |
| `GET` | `/api/v1/evidence/{id}/tracks`| Tracked entity summaries & lifetime metrics |
| `GET` | `/api/v1/evidence/{id}/keyframes`| Extracted keyframe images & SHA-256 hashes |
| `GET` | `/api/v1/evidence/{id}/timeline`| Chronological forensic observations timeline |
| `GET` | `/api/v1/evidence/{id}/activity`| Motion activity intervals |
| `GET` | `/api/v1/evidence/{id}/manifest`| Machine-readable JSON analysis manifest |
| `GET` | `/api/v1/audit` | Retrieve chain-of-custody audit logs |

---

## 🔮 Phase 1C AI Extension Roadmap

- **Phase 1C**: Qwen-VL / LLaVA Vision-Language Model Investigation Assistant, ChromaDB Vector Evidence Indexing, Court-Admissible PDF Forensic Export.
