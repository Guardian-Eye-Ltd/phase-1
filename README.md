# GuardianEye

GuardianEye is an AI-powered intelligent video surveillance, investigation, and digital forensics platform. The system is designed to analyze CCTV footage, identify relevant forensic events (objects, people, activities), and allow natural language semantic searches against the video evidence.

## Architecture & Technology Stack

The project uses a modern decoupled architecture:

### Frontend
- **Framework:** Next.js 14 (App Router)
- **Language:** TypeScript
- **Styling:** TailwindCSS
- **Icons:** Lucide React
- **Data Fetching:** Axios

### Backend & AI Engine
- **Framework:** FastAPI (Python)
- **Database:** SQLite (aiosqlite) with SQLAlchemy (Asynchronous ORM)
- **Authentication:** JWT (JSON Web Tokens)
- **Computer Vision Pipeline:** 
  - **Detections:** YOLOv8 (ultralytics) / OpenCV
  - **Tracking:** Custom Hybrid IoU + Centroid Tracker
- **Semantic Search (Phase 1C):**
  - **Vision-Language Model (VLM):** BLIP via Hugging Face `transformers`
  - **Vector Database:** ChromaDB
  - **Embeddings:** `sentence-transformers`

---

## Core Features

- **Evidence Ingestion:** Secure upload, SHA-256 cryptographic hashing, and duplicate detection.
- **Computer Vision Pipeline:** Automated frame sampling, motion filtering, YOLO-based object detection, and multi-object tracking.
- **Forensic Keyframe Extraction:** Intelligent extraction of keyframes based on track entries, exits, and visual anomalies.
- **Action & Interaction Detection:** Captures spatial proximity and interaction between tracked entities.
- **Vision-Language Analysis:** Natural language visual descriptions of keyframes using advanced VLMs.
- **Semantic Hybrid Search:** Search video events using natural language (e.g., *"Find a person carrying a backpack near a vehicle"*).

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Backend Setup

Open a terminal and navigate to the `backend` directory:

```bash
cd backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install all required Python dependencies
pip install -r requirements.txt
```

#### Running the Backend

```bash
# Start the FastAPI server using Uvicorn

uvicorn main:app --host 0.0.0.0 --port 8000 --reload   
# OR alternatively:
python main.py
```
The backend API will be running at `http://localhost:8000`. 
Interactive API documentation is available at `http://localhost:8000/docs`.

### 2. Frontend Setup

Open a new terminal window and navigate to the `frontend` directory:

```bash
cd frontend

# Install Node.js dependencies
npm install
```

#### Running the Frontend

```bash
# Start the Next.js development server
npm run dev
```
The frontend UI will be running at `http://localhost:3000`.

---

## Default Users

When the database is initialized, two default accounts are generated for testing purposes:

- **Admin Account:**
  - Username: `admin`
  - Password: `Admin123!`

- **Investigator Account:**
  - Username: `investigator`
  - Password: `Investigator123!`

---

## Project Structure

```
phase-1/
├── backend/
│   ├── app/                      # Main FastAPI application logic (routes, services, models)
│   ├── storage/                  # Local storage for evidence, chroma vector db, and derived data
│   ├── alembic/                  # Database migration scripts
│   ├── main.py                   # FastAPI application entrypoint
│   └── requirements.txt          # Python dependencies
├── frontend/                     # Next.js web application
│   ├── app/                      # Next.js page routing
│   ├── components/               # Reusable React components
│   ├── services/                 # Axios API clients
│   └── types/                    # TypeScript interfaces
├── docker-compose.yml            
└── README.md                     # You are here!
```
