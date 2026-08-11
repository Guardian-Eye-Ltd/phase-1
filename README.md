# GuardianEye

GuardianEye is an AI-Powered Surveillance Platform.

## 🚀 Tech Stack

### Backend
*   **Framework**: FastAPI
*   **Database**: PostgreSQL (via SQLAlchemy & asyncpg)
*   **Authentication**: JWT & bcrypt
*   **Video Processing**: OpenCV & PyAV
*   **Real-time Communication**: WebSockets
*   **Testing**: Pytest

### Frontend
*   **Framework**: Next.js 14 (React 18)
*   **Styling**: Tailwind CSS
*   **Icons**: Lucide React
*   **Charts**: Recharts
*   **Form Handling**: React Hook Form
*   **HTTP Client**: Axios

## 📁 Project Structure

*   `/backend` - Contains the FastAPI backend application.
    *   `/app` - Main application code.
    *   `/tests` - Pytest tests.
    *   `/alembic` - Database migrations.
*   `/frontend` - Contains the Next.js frontend application.
    *   `/app` - Next.js App Router code.
    *   `/components` - Reusable UI components.
    *   `/services` - API integration.
    *   `/contexts` - React contexts for state management.

 ##Database Schema explanation.
 
 Role: Defines authorization levels (e.g., admin, operator) for access control.

User: Represents the individuals who can log into the platform. A user is tied to a specific Role.

Camera: Stores details about surveillance devices, including their RTSP stream URLs, credentials, locations, and current health status.

Alert: Records specific events flagged by a camera (like motion detection or unauthorized access) with a severity level and timestamp. Linked directly to a Camera.

Incident: Captures user-reported or escalated broader events. It tracks a lifecycle (OPEN, IN_PROGRESS, CLOSED) and can be linked to the User who reported it.


## 🛠️ Getting Started

### Backend Setup
1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```
2.  Install dependencies (preferably in a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up environment variables using `.env.example` as a template.
4.  Run the backend development server:
    ```bash
    python main.py
    # or
    uvicorn main:app --reload
    ```
    The API will be available at `http://localhost:8000`.

### Frontend Setup
1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Run the frontend development server:
    ```bash
    npm run dev
    ```
    The application will be available at `http://localhost:3000`.
