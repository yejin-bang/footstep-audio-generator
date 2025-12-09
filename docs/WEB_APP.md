# Web Application

## 🌐 Overview

A browser-based interface for the footstep audio generation pipeline, providing an intuitive way to upload videos and download generated results.

**Status:** ⚠️ **In Development** - Frontend and backend partially implemented

---

## Current Implementation

### ✅ Completed Features

**Frontend (React + Vite + Tailwind CSS):**
- ✅ Video upload component with drag-and-drop
- ✅ File validation (size: 500MB max, type: mp4/avi/mov)
- ✅ Real-time processing status (6-stage pipeline visualization)
- ✅ Results viewer with side-by-side video comparison
- ✅ Download buttons for generated audio and video
- ✅ Responsive UI with modern design

**Backend (FastAPI):**
- ✅ File upload endpoint (`POST /upload`)
- ✅ Status polling endpoint (`GET /status/{task_id}`)
- ✅ Video preview endpoints (`GET /preview/{task_id}/original` and `/generated`)
- ✅ Download endpoints (`GET /download/{task_id}/video` and `/audio`)
- ✅ CORS middleware configured
- ✅ Video streaming with Accept-Ranges headers

---

## 🚧 Pending Implementation

### ⚠️ Critical (Blocking Web App Launch)

1. **Celery Task Integration (`web/backend/tasks.py`)**
   - Currently a placeholder
   - Needs to integrate with `src.main_pipeline.py`
   - Must update task state at each pipeline stage
   - Save results to `web/backend/results/{task_id}/`

2. **Redis Setup**
   - Required for Celery task queue
   - Not yet configured

### 📋 Nice-to-Have

- Database for job history
- User authentication
- Batch processing support
- WebSocket for real-time updates (instead of polling)
- Result expiration and cleanup

---

## 🏗️ Architecture

```
┌─────────────────┐
│  React Frontend │  (Port 5173)
│  - Upload UI    │
│  - Status Poll  │
│  - Result View  │
└────────┬────────┘
         │ HTTP/REST
         ↓
┌─────────────────┐
│  FastAPI Server │  (Port 8000)
│  - Upload       │
│  - Status API   │
│  - Downloads    │
└────────┬────────┘
         │ Task Queue
         ↓
┌─────────────────┐         ┌──────────────────┐
│  Celery Worker  │ ------> │  Main Pipeline   │
│  - Async Tasks  │         │  - Detection     │
│  - State Update │         │  - Generation    │
└────────┬────────┘         │  - Spatial Audio │
         │                  └──────────────────┘
         ↓
┌─────────────────┐
│  Redis Broker   │  (Port 6379)
│  - Task Queue   │
│  - Results      │
└─────────────────┘
```

---

## 🚀 Local Development Setup

### Prerequisites

```bash
# Install Redis
brew install redis          # macOS
sudo apt-get install redis  # Linux

# Start Redis
redis-server
```

### Backend Setup

```bash
cd web/backend

# Install dependencies
pip install fastapi uvicorn celery redis python-multipart

# Start FastAPI server
uvicorn app:app --reload --port 8000

# Start Celery worker (after tasks.py is implemented)
celery -A tasks worker --loglevel=info
```

### Frontend Setup

```bash
cd web/frontend

# Install dependencies
npm install

# Start development server
npm run dev  # Runs on http://localhost:5173
```

---

## 📡 API Endpoints

### Upload Video

```http
POST /upload
Content-Type: multipart/form-data

{
  "file": <video_file>
}

Response:
{
  "task_id": "abc123",
  "status": "pending"
}
```

### Get Processing Status

```http
GET /status/{task_id}

Response:
{
  "task_id": "abc123",
  "status": "processing",  // pending, processing, completed, failed
  "progress": {
    "stage": "audio_generated",  // video_validated, pose_detected, scene_analyzed, audio_generated, spatial_processed, video_merged
    "percentage": 75,
    "message": "Generating audio variations..."
  },
  "stats": {
    "num_footsteps": 12,
    "processing_time_seconds": 45.3
  }
}
```

### Download Results

```http
GET /download/{task_id}/video
GET /download/{task_id}/audio

Response: File stream
```

---

## 🔧 Configuration

**File:** `web/backend/config.py`

```python
# Upload settings
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}

# Paths
UPLOAD_DIR = PROJECT_ROOT / "web" / "uploads"
RESULTS_DIR = PROJECT_ROOT / "web" / "results"

# Redis
REDIS_URL = "redis://localhost:6379/0"
```

---

## 🐛 Known Issues

1. **Celery task not implemented** - `web/backend/tasks.py` is a placeholder
2. **No database** - Results stored on filesystem only
3. **No authentication** - Open access to all endpoints
4. **CORS wide open** - `allow_origins=["*"]` for development
5. **No rate limiting** - Vulnerable to abuse

---

## 🎯 Recommended Next Steps

If you want to deploy the web app publicly:

### Week 1: Core Functionality
1. Implement Celery task in `tasks.py`
2. Test full upload → process → download flow
3. Add basic error handling

### Week 2: Production Readiness
4. Add authentication (JWT tokens)
5. Implement rate limiting
6. Configure CORS for specific origin
7. Add database for job tracking
8. Implement result cleanup (expire after 24 hours)

### Week 3: Deployment
9. Create Docker setup
10. Deploy to cloud (Heroku, DigitalOcean, AWS)
11. Set up monitoring and logging

---

## 💡 Alternative Approach (Simpler)

**Instead of full web app deployment, consider:**

1. **Static Demo Site**
   - Deploy frontend only
   - Show pre-recorded demo results
   - "Try it yourself" button links to GitHub

2. **Local-Only Web App**
   - Include web app in README
   - Users run it locally: `docker-compose up`
   - Good for interviews (can demo live)

3. **Video Demo Only**
   - Record screen capture of web app
   - Upload to YouTube
   - Embed in README
   - **Recommended for 1-2 week timeline**

---

## 📚 File Structure

```
web/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── VideoUpload.jsx        ✅ Complete
│   │   │   ├── ProcessingStatus.jsx   ✅ Complete
│   │   │   └── ResultViewer.jsx       ✅ Complete
│   │   ├── App.jsx                    ✅ Complete
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   └── README.md
│
├── backend/
│   ├── app.py                         ✅ Complete
│   ├── config.py                      ✅ Complete
│   ├── tasks.py                       ⚠️  Placeholder (needs implementation)
│   └── requirements.txt
│
├── uploads/                           (auto-created)
└── results/                           (auto-created)
    └── {task_id}/
        ├── {video}_with_footsteps.mp4
        ├── {video}_footsteps.wav
        └── {video}_footsteps.json
```

---

## 🎓 For Interviews

**If asked about the web app:**

✅ **What's implemented:**
- "I built a full-stack web interface with React and FastAPI"
- "Frontend has upload, progress tracking, and result viewing"
- "Backend has all REST endpoints and file handling"
- "Demonstrates full-stack skills beyond ML/backend"

⚠️ **What's pending:**
- "Celery integration is partially done - prioritized core ML pipeline first"
- "Would need Redis setup for production deployment"
- "Decided to focus on perfecting the ML components over web hosting"

💡 **If they want to see it:**
- "I can run it locally and demo the frontend interface"
- "The ML pipeline works perfectly via CLI - web is just a UI layer"
- "For portfolio demo, I created a video walkthrough instead of hosting costs"

---

## 🌟 Summary

The web app showcases full-stack development skills but is **not critical for ML Engineer roles**. The core ML pipeline is production-ready and well-documented.

**Recommendation:** Focus on:
1. ✅ Polished README with demo video
2. ✅ Strong technical documentation
3. ✅ Core pipeline testing and quality
4. ⚠️ Web app deployment (only if time permits or for full-stack roles)

The web app adds value but is not required for a strong ML portfolio.
