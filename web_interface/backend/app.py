from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import shutil
from pathlib import Path
import uuid
from datetime import datetime
import glob
import os

from config import UPLOAD_DIR, RESULTS_DIR, ALLOWED_VIDEO_EXTENSIONS, MAX_UPLOAD_SIZE
from tasks import process_footstep_video, celery_app

app = FastAPI(title="Footstep Generator API")

# Enable CORS so your React frontend can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Footstep Generator API is running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "upload_dir": str(UPLOAD_DIR),
        "upload_dir_exists": UPLOAD_DIR.exists()
    }


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    mode: str = "automatic"
):
    """
    Upload a video file for footstep generation
    Returns a job_id for tracking processing status
    
    Args:
        file: Video file to process
        mode: 'automatic' or 'manual' processing mode
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        )
    
    # Check file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {MAX_UPLOAD_SIZE / (1024*1024)}MB"
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    video_filename = f"{job_id}{file_ext}"
    upload_path = UPLOAD_DIR / video_filename
    
    try:
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Trigger background task
    task = process_footstep_video.delay(job_id, video_filename, mode)
    
    return {
        "job_id": job_id,
        "task_id": task.id,
        "filename": file.filename,
        "mode": mode,
        "status": "queued",
        "message": "Video uploaded successfully. Processing started in background."
    }

@app.get("/status/{task_id}")
async def get_job_status(task_id: str):
    """
    Check the processing status of a Celery task

    Args:
        task_id: The Celery task ID (returned as 'task_id' from /upload)
    """
    from celery.result import AsyncResult

    # Query Celery task by task_id
    result = AsyncResult(task_id, app=celery_app)

    if result.state == 'PENDING':
        response = {
            'task_id': task_id,
            'status': 'pending',
            'message': 'Task is waiting to start...'
        }
    elif result.state == 'PROCESSING':
        response = {
            'task_id': task_id,
            'status': 'processing',
            'result': result.info
        }
    elif result.state == 'SUCCESS':
        response = {
            'task_id': task_id,
            'status': 'completed',
            'result': result.info
        }
    elif result.state == 'FAILURE':
        response = {
            'task_id': task_id,
            'status': 'failed',
            'error': str(result.info)
        }
    else:
        response = {
            'task_id': task_id,
            'status': result.state.lower(),
            'message': 'Unknown status'
        }

    return response


@app.get("/preview/{job_id}/original")
async def preview_original_video(job_id: str, request: Request):
    """
    Stream original uploaded video for preview

    Args:
        job_id: The job ID (returned as 'job_id' from /upload)
    """
    # Find the original uploaded video
    video_files = list(UPLOAD_DIR.glob(f"{job_id}.*"))

    if not video_files:
        raise HTTPException(status_code=404, detail="Original video not found")

    video_path = video_files[0]

    return ranged_response(video_path, request)


def ranged_response(file_path: Path, request: Request):
    """
    Create a streaming response that supports HTTP range requests for video playback
    """
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        # Parse range header (e.g., "bytes=0-1023")
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0]) if byte_range[0] else 0
        end = int(byte_range[1]) if byte_range[1] and byte_range[1] != "" else file_size - 1

        # Ensure end doesn't exceed file size
        end = min(end, file_size - 1)
        content_length = end - start + 1

        def iterfile():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                chunk_size = 8192
                while remaining > 0:
                    chunk = f.read(min(chunk_size, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": "video/mp4",
        }

        return StreamingResponse(
            iterfile(),
            status_code=206,
            headers=headers
        )
    else:
        # No range header, return entire file
        def iterfile():
            with open(file_path, "rb") as f:
                yield from f

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": "video/mp4",
        }

        return StreamingResponse(
            iterfile(),
            headers=headers
        )


@app.get("/preview/{job_id}/generated")
async def preview_generated_video(job_id: str, request: Request):
    """
    Stream generated video with footsteps for preview

    Args:
        job_id: The job ID (returned as 'job_id' from /upload)
    """
    # Find the generated video in results directory (flat structure)
    # Pattern: results/{job_id}_with_footsteps.mp4
    video_path = RESULTS_DIR / f"{job_id}_with_footsteps.mp4"

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Generated video not found")

    return ranged_response(video_path, request)


@app.get("/download/{job_id}/video")
async def download_video(job_id: str):
    """
    Download generated video with footsteps

    Args:
        job_id: The job ID (returned as 'job_id' from /upload)
    """
    # Find the generated video in results directory (flat structure)
    video_path = RESULTS_DIR / f"{job_id}_with_footsteps.mp4"

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Generated video not found")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'attachment; filename="{video_path.name}"'
        },
        filename=video_path.name
    )


@app.get("/download/{job_id}/audio")
async def download_audio(job_id: str):
    """
    Download generated audio file

    Args:
        job_id: The job ID (returned as 'job_id' from /upload)
    """
    # Find the audio file in results directory (flat structure)
    audio_path = RESULTS_DIR / f"{job_id}_footsteps.wav"

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{audio_path.name}"'
        },
        filename=audio_path.name
    )