"""
REST API Routes
POST /analyse         — upload MP4 + MP3, start analysis job
GET  /jobs/{job_id}   — poll job status + progress
GET  /results/{job_id}— fetch full analysis result
GET  /report/{job_id} — download PDF report
GET  /jobs            — list all jobs
DELETE /jobs/{job_id} — remove job + files
"""

import os
import shutil
import logging
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

from core.config import settings
from core.schemas import AnalysisJobResponse, AnalysisStatus, StatusResponse
from core import job_manager

router = APIRouter(tags=["Analysis"])
logger = logging.getLogger(__name__)

MAX_VIDEO_BYTES = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024
MAX_AUDIO_BYTES = settings.MAX_AUDIO_SIZE_MB  * 1024 * 1024


# ── Helper ─────────────────────────────────────────────────────────────────────

async def _save_upload(upload: UploadFile, dest: str, max_bytes: int) -> str:
    """Stream-save an uploaded file, enforcing size limit."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    total = 0
    with open(dest, "wb") as f:
        while chunk := await upload.read(1024 * 256):  # 256 KB chunks
            total += len(chunk)
            if total > max_bytes:
                os.remove(dest)
                raise HTTPException(
                    413,
                    detail=f"File exceeds size limit ({max_bytes // 1024 // 1024} MB)",
                )
            f.write(chunk)
    return dest


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/analyse", response_model=AnalysisJobResponse, status_code=202)
async def start_analysis(
    video: UploadFile = File(..., description="MP4 video file"),
    audio: UploadFile = File(..., description="MP3 audio file"),
):
    """
    Upload an MP4 and MP3 and kick off the analysis pipeline.
    Returns a job_id to poll for progress.
    """
    # Validate types
    if not video.filename.lower().endswith(".mp4"):
        raise HTTPException(400, "Video must be an MP4 file")
    if not audio.filename.lower().endswith(".mp3"):
        raise HTTPException(400, "Audio must be an MP3 file")

    job_id = job_manager.create_job()
    job_upload_dir = os.path.join(settings.UPLOAD_DIR, job_id)

    video_path = os.path.join(job_upload_dir, video.filename)
    audio_path = os.path.join(job_upload_dir, audio.filename)

    await _save_upload(video, video_path, MAX_VIDEO_BYTES)
    await _save_upload(audio, audio_path, MAX_AUDIO_BYTES)

    logger.info(f"Job {job_id}: video={video.filename}, audio={audio.filename}")
    job_manager.run_job_async(job_id, video_path, audio_path)

    return AnalysisJobResponse(
        job_id=job_id,
        status=AnalysisStatus.PROCESSING,
        message=f"Analysis started. Poll /api/v1/jobs/{job_id} for progress.",
    )


@router.get("/jobs/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """Poll the progress of a running analysis job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        current_phase=job["phase"],
    )


@router.get("/results/{job_id}")
async def get_results(job_id: str):
    """Fetch the full analysis result for a completed job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] == AnalysisStatus.FAILED:
        raise HTTPException(500, detail=job.get("error", "Analysis failed"))
    if job["status"] != AnalysisStatus.COMPLETE:
        raise HTTPException(202, detail="Analysis still in progress")
    if not job["result"]:
        raise HTTPException(500, "Result not available")
    return job["result"].model_dump()


@router.get("/report/{job_id}")
async def download_report(job_id: str):
    """Download the PDF incident report for a completed job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] != AnalysisStatus.COMPLETE:
        raise HTTPException(202, detail="Analysis not complete")

    result = job.get("result")
    if not result or not result.report_path:
        raise HTTPException(404, "Report not generated")

    abs_path = os.path.join(settings.OUTPUT_DIR, result.report_path)
    if not os.path.exists(abs_path):
        raise HTTPException(404, "Report file not found on disk")

    return FileResponse(
        abs_path,
        media_type="application/pdf",
        filename=Path(abs_path).name,
    )


@router.get("/jobs")
async def list_jobs():
    """List all jobs (status + metadata)."""
    jobs = job_manager.get_all_jobs()
    return {"jobs": [
        {
            "job_id":     j["job_id"],
            "status":     j["status"],
            "progress":   j["progress"],
            "phase":      j["phase"],
            "created_at": j["created_at"],
        }
        for j in jobs
    ]}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Remove a job and its uploaded/output files."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    for d in [
        os.path.join(settings.UPLOAD_DIR, job_id),
        os.path.join(settings.OUTPUT_DIR, job_id),
    ]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)

    from core.job_manager import _jobs, _lock
    with _lock:
        _jobs.pop(job_id, None)

    return {"message": f"Job {job_id} deleted"}
