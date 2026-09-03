import os, shutil, logging
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from core.config import settings
from core.schemas import AnalysisJobResponse, AnalysisStatus, StatusResponse
from core import job_manager

router = APIRouter(tags=["Analysis"])
logger = logging.getLogger(__name__)

async def _save(upload: UploadFile, dest: str, max_mb: int) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    total = 0
    with open(dest,"wb") as f:
        while chunk := await upload.read(256*1024):
            total += len(chunk)
            if total > max_mb*1024*1024:
                os.remove(dest)
                raise HTTPException(413, f"File exceeds {max_mb}MB limit")
            f.write(chunk)
    return dest

@router.post("/analyse", response_model=AnalysisJobResponse, status_code=202)
async def start_analysis(video: UploadFile = File(...), audio: UploadFile = File(...)):
    if not video.filename.lower().endswith(".mp4"):
        raise HTTPException(400, "Video must be MP4")
    if not audio.filename.lower().endswith(".mp3"):
        raise HTTPException(400, "Audio must be MP3")
    jid  = job_manager.create_job()
    udir = os.path.join(settings.UPLOAD_DIR, jid)
    vp   = await _save(video, os.path.join(udir, video.filename), settings.MAX_VIDEO_SIZE_MB)
    ap   = await _save(audio, os.path.join(udir, audio.filename), settings.MAX_AUDIO_SIZE_MB)
    job_manager.run_job_async(jid, vp, ap)
    return AnalysisJobResponse(job_id=jid, status=AnalysisStatus.PROCESSING,
                               message=f"Started. Poll /api/v1/jobs/{jid}")

@router.get("/jobs/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job: raise HTTPException(404, "Job not found")
    return StatusResponse(job_id=job_id, status=job["status"],
                          progress=job["progress"], current_phase=job["phase"])

@router.get("/results/{job_id}")
async def get_results(job_id: str):
    job = job_manager.get_job(job_id)
    if not job: raise HTTPException(404, "Job not found")
    if job["status"] == AnalysisStatus.FAILED:
        raise HTTPException(500, detail=job.get("error","Failed"))
    if job["status"] != AnalysisStatus.COMPLETE:
        raise HTTPException(202, detail="Still processing")
    return job["result"].model_dump()

@router.get("/report/{job_id}")
async def download_report(job_id: str):
    job = job_manager.get_job(job_id)
    if not job: raise HTTPException(404, "Not found")
    if job["status"] != AnalysisStatus.COMPLETE: raise HTTPException(202, "Not complete")
    r = job.get("result")
    if not r or not r.report_path: raise HTTPException(404, "Report not generated")
    path = os.path.join(settings.OUTPUT_DIR, r.report_path)
    if not os.path.exists(path): raise HTTPException(404, "File missing")
    return FileResponse(path, media_type="application/pdf", filename=Path(path).name)

@router.get("/jobs")
async def list_jobs():
    return {"jobs": [{"job_id":j["job_id"],"status":j["status"],"progress":j["progress"],
                      "phase":j["phase"],"created_at":j["created_at"]}
                     for j in job_manager.get_all_jobs()]}

@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if not job_manager.get_job(job_id): raise HTTPException(404, "Not found")
    for d in [os.path.join(settings.UPLOAD_DIR, job_id), os.path.join(settings.OUTPUT_DIR, job_id)]:
        if os.path.exists(d): shutil.rmtree(d, ignore_errors=True)
    from core.job_manager import _jobs, _lock
    with _lock: _jobs.pop(job_id, None)
    return {"message": f"Job {job_id} deleted"}
