"""
Job Manager
Tracks analysis jobs in memory, runs them in background threads,
and streams progress updates via a shared state dict.
"""

import uuid
import threading
import logging
import os
from datetime import datetime

from core.schemas import (
    AnalysisResult, AnalysisStatus, AnalysisSummary,
    HeatmapData, SeverityLevel,
)
from core.config import settings

logger = logging.getLogger(__name__)

# ── In-memory job store ────────────────────────────────────────────────────────
# { job_id: { "status", "progress", "phase", "result", "error", "created_at" } }
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def create_job() -> str:
    job_id = str(uuid.uuid4())[:8].upper()
    with _lock:
        _jobs[job_id] = {
            "status":     AnalysisStatus.QUEUED,
            "progress":   0,
            "phase":      "Queued",
            "result":     None,
            "error":      None,
            "created_at": datetime.utcnow().isoformat(),
        }
    return job_id


def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)


def get_all_jobs() -> list[dict]:
    return [{"job_id": jid, **info} for jid, info in _jobs.items()]


def _update(job_id: str, **kwargs):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def run_job_async(job_id: str, video_path: str, audio_path: str):
    """Launch the analysis pipeline in a background thread."""
    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, video_path, audio_path),
        daemon=True,
    )
    thread.start()


def _run_pipeline(job_id: str, video_path: str, audio_path: str):
    _update(job_id, status=AnalysisStatus.PROCESSING, progress=5, phase="Starting…")

    output_dir = os.path.join(settings.OUTPUT_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)

    def progress_cb(pct: int, phase: str):
        _update(job_id, progress=pct, phase=phase)

    try:
        # ── Video analysis ─────────────────────────────────────────────────────
        from core.video_analyzer import VideoAnalyzer
        progress_cb(10, "Loading video analyser…")
        va = VideoAnalyzer()
        video_results = va.analyse(video_path, output_dir, progress_cb)

        # ── Audio analysis ─────────────────────────────────────────────────────
        from core.audio_analyzer import AudioAnalyzer
        progress_cb(61, "Loading audio analyser…")
        aa = AudioAnalyzer()
        audio_results = aa.analyse(audio_path, progress_cb)

        # ── Merge results ──────────────────────────────────────────────────────
        progress_cb(90, "Merging results…")
        all_alerts   = video_results["alerts"]   + audio_results["alerts"]
        all_timeline = video_results["timeline"] + audio_results["timeline"]
        all_timeline.sort(key=lambda e: e.timestamp)

        vs = video_results["summary_stats"]
        as_ = audio_results["summary_stats"]

        high_count = len([a for a in all_alerts if a.severity == SeverityLevel.HIGH])
        med_count  = len([a for a in all_alerts if a.severity == SeverityLevel.MEDIUM])
        threat_lvl = (
            SeverityLevel.HIGH   if high_count > 0 else
            SeverityLevel.MEDIUM if med_count  > 0 else
            SeverityLevel.LOW
        )

        summary = AnalysisSummary(
            persons_detected   = vs.get("persons_detected", 0),
            objects_flagged    = vs.get("objects_flagged", 0),
            faces_recognised   = vs.get("faces_recognised", 0),
            watchlist_matches  = vs.get("watchlist_matches", 0),
            audio_anomalies    = as_.get("audio_anomalies", 0),
            threats_flagged    = high_count + med_count,
            duration_analysed  = vs.get("duration_analysed", "00:00:00"),
            frames_processed   = vs.get("frames_processed", 0),
            overall_threat_level = threat_lvl,
        )

        result = AnalysisResult(
            job_id           = job_id,
            status           = AnalysisStatus.COMPLETE,
            video_file       = video_path,
            audio_file       = audio_path,
            alerts           = all_alerts,
            timeline         = all_timeline,
            heatmap          = video_results["heatmap"],
            detected_objects = video_results["detected_objects"],
            detected_faces   = video_results["detected_faces"],
            audio_anomalies  = audio_results["audio_anomalies"],
            summary          = summary,
        )

        # ── PDF report ─────────────────────────────────────────────────────────
        progress_cb(93, "Generating PDF report…")
        try:
            from core.report_generator import generate_report
            pdf_path = generate_report(result, output_dir)
            result.report_path = os.path.relpath(pdf_path, settings.OUTPUT_DIR)
        except Exception as e:
            logger.warning(f"PDF generation failed: {e}")

        progress_cb(100, "Complete")
        _update(job_id, status=AnalysisStatus.COMPLETE, progress=100,
                phase="Complete", result=result)
        logger.info(f"Job {job_id} complete — {len(all_alerts)} alerts")

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        _update(job_id, status=AnalysisStatus.FAILED, progress=0,
                phase="Failed", error=str(e))
