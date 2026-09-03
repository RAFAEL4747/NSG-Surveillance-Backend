import uuid, threading, logging, os
from datetime import datetime
from core.schemas import AnalysisResult, AnalysisStatus, AnalysisSummary, HeatmapData, SeverityLevel
from core.config import settings

logger = logging.getLogger(__name__)
_jobs: dict[str, dict] = {}
_lock = threading.Lock()

def create_job() -> str:
    jid = str(uuid.uuid4())[:8].upper()
    with _lock:
        _jobs[jid] = {"status": AnalysisStatus.QUEUED, "progress": 0,
                      "phase": "Queued", "result": None, "error": None,
                      "created_at": datetime.utcnow().isoformat()}
    return jid

def get_job(jid): return _jobs.get(jid)
def get_all_jobs(): return [{"job_id": k, **v} for k,v in _jobs.items()]

def _upd(jid, **kw):
    with _lock:
        if jid in _jobs: _jobs[jid].update(kw)

def run_job_async(jid, video_path, audio_path):
    threading.Thread(target=_run, args=(jid, video_path, audio_path), daemon=True).start()

def _run(jid, video_path, audio_path):
    _upd(jid, status=AnalysisStatus.PROCESSING, progress=5, phase="Starting…")
    out = os.path.join(settings.OUTPUT_DIR, jid)
    os.makedirs(out, exist_ok=True)

    def cb(pct, phase): _upd(jid, progress=pct, phase=phase)

    try:
        from core.video_analyzer import VideoAnalyzer
        cb(8, "Loading video analyser…")
        va = VideoAnalyzer()
        vr = va.analyse(video_path, out, cb)

        from core.audio_analyzer import AudioAnalyzer
        cb(61, "Loading audio analyser…")
        aa = AudioAnalyzer()
        ar = aa.analyse(audio_path, cb)

        cb(90, "Compiling results…")
        all_alerts   = vr["alerts"]   + ar["alerts"]
        all_timeline = sorted(vr["timeline"] + ar["timeline"], key=lambda e: e.timestamp)
        capabilities = {**vr.get("capabilities",{}), **ar.get("capabilities",{})}

        hi  = len([a for a in all_alerts if a.severity == SeverityLevel.HIGH])
        med = len([a for a in all_alerts if a.severity == SeverityLevel.MEDIUM])
        lvl = SeverityLevel.HIGH if hi>0 else SeverityLevel.MEDIUM if med>0 else SeverityLevel.LOW

        vs, as_ = vr["summary_stats"], ar["summary_stats"]
        summary = AnalysisSummary(
            persons_detected  = vs.get("persons_detected",0),
            objects_flagged   = vs.get("objects_flagged",0),
            faces_recognised  = vs.get("faces_recognised",0),
            watchlist_matches = vs.get("watchlist_matches",0),
            audio_anomalies   = as_.get("audio_anomalies",0),
            threats_flagged   = hi+med,
            duration_analysed = vs.get("duration_analysed","00:00:00"),
            frames_processed  = vs.get("frames_processed",0),
            overall_threat_level = lvl,
        )

        result = AnalysisResult(
            job_id=jid, status=AnalysisStatus.COMPLETE,
            video_file=video_path, audio_file=audio_path,
            alerts=all_alerts, timeline=all_timeline,
            heatmap=vr["heatmap"],
            detected_objects=vr["detected_objects"],
            detected_faces=vr["detected_faces"],
            audio_anomalies=ar["audio_anomalies"],
            summary=summary, capabilities=capabilities,
        )

        cb(94, "Generating report…")
        try:
            from core.report_generator import generate_report
            pdf = generate_report(result, out)
            result.report_path = os.path.relpath(pdf, settings.OUTPUT_DIR)
        except Exception as e:
            logger.warning(f"PDF failed: {e}")

        cb(100, "Complete")
        _upd(jid, status=AnalysisStatus.COMPLETE, progress=100, phase="Complete", result=result)

    except Exception as e:
        logger.exception(f"Job {jid} failed: {e}")
        _upd(jid, status=AnalysisStatus.FAILED, progress=0, phase="Failed", error=str(e))
