"""
NSG Backend — Smoke Test
Run: python tests/test_smoke.py
Tests that all modules import correctly and core logic runs without error.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tempfile
import wave
import struct
import cv2

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
errors = []

def check(label, fn):
    try:
        fn()
        print(f"  {PASS}  {label}")
    except Exception as e:
        print(f"  {FAIL}  {label}  →  {e}")
        errors.append(label)

print("\n━━━  NSG Backend Smoke Test  ━━━\n")

# ── 1. Config ──────────────────────────────────────────────────────────────────
print("[ Config ]")
def test_config():
    from core.config import settings
    assert settings.FRAME_SAMPLE_RATE == 1
    assert settings.GUNSHOT_DB_THRESHOLD == 85.0
    assert "target" in settings.THREAT_KEYWORDS
check("Settings load and values correct", test_config)

# ── 2. Schemas ─────────────────────────────────────────────────────────────────
print("\n[ Schemas ]")
def test_schemas():
    from core.schemas import Alert, SeverityLevel, AlertType, HeatmapData
    a = Alert(
        timestamp="00:01:23", severity=SeverityLevel.HIGH,
        alert_type=AlertType.VIDEO, category="weapon_detected",
        description="Test alert", confidence=0.92,
    )
    assert a.severity == SeverityLevel.HIGH
    hm = HeatmapData(grid_size=8, cells=[0.5]*64)
    assert len(hm.cells) == 64
check("Alert and HeatmapData schemas", test_schemas)

# ── 3. Job Manager ─────────────────────────────────────────────────────────────
print("\n[ Job Manager ]")
def test_job_manager():
    from core.job_manager import create_job, get_job
    from core.schemas import AnalysisStatus
    jid = create_job()
    assert len(jid) == 8
    job = get_job(jid)
    assert job["status"] == AnalysisStatus.QUEUED
    assert job["progress"] == 0
check("Create and retrieve job", test_job_manager)

# ── 4. Video Analyzer (no model) ───────────────────────────────────────────────
print("\n[ Video Analyzer ]")
def test_video_analyzer_init():
    from core.video_analyzer import VideoAnalyzer, _frame_to_ts
    assert _frame_to_ts(0, 25.0)   == "00:00:00"
    assert _frame_to_ts(150, 25.0) == "00:00:06"
    assert _frame_to_ts(1500, 25.0)== "00:01:00"
    va = VideoAnalyzer()
    assert va.face_cascade is not None
check("VideoAnalyzer init + timestamp helper", test_video_analyzer_init)

def test_video_on_synthetic():
    from core.video_analyzer import VideoAnalyzer
    # Create a tiny synthetic MP4 in a temp dir
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "test.mp4")
        out_dir    = os.path.join(tmp, "out")
        os.makedirs(out_dir)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(video_path, fourcc, 5, (320, 240))
        for _ in range(10):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            vw.write(frame)
        vw.release()

        va = VideoAnalyzer()
        results = va.analyse(video_path, out_dir)
        assert "heatmap" in results
        assert len(results["heatmap"].cells) == 64
        assert "summary_stats" in results
check("VideoAnalyzer on synthetic MP4", test_video_on_synthetic)

# ── 5. Audio Analyzer ──────────────────────────────────────────────────────────
print("\n[ Audio Analyzer ]")
def test_audio_on_synthetic():
    from core.audio_analyzer import AudioAnalyzer, _sec_to_ts
    assert _sec_to_ts(0)    == "00:00:00"
    assert _sec_to_ts(3661) == "01:01:01"

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, "test.mp3")

        # Write a minimal WAV, save as .mp3 path (librosa reads both)
        sr, duration = 16000, 3
        samples = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, sr * duration))
                   * 0.3).astype(np.float32)
        import soundfile as sf
        sf.write(audio_path.replace(".mp3", ".wav"), samples, sr)
        # rename to .mp3 — librosa handles it
        os.rename(audio_path.replace(".mp3", ".wav"), audio_path)

        aa = AudioAnalyzer()
        results = aa.analyse(audio_path)
        assert "alerts" in results
        assert "audio_anomalies" in results
        assert isinstance(results["transcript"], str)
check("AudioAnalyzer on synthetic tone", test_audio_on_synthetic)

def test_keyword_scan():
    from core.audio_analyzer import AudioAnalyzer
    aa = AudioAnalyzer()
    res = aa._scan_keywords("the target is at the position ready to attack")
    assert len(res["alerts"]) > 0
    assert res["alerts"][0].category == "keyword_alert"
check("Keyword scanner flags threat words", test_keyword_scan)

# ── 6. Report Generator ────────────────────────────────────────────────────────
print("\n[ Report Generator ]")
def test_report():
    from core.schemas import (
        AnalysisResult, AnalysisStatus, AnalysisSummary,
        HeatmapData, SeverityLevel, Alert, AlertType, TimelineEvent,
    )
    from core.report_generator import generate_report

    with tempfile.TemporaryDirectory() as tmp:
        result = AnalysisResult(
            job_id="TESTJOB",
            status=AnalysisStatus.COMPLETE,
            video_file="/tmp/test.mp4",
            audio_file="/tmp/test.mp3",
            alerts=[
                Alert(
                    timestamp="00:01:00", severity=SeverityLevel.HIGH,
                    alert_type=AlertType.VIDEO, category="weapon_detected",
                    description="Knife detected", confidence=0.90,
                )
            ],
            timeline=[
                TimelineEvent(timestamp="00:01:00", source=AlertType.VIDEO,
                              event="Knife detected in frame")
            ],
            heatmap=HeatmapData(grid_size=8, cells=[0.5]*64),
            summary=AnalysisSummary(
                persons_detected=2, objects_flagged=1, faces_recognised=1,
                watchlist_matches=0, audio_anomalies=0, threats_flagged=1,
                duration_analysed="00:01:30", frames_processed=90,
                overall_threat_level=SeverityLevel.HIGH,
            ),
        )
        pdf_path = generate_report(result, tmp)
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 1000  # non-empty PDF
check("PDF report generates without error", test_report)

# ── 7. FastAPI app imports ─────────────────────────────────────────────────────
print("\n[ FastAPI App ]")
def test_app_import():
    from main import app
    assert app.title == "NSG Surveillance Analysis API"
check("FastAPI app imports successfully", test_app_import)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n━━━  {'All tests passed!' if not errors else f'{len(errors)} test(s) failed: {errors}'}  ━━━\n")
sys.exit(1 if errors else 0)
