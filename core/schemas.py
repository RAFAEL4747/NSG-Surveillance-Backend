"""
Pydantic schemas — the contract between every layer of the system.
All request/response shapes defined here.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel


# ── Enums ──────────────────────────────────────────────────────────────────────

class SeverityLevel(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"

class AlertType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"

class AnalysisStatus(str, Enum):
    QUEUED     = "queued"
    PROCESSING = "processing"
    COMPLETE   = "complete"
    FAILED     = "failed"


# ── Alert ──────────────────────────────────────────────────────────────────────

class Alert(BaseModel):
    timestamp: str              # "00:02:14"
    severity: SeverityLevel
    alert_type: AlertType
    category: str               # e.g. "weapon_detected", "gunshot", "loitering"
    description: str
    confidence: float           # 0.0 – 1.0
    frame_number: Optional[int] = None
    snapshot_path: Optional[str] = None  # relative path to cropped frame


# ── Timeline event ─────────────────────────────────────────────────────────────

class TimelineEvent(BaseModel):
    timestamp: str
    source: AlertType
    event: str
    confidence: Optional[float] = None


# ── Heatmap ───────────────────────────────────────────────────────────────────

class HeatmapData(BaseModel):
    grid_size: int              # N×N
    cells: list[float]          # flattened N² values, 0.0–1.0


# ── Detection results ─────────────────────────────────────────────────────────

class DetectedObject(BaseModel):
    label: str
    confidence: float
    bbox: list[float]           # [x1, y1, x2, y2] normalised 0–1
    frame_number: int
    timestamp: str

class DetectedFace(BaseModel):
    face_id: str
    timestamp: str
    frame_number: int
    watchlist_match: Optional[str] = None  # watchlist entry ID if matched
    match_confidence: Optional[float] = None
    snapshot_path: Optional[str] = None

class AudioAnomaly(BaseModel):
    timestamp: str
    anomaly_type: str           # "gunshot", "explosion", "scream", "keyword"
    confidence: float
    detail: str                 # e.g. matched keyword, dB level


# ── Summary ───────────────────────────────────────────────────────────────────

class AnalysisSummary(BaseModel):
    persons_detected: int
    objects_flagged: int
    faces_recognised: int
    watchlist_matches: int
    audio_anomalies: int
    threats_flagged: int
    duration_analysed: str
    frames_processed: int
    overall_threat_level: SeverityLevel


# ── Full analysis result ───────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    job_id: str
    status: AnalysisStatus
    video_file: str
    audio_file: str
    alerts: list[Alert]                     = []
    timeline: list[TimelineEvent]           = []
    heatmap: HeatmapData
    detected_objects: list[DetectedObject]  = []
    detected_faces: list[DetectedFace]      = []
    audio_anomalies: list[AudioAnomaly]     = []
    summary: AnalysisSummary
    report_path: Optional[str]              = None   # path to generated PDF
    error: Optional[str]                    = None


# ── API request / response ────────────────────────────────────────────────────

class AnalysisJobResponse(BaseModel):
    job_id: str
    status: AnalysisStatus
    message: str

class StatusResponse(BaseModel):
    job_id: str
    status: AnalysisStatus
    progress: int               # 0–100
    current_phase: str
