from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel

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

class Alert(BaseModel):
    timestamp: str
    severity: SeverityLevel
    alert_type: AlertType
    category: str
    description: str
    confidence: float
    frame_number: Optional[int] = None
    snapshot_path: Optional[str] = None

class TimelineEvent(BaseModel):
    timestamp: str
    source: AlertType
    event: str
    confidence: Optional[float] = None

class HeatmapData(BaseModel):
    grid_size: int
    cells: list[float]

class DetectedObject(BaseModel):
    label: str
    confidence: float
    bbox: list[float]
    frame_number: int
    timestamp: str

class DetectedFace(BaseModel):
    face_id: str
    timestamp: str
    frame_number: int
    watchlist_match: Optional[str] = None
    match_confidence: Optional[float] = None
    snapshot_path: Optional[str] = None

class AudioAnomaly(BaseModel):
    timestamp: str
    anomaly_type: str
    confidence: float
    detail: str

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

class AnalysisResult(BaseModel):
    job_id: str
    status: AnalysisStatus
    video_file: str
    audio_file: str
    alerts: list[Alert]                    = []
    timeline: list[TimelineEvent]          = []
    heatmap: HeatmapData
    detected_objects: list[DetectedObject] = []
    detected_faces: list[DetectedFace]     = []
    audio_anomalies: list[AudioAnomaly]    = []
    summary: AnalysisSummary
    report_path: Optional[str]             = None
    error: Optional[str]                   = None
    capabilities: dict                     = {}   # what modules actually ran

class AnalysisJobResponse(BaseModel):
    job_id: str
    status: AnalysisStatus
    message: str

class StatusResponse(BaseModel):
    job_id: str
    status: AnalysisStatus
    progress: int
    current_phase: str
