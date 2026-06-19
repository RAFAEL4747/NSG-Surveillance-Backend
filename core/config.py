"""
NSG System Configuration
All tuneable settings live here — no magic numbers scattered across the codebase.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Paths ──────────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: str = str(Path(__file__).resolve().parent.parent / "uploads")
    OUTPUT_DIR: str = str(Path(__file__).resolve().parent.parent / "outputs")
    MODEL_DIR: str = str(Path(__file__).resolve().parent.parent / "models")

    # ── API ────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ── Video analysis ─────────────────────────────────────────────────
    FRAME_SAMPLE_RATE: int = 1          # Extract 1 frame per second
    MAX_VIDEO_SIZE_MB: int = 500
    YOLO_CONFIDENCE: float = 0.45       # Detection confidence threshold
    YOLO_MODEL: str = "yolov8n.pt"      # nano = fast; swap to yolov8m.pt for accuracy

    # ── Face detection ─────────────────────────────────────────────────
    FACE_CONFIDENCE: float = 0.6
    WATCHLIST_DIR: str = str(Path(__file__).resolve().parent.parent / "models" / "watchlist")

    # ── Audio analysis ─────────────────────────────────────────────────
    MAX_AUDIO_SIZE_MB: int = 100
    AUDIO_CHUNK_SEC: int = 5            # Analyse in 5-second windows
    GUNSHOT_DB_THRESHOLD: float = 85.0  # dB level to flag as potential gunshot
    THREAT_KEYWORDS: list = [
        "target", "eliminate", "attack", "bomb", "weapon", "kill",
        "position", "breach", "detonate", "ambush", "hostage"
    ]

    # ── Anomaly detection ──────────────────────────────────────────────
    LOITER_SECONDS: int = 120           # Flag person stationary > 2 min
    CROWD_THRESHOLD: int = 6            # Flag crowd formation >= 6 persons

    # ── Output ─────────────────────────────────────────────────────────
    HEATMAP_GRID: int = 8               # 8×8 = 64 zone grid
    REPORT_LOGO: str = ""               # Path to NSG logo for PDF reports

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Ensure required directories exist
for d in [settings.UPLOAD_DIR, settings.OUTPUT_DIR, settings.MODEL_DIR, settings.WATCHLIST_DIR]:
    os.makedirs(d, exist_ok=True)
