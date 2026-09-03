import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: str  = str(Path(__file__).resolve().parent.parent / "uploads")
    OUTPUT_DIR: str  = str(Path(__file__).resolve().parent.parent / "outputs")
    MODEL_DIR: str   = str(Path(__file__).resolve().parent.parent / "models")

    # Video — raise confidence bar to cut false positives
    FRAME_SAMPLE_RATE: int   = 1        # 1 frame per second
    MAX_VIDEO_SIZE_MB: int   = 500
    YOLO_CONFIDENCE: float   = 0.60     # was 0.45 — higher = fewer false positives
    YOLO_MODEL: str          = "yolov8n.pt"

    # Face detection
    FACE_CONFIDENCE: float   = 0.75     # only strong matches
    WATCHLIST_DIR: str       = str(Path(__file__).resolve().parent.parent / "models" / "watchlist")

    # Audio — real acoustic thresholds
    MAX_AUDIO_SIZE_MB: int   = 100
    AUDIO_CHUNK_SEC: int     = 3
    # Gunshot: needs BOTH high dB AND sharp onset — prevents music/door slams
    GUNSHOT_DB_THRESHOLD: float  = 92.0   # raised from 85
    GUNSHOT_ONSET_THRESHOLD: float = 20.0  # sharp transient required
    # Crowd / loiter — require sustained evidence
    LOITER_SECONDS: int      = 180      # 3 min before flagging
    CROWD_THRESHOLD: int     = 8        # raised from 6

    THREAT_KEYWORDS: list = [
        "target", "eliminate", "attack", "bomb", "weapon", "kill",
        "position", "breach", "detonate", "ambush", "hostage", "fire",
    ]

    HEATMAP_GRID: int = 8

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
for d in [settings.UPLOAD_DIR, settings.OUTPUT_DIR, settings.MODEL_DIR, settings.WATCHLIST_DIR]:
    os.makedirs(d, exist_ok=True)
