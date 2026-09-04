"""
FAST Audio Analysis Engine
Designed for low-CPU Render deployments.

Principles:
- Never process unlimited audio.
- Audio is converted to 16 kHz mono.
- Maximum audio analysis duration is 120 seconds.
- Whisper tiny is loaded once per server.
- Speaker clustering is disabled because it is too expensive on Render Free.
- Only report real detections.
"""

import os
import logging
import tempfile
import numpy as np
from typing import Callable

from core.config import settings
from core.schemas import (
    Alert,
    AlertType,
    AudioAnomaly,
    SeverityLevel,
    TimelineEvent,
)

logger = logging.getLogger(__name__)


# HARD LIMIT FOR RENDER
MAX_AUDIO_SECONDS = 120

# Whisper tiny is intentionally used
WHISPER_MODEL = "tiny"


def _ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class AudioAnalyzer:

    # ---------------------------------------------------------
    # LOAD WHISPER ONLY ONCE PER SERVER
    # ---------------------------------------------------------

    _whisper_model = None
    _whisper_attempted = False

    def __init__(self):

        self.whisper = None

        self.capabilities = {
            "transcription": False,
            "gunshot_detection": True,
            "speaker_identification": False,
        }

        self._load_whisper_once()

    def _load_whisper_once(self):

        if AudioAnalyzer._whisper_attempted:

            self.whisper = AudioAnalyzer._whisper_model

            if self.whisper is not None:
                self.capabilities["transcription"] = True

            return

        AudioAnalyzer._whisper_attempted = True

        try:

            import whisper

            logger.info("Loading Whisper tiny...")

            AudioAnalyzer._whisper_model = whisper.load_model(
                WHISPER_MODEL
            )

            self.whisper = AudioAnalyzer._whisper_model

            self.capabilities["transcription"] = True

            logger.info("Whisper tiny loaded successfully")

        except Exception as e:

            AudioAnalyzer._whisper_model = None
            self.whisper = None

            self.capabilities["transcription"] = False
            self.capabilities["transcription_reason"] = str(e)

            logger.warning(
                "Whisper unavailable: %s",
                e
            )

    # ---------------------------------------------------------
    # MAIN ANALYSIS
    # ---------------------------------------------------------

    def analyse(
        self,
        audio_path: str,
        progress_cb: Callable | None = None
    ) -> dict:

        import librosa

        if progress_cb:
            progress_cb(
                62,
                "Loading audio..."
            )

        # -----------------------------------------------------
        # FAST AUDIO LOAD
        # -----------------------------------------------------

        try:

            # IMPORTANT:
            # 16 kHz mono is dramatically cheaper than
            # preserving the original sample rate.

            y, sr = librosa.load(
                audio_path,
                sr=16000,
                mono=True,
                duration=MAX_AUDIO_SECONDS
            )

        except Exception as e:

            logger.exception(
                "Audio loading failed"
            )

            return self._empty_result(
                f"Audio loading failed: {e}"
            )

        duration = len(y) / sr

        logger.info(
            "Audio loaded: %.1f seconds @ %d Hz",
            duration,
            sr
        )

        alerts = []
        timeline = []
        anomalies = []

        transcript = ""

        # -----------------------------------------------------
        # 1. GUNSHOT DETECTION
        # -----------------------------------------------------

        if progress_cb:
            progress_cb(
                66,
                "Gunshot detection..."
            )

        try:

            gs = self._detect_impulses_fast(
                y,
                sr
            )

            alerts.extend(gs["alerts"])
            timeline.extend(gs["timeline"])
            anomalies.extend(gs["anomalies"])

        except Exception as e:

            logger.warning(
                "Gunshot detection failed: %s",
                e
            )

        # -----------------------------------------------------
        # 2. WHISPER
        # -----------------------------------------------------

        if progress_cb:
            progress_cb(
                73,
                "Speech transcription..."
            )

        if self.whisper:

            try:

                transcript, speech_timeline = (
                    self._transcribe_fast(
                        y,
                        sr
                    )
                )

                timeline.extend(
                    speech_timeline
                )

            except Exception as e:

                logger.warning(
                    "Whisper failed: %s",
                    e
                )

        else:

            timeline.append(
                TimelineEvent(
                    timestamp="00:00:00",
                    source=AlertType.AUDIO,
                    event=(
                        "[Speech transcription unavailable]"
                    )
                )
            )

        # -----------------------------------------------------
        # 3. KEYWORDS
        # -----------------------------------------------------

        if progress_cb:
            progress_cb(
                82,
                "Keyword scan..."
            )

        if transcript:

            kw = self._keyword_scan(
                transcript
            )

            alerts.extend(
                kw["alerts"]
            )

            timeline.extend(
                kw["timeline"]
            )

            anomalies.extend(
                kw["anomalies"]
            )

        # -----------------------------------------------------
        # 4. SPEAKER IDENTIFICATION
        # -----------------------------------------------------

        if progress_cb:
            progress_cb(
                88,
                "Finishing audio analysis..."
            )

        # DISABLED FOR FAST RENDER MODE.
        #
        # KMeans over every MFCC frame can become extremely
        # expensive on CPU-only Render instances.

        logger.info(
            "Speaker identification skipped "
            "in FAST mode"
        )

        # -----------------------------------------------------
        # RETURN
        # -----------------------------------------------------

        if progress_cb:
            progress_cb(
                94,
                "Audio analysis complete..."
            )

        return {

            "alerts": alerts,

            "timeline": timeline,

            "audio_anomalies": anomalies,

            "transcript": transcript,

            "summary_stats": {

                "audio_anomalies":
                    len(anomalies),

                "speakers_detected":
                    0,

                "duration_sec":
                    round(
                        duration,
                        1
                    ),
            },

            "capabilities":
                self.capabilities,
        }

    # ---------------------------------------------------------
    # FAST IMPULSE DETECTION
    # ---------------------------------------------------------

    def _detect_impulses_fast(
        self,
        y,
        sr
    ):

        import librosa

        alerts = []
        timeline = []
        anomalies = []

        # Larger chunks = fewer expensive librosa calls
        chunk_seconds = 5

        chunk_samples = (
            chunk_seconds * sr
        )

        n_chunks = int(
            np.ceil(
                len(y) /
                chunk_samples
            )
        )

        for i in range(n_chunks):

            start = int(
                i * chunk_samples
            )

            end = int(
                min(
                    len(y),
                    start + chunk_samples
                )
            )

            seg = y[start:end]

            if len(seg) < sr * 0.25:
                continue

            t = start / sr

            rms = float(
                np.sqrt(
                    np.mean(
                        seg ** 2
                    )
                )
            )

            if rms < 1e-7:
                continue

            db = (
                20 *
                np.log10(rms)
                + 96
            )

            # Cheap onset calculation
            onset_env = (
                librosa.onset.onset_strength(
                    y=seg,
                    sr=sr,
                    hop_length=1024
                )
            )

            if len(onset_env) == 0:
                continue

            peak_onset = float(
                np.max(onset_env)
            )

            # Conservative detection
            if (
                db >
                settings.GUNSHOT_DB_THRESHOLD
                and
                peak_onset >
                settings.GUNSHOT_ONSET_THRESHOLD
            ):

                conf = min(
                    0.95,
                    0.60
                    +
                    (
                        db -
                        settings.GUNSHOT_DB_THRESHOLD
                    ) / 40
                    +
                    (
                        peak_onset -
                        settings.GUNSHOT_ONSET_THRESHOLD
                    ) / 60
                )

                severity = (
                    SeverityLevel.HIGH
                    if conf > 0.80
                    else SeverityLevel.MEDIUM
                )

                timestamp = _ts(t)

                alerts.append(
                    Alert(
                        timestamp=timestamp,
                        severity=severity,
                        alert_type=AlertType.AUDIO,
                        category="gunshot_detected",
                        description=(
                            f"Possible acoustic impulse "
                            f"at {timestamp} "
                            f"(dB {db:.1f}, "
                            f"confidence {conf:.0%})"
                        ),
                        confidence=round(
                            conf,
                            2
                        ),
                    )
                )

                timeline.append(
                    TimelineEvent(
                        timestamp=timestamp,
                        source=AlertType.AUDIO,
                        event=(
                            f"Acoustic impulse — "
                            f"possible gunshot "
                            f"({conf:.0%})"
                        ),
                        confidence=round(
                            conf,
                            2
                        ),
                    )
                )

                anomalies.append(
                    AudioAnomaly(
                        timestamp=timestamp,
                        anomaly_type="gunshot",
                        confidence=round(
                            conf,
                            2
                        ),
                        detail=(
                            f"dB:{db:.1f} "
                            f"onset:{peak_onset:.1f}"
                        ),
                    )
                )

        return {
            "alerts": alerts,
            "timeline": timeline,
            "anomalies": anomalies,
        }

    # ---------------------------------------------------------
    # FAST WHISPER
    # ---------------------------------------------------------

    def _transcribe_fast(
        self,
        y,
        sr
    ):

        timeline = []

        if self.whisper is None:
            return "", timeline

        try:

            # Whisper accepts a numpy waveform.
            # No temporary MP3/WAV conversion required.

            result = self.whisper.transcribe(
                y,
                language="en",
                fp16=False,
                temperature=0,
                verbose=False,
                condition_on_previous_text=False,
                word_timestamps=False
            )

            text = (
                result
                .get("text", "")
                .strip()
            )

            for seg in result.get(
                "segments",
                []
            ):

                no_speech = seg.get(
                    "no_speech_prob",
                    1.0
                )

                if no_speech < 0.5:

                    timeline.append(
                        TimelineEvent(
                            timestamp=_ts(
                                seg["start"]
                            ),
                            source=AlertType.AUDIO,
                            event=(
                                "Speech: "
                                f'"{seg["text"].strip()}"'
                            ),
                            confidence=round(
                                1 - no_speech,
                                2
                            ),
                        )
                    )

            return text, timeline

        except Exception as e:

            logger.warning(
                "Whisper transcription failed: %s",
                e
            )

            return "", timeline

    # ---------------------------------------------------------
    # KEYWORD SCAN
    # ---------------------------------------------------------

    def _keyword_scan(
        self,
        transcript
    ):

        alerts = []
        timeline = []
        anomalies = []

        words = set(
            transcript.lower().split()
        )

        matched = [
            kw
            for kw in settings.THREAT_KEYWORDS
            if kw.lower() in words
        ]

        if matched:

            n = len(matched)

            severity = (
                SeverityLevel.HIGH
                if n >= 4
                else SeverityLevel.MEDIUM
                if n >= 2
                else SeverityLevel.LOW
            )

            confidence = min(
                0.92,
                0.50 + n * 0.10
            )

            alerts.append(
                Alert(
                    timestamp="00:00:00",
                    severity=severity,
                    alert_type=AlertType.AUDIO,
                    category="keyword_alert",
                    description=(
                        "Threat keywords in transcript: "
                        + ", ".join(matched)
                    ),
                    confidence=confidence,
                )
            )

            timeline.append(
                TimelineEvent(
                    timestamp="00:00:00",
                    source=AlertType.AUDIO,
                    event=(
                        "Keywords matched: "
                        + ", ".join(matched)
                    ),
                )
            )

            anomalies.append(
                AudioAnomaly(
                    timestamp="00:00:00",
                    anomaly_type="keyword",
                    confidence=confidence,
                    detail=(
                        f"Matched {n} keyword(s): "
                        + ", ".join(matched)
                    ),
                )
            )

        return {
            "alerts": alerts,
            "timeline": timeline,
            "anomalies": anomalies,
        }

    # ---------------------------------------------------------
    # EMPTY RESULT
    # ---------------------------------------------------------

    def _empty_result(
        self,
        reason
    ):

        return {

            "alerts": [],

            "timeline": [
                TimelineEvent(
                    timestamp="00:00:00",
                    source=AlertType.AUDIO,
                    event=reason,
                )
            ],

            "audio_anomalies": [],

            "transcript": "",

            "summary_stats": {
                "audio_anomalies": 0,
                "speakers_detected": 0,
                "duration_sec": 0,
            },

            "capabilities":
                self.capabilities,
        }