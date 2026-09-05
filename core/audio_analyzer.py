

import os
import time
import logging
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


# ============================================================
# HARD LIMITS
# ============================================================

MAX_AUDIO_SECONDS = 180          # Never analyse more than 3 minutes
MAX_ANALYSIS_SECONDS = 240       # Absolute 4-minute safety budget

WHISPER_MODEL_NAME = "tiny"


def _ts(sec: float) -> str:
    sec = max(0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class AudioAnalyzer:

    # --------------------------------------------------------
    # Whisper is shared between jobs.
    # It is NOT loaded when the FastAPI server starts.
    # --------------------------------------------------------

    _whisper_model = None
    _whisper_attempted = False

    def __init__(self):

        self.whisper = None

        self.capabilities = {
            "transcription": False,
            "gunshot_detection": True,
            "speaker_identification": False,
        }

    # ========================================================
    # WHISPER
    # ========================================================

    def _load_whisper(self):

        if AudioAnalyzer._whisper_attempted:

            self.whisper = AudioAnalyzer._whisper_model

            self.capabilities["transcription"] = (
                self.whisper is not None
            )

            return

        AudioAnalyzer._whisper_attempted = True

        try:

            logger.info(
                "Loading Whisper tiny lazily..."
            )

            import whisper

            AudioAnalyzer._whisper_model = whisper.load_model(
                WHISPER_MODEL_NAME,
                device="cpu"
            )

            self.whisper = AudioAnalyzer._whisper_model

            self.capabilities["transcription"] = True

            logger.info(
                "Whisper tiny loaded successfully"
            )

        except Exception as e:

            self.whisper = None

            self.capabilities["transcription"] = False
            self.capabilities["transcription_reason"] = str(e)

            logger.warning(
                f"Whisper unavailable: {e}"
            )

    # ========================================================
    # MAIN ANALYSIS
    # ========================================================

    def analyse(
        self,
        audio_path: str,
        progress_cb: Callable | None = None
    ) -> dict:

        started = time.monotonic()

        alerts = []
        timeline = []
        anomalies = []

        transcript = ""

        # ----------------------------------------------------
        # AUDIO LOAD
        # ----------------------------------------------------

        if progress_cb:
            progress_cb(62, "Loading audio...")

        try:

            import librosa

            # IMPORTANT:
            # 16 kHz mono is enough for speech/acoustic analysis.
            # duration prevents huge files from killing Render.

            y, sr = librosa.load(
                audio_path,
                sr=16000,
                mono=True,
                duration=MAX_AUDIO_SECONDS
            )

        except Exception as e:

            logger.error(
                f"Audio loading failed: {e}"
            )

            return {
                "alerts": [],
                "timeline": [],
                "audio_anomalies": [],
                "transcript": "",
                "summary_stats": {
                    "audio_anomalies": 0,
                    "speakers_detected": 0,
                    "duration_sec": 0,
                },
                "capabilities": self.capabilities,
                "error": f"Audio loading failed: {e}",
            }

        duration = len(y) / float(sr)

        if duration <= 0:

            return {
                "alerts": [],
                "timeline": [],
                "audio_anomalies": [],
                "transcript": "",
                "summary_stats": {
                    "audio_anomalies": 0,
                    "speakers_detected": 0,
                    "duration_sec": 0,
                },
                "capabilities": self.capabilities,
            }

        logger.info(
            f"Audio loaded: {duration:.1f}s @ {sr}Hz"
        )

        # ====================================================
        # 1. FAST ACOUSTIC IMPULSE DETECTION
        # ====================================================

        if progress_cb:
            progress_cb(
                66,
                "Fast acoustic analysis..."
            )

        if not self._time_exceeded(started):

            gs = self._detect_impulses_fast(
                y,
                sr,
                started
            )

            alerts.extend(gs["alerts"])
            timeline.extend(gs["timeline"])
            anomalies.extend(gs["anomalies"])

        # ====================================================
        # 2. SPEECH TRANSCRIPTION
        # ====================================================

        if progress_cb:
            progress_cb(
                73,
                "Speech transcription..."
            )

        # Only attempt Whisper if we still have time.
        if not self._time_exceeded(started):

            transcript, t_tl = self._transcribe_fast(
                audio_path,
                started
            )

            timeline.extend(t_tl)

        # ====================================================
        # 3. KEYWORD SCAN
        # ====================================================

        if progress_cb:
            progress_cb(
                82,
                "Keyword scan..."
            )

        if transcript and not self._time_exceeded(started):

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

        # ====================================================
        # 4. NO EXPENSIVE KMEANS SPEAKER ID
        # ====================================================
        #
        # KMeans over every MFCC frame was one of the biggest
        # unnecessary CPU costs on Render Free.
        #
        # We deliberately do NOT run it here.
        #
        # This does not invent speaker identities.
        # It simply reports that speaker identification is
        # unavailable in the Render-optimized configuration.
        # ====================================================

        if progress_cb:
            progress_cb(
                88,
                "Finalizing audio analysis..."
            )

        speakers_detected = 0

        # ====================================================
        # FINISH
        # ====================================================

        elapsed = time.monotonic() - started

        logger.info(
            f"Audio analysis finished in {elapsed:.1f}s"
        )

        if progress_cb:
            progress_cb(
                94,
                "Generating report..."
            )

        return {
            "alerts": alerts,
            "timeline": timeline,
            "audio_anomalies": anomalies,
            "transcript": transcript,

            "summary_stats": {
                "audio_anomalies": len(anomalies),
                "speakers_detected": speakers_detected,
                "duration_sec": round(duration, 1),
                "analysis_time_sec": round(elapsed, 1),
            },

            "capabilities": self.capabilities,
        }

    # ========================================================
    # TIME LIMIT
    # ========================================================

    @staticmethod
    def _time_exceeded(started):

        return (
            time.monotonic() - started
            >= MAX_ANALYSIS_SECONDS
        )

    # ========================================================
    # FAST IMPULSE DETECTION
    # ========================================================

    def _detect_impulses_fast(
        self,
        y,
        sr,
        started
    ):

        alerts = []
        timeline = []
        anomalies = []

        # Analyse 2-second blocks.
        chunk_seconds = 2.0
        chunk_size = int(
            sr * chunk_seconds
        )

        if chunk_size <= 0:
            return {
                "alerts": alerts,
                "timeline": timeline,
                "anomalies": anomalies,
            }

        # Only inspect the allowed audio.
        total_chunks = (
            len(y) // chunk_size
        )

        # Avoid pathological huge loops.
        total_chunks = min(
            total_chunks,
            int(MAX_AUDIO_SECONDS / chunk_seconds)
        )

        previous_rms = 0.0

        for i in range(total_chunks):

            if self._time_exceeded(started):
                logger.warning(
                    "Audio analysis time budget reached."
                )
                break

            start = i * chunk_size
            end = start + chunk_size

            seg = y[start:end]

            if len(seg) < sr * 0.25:
                continue

            timestamp = i * chunk_seconds

            # ------------------------------------------------
            # RMS
            # ------------------------------------------------

            rms = float(
                np.sqrt(
                    np.mean(
                        np.square(seg)
                    ) + 1e-12
                )
            )

            if rms < 1e-6:
                continue

            db = 20.0 * np.log10(
                rms
            ) + 96.0

            # ------------------------------------------------
            # Peak / crest factor
            # ------------------------------------------------

            peak = float(
                np.max(
                    np.abs(seg)
                )
            )

            crest = peak / max(
                rms,
                1e-8
            )

            # ------------------------------------------------
            # Energy change
            # ------------------------------------------------

            energy_jump = (
                rms /
                max(previous_rms, rms * 0.1)
            )

            previous_rms = rms

            # ------------------------------------------------
            # Conservative impulse rule
            #
            # This is intentionally NOT called a confirmed
            # gunshot. It is a possible acoustic impulse.
            # ------------------------------------------------

            possible_impulse = (
                db >= 82.0
                and
                crest >= 5.0
                and
                energy_jump >= 1.5
            )

            if possible_impulse:

                confidence = 0.55

                if db >= 90:
                    confidence += 0.10

                if crest >= 7:
                    confidence += 0.10

                if energy_jump >= 2.5:
                    confidence += 0.10

                confidence = min(
                    0.85,
                    confidence
                )

                severity = (
                    SeverityLevel.HIGH
                    if confidence >= 0.75
                    else SeverityLevel.MEDIUM
                )

                ts = _ts(timestamp)

                alerts.append(
                    Alert(
                        timestamp=ts,
                        severity=severity,
                        alert_type=AlertType.AUDIO,
                        category="acoustic_impulse",
                        description=(
                            f"Possible acoustic impulse at "
                            f"{ts} "
                            f"(dB {db:.1f}, "
                            f"crest {crest:.1f}, "
                            f"confidence "
                            f"{confidence:.0%})"
                        ),
                        confidence=round(
                            confidence,
                            2
                        ),
                    )
                )

                timeline.append(
                    TimelineEvent(
                        timestamp=ts,
                        source=AlertType.AUDIO,
                        event=(
                            f"Acoustic impulse detected "
                            f"({confidence:.0%})"
                        ),
                        confidence=round(
                            confidence,
                            2
                        ),
                    )
                )

                anomalies.append(
                    AudioAnomaly(
                        timestamp=ts,
                        anomaly_type="acoustic_impulse",
                        confidence=round(
                            confidence,
                            2
                        ),
                        detail=(
                            f"dB:{db:.1f} "
                            f"crest:{crest:.1f}"
                        ),
                    )
                )

            # ------------------------------------------------
            # Loud sustained audio
            # ------------------------------------------------

            elif (
                db >= 80.0
                and crest < 5.0
            ):

                anomalies.append(
                    AudioAnomaly(
                        timestamp=_ts(timestamp),
                        anomaly_type="elevated_audio",
                        confidence=0.50,
                        detail=(
                            f"Sustained loud audio "
                            f"(dB {db:.1f})"
                        ),
                    )
                )

        return {
            "alerts": alerts,
            "timeline": timeline,
            "anomalies": anomalies,
        }

    # ========================================================
    # FAST WHISPER
    # ========================================================

    def _transcribe_fast(
        self,
        path,
        started
    ):

        timeline = []

        # Load model lazily.
        self._load_whisper()

        if self.whisper is None:

            timeline.append(
                TimelineEvent(
                    timestamp="00:00:00",
                    source=AlertType.AUDIO,
                    event=(
                        "[Speech transcription "
                        "unavailable]"
                    ),
                )
            )

            return "", timeline

        # ----------------------------------------------------
        # IMPORTANT:
        # Render Free CPU is extremely limited.
        # Don't let Whisper consume the entire job forever.
        # ----------------------------------------------------

        if self._time_exceeded(started):

            logger.warning(
                "Skipping Whisper: time budget reached."
            )

            return "", timeline

        try:

            logger.info(
                "Starting Whisper tiny transcription..."
            )

            result = self.whisper.transcribe(
                path,

                language="en",

                fp16=False,

                temperature=0,

                condition_on_previous_text=False,

                word_timestamps=False,

                verbose=False,

                # Keep decoding conservative.
                beam_size=1,

                best_of=1,
            )

            text = (
                result
                .get("text", "")
                .strip()
            )

            # ------------------------------------------------
            # Segment timeline
            # ------------------------------------------------

            for seg in result.get(
                "segments",
                []
            ):

                if self._time_exceeded(started):
                    break

                no_speech = float(
                    seg.get(
                        "no_speech_prob",
                        1.0
                    )
                )

                if no_speech < 0.5:

                    timestamp = _ts(
                        seg.get(
                            "start",
                            0
                        )
                    )

                    segment_text = (
                        seg.get(
                            "text",
                            ""
                        )
                        .strip()
                    )

                    if segment_text:

                        timeline.append(
                            TimelineEvent(
                                timestamp=timestamp,
                                source=AlertType.AUDIO,
                                event=(
                                    f'Speech: '
                                    f'"{segment_text}"'
                                ),
                                confidence=round(
                                    1.0 - no_speech,
                                    2
                                ),
                            )
                        )

            logger.info(
                "Whisper transcription completed."
            )

            return text, timeline

        except Exception as e:

            logger.error(
                f"Whisper transcription failed: {e}"
            )

            return "", timeline

    # ========================================================
    # KEYWORD DETECTION
    # ========================================================

    def _keyword_scan(
        self,
        transcript
    ):

        alerts = []
        timeline = []
        anomalies = []

        text = (
            transcript
            .lower()
        )

        words = set(
            text.split()
        )

        matched = []

        for keyword in settings.THREAT_KEYWORDS:

            keyword = (
                str(keyword)
                .lower()
                .strip()
            )

            if not keyword:
                continue

            if keyword in words:
                matched.append(
                    keyword
                )

        if matched:

            count = len(
                matched
            )

            if count >= 4:
                severity = SeverityLevel.HIGH
            elif count >= 2:
                severity = SeverityLevel.MEDIUM
            else:
                severity = SeverityLevel.LOW

            confidence = min(
                0.90,
                0.50 + count * 0.10
            )

            description = (
                "Threat keywords found "
                "in verified transcript: "
                + ", ".join(matched)
            )

            alerts.append(
                Alert(
                    timestamp="00:00:00",
                    severity=severity,
                    alert_type=AlertType.AUDIO,
                    category="keyword_alert",
                    description=description,
                    confidence=round(
                        confidence,
                        2
                    ),
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
                    confidence=round(
                        confidence,
                        2
                    ),
                )
            )

            anomalies.append(
                AudioAnomaly(
                    timestamp="00:00:00",
                    anomaly_type="keyword",
                    confidence=round(
                        confidence,
                        2
                    ),
                    detail=(
                        f"Matched {count} "
                        f"keyword(s): "
                        + ", ".join(matched)
                    ),
                )
            )

        return {
            "alerts": alerts,
            "timeline": timeline,
            "anomalies": anomalies,
        }