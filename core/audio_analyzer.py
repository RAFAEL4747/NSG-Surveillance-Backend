"""
Audio Analysis Engine

KEY PRINCIPLE:
Only report real detections. If Whisper is unavailable, we report that
transcription is unavailable — we never invent transcript content.

Performance:
- Whisper uses the tiny model.
- Whisper is loaded lazily, only when transcription is needed.
- Whisper is cached between jobs during the lifetime of the server.
- Speaker analysis is limited to 5 minutes and uses a larger hop length.
- CPU-safe Whisper settings are used for Render.
"""

import os
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


def _ts(sec: float) -> str:
    """Convert seconds to HH:MM:SS."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)

    return f"{h:02d}:{m:02d}:{s:02d}"


class AudioAnalyzer:

    # ---------------------------------------------------------
    # Whisper is shared between jobs.
    # This prevents loading the model repeatedly.
    # ---------------------------------------------------------
    _whisper_model = None
    _whisper_attempted = False

    def __init__(self):
        self.whisper = None

        self.capabilities = {
            "transcription": True
        }

    # ---------------------------------------------------------
    # LOAD WHISPER
    # ---------------------------------------------------------
    def _load_whisper(self):
        """
        Load Whisper only once per running server.

        The model is NOT loaded when AudioAnalyzer is created.
        It is loaded only when transcription is actually required.
        """

        # Already attempted during this server session
        if AudioAnalyzer._whisper_attempted:
            self.whisper = AudioAnalyzer._whisper_model

            if self.whisper is None:
                self.capabilities["transcription"] = False

            return

        AudioAnalyzer._whisper_attempted = True

        try:
            import whisper

            logger.info("Loading Whisper tiny model...")

            AudioAnalyzer._whisper_model = whisper.load_model("tiny")

            self.whisper = AudioAnalyzer._whisper_model

            self.capabilities["transcription"] = True

            logger.info("Whisper tiny model loaded successfully")

        except Exception as e:

            AudioAnalyzer._whisper_model = None
            self.whisper = None

            self.capabilities["transcription"] = False
            self.capabilities["transcription_reason"] = str(e)

            logger.warning(
                f"Whisper unavailable: {e}"
            )

    # ---------------------------------------------------------
    # MAIN AUDIO ANALYSIS
    # ---------------------------------------------------------
    def analyse(
        self,
        audio_path: str,
        progress_cb: Callable | None = None
    ) -> dict:

        import librosa

        # -----------------------------------------------------
        # AUDIO INGESTION
        # -----------------------------------------------------
        if progress_cb:
            progress_cb(
                62,
                "Loading audio..."
            )

        try:
            y, sr = librosa.load(
                audio_path,
                sr=None,
                mono=True
            )

            duration = librosa.get_duration(
                y=y,
                sr=sr
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
                "capabilities": {
                    "transcription": False,
                    "audio_loading": False,
                    "audio_loading_reason": str(e),
                },
            }

        alerts = []
        timeline = []
        anomalies = []

        transcript = ""

        # -----------------------------------------------------
        # 1. GUNSHOT / IMPULSE DETECTION
        # -----------------------------------------------------
        if progress_cb:
            progress_cb(
                66,
                "Gunshot detection..."
            )

        gs = self._detect_impulses(
            y,
            sr
        )

        alerts.extend(
            gs["alerts"]
        )

        timeline.extend(
            gs["timeline"]
        )

        anomalies.extend(
            gs["anomalies"]
        )

        # -----------------------------------------------------
        # 2. SPEECH TRANSCRIPTION
        # -----------------------------------------------------
        if progress_cb:
            progress_cb(
                73,
                "Speech transcription..."
            )

        # Load Whisper only now.
        self._load_whisper()

        if self.whisper:

            transcript, t_tl = self._transcribe(
                audio_path
            )

            timeline.extend(
                t_tl
            )

        else:

            transcript = ""

            timeline.append(
                TimelineEvent(
                    timestamp="00:00:00",
                    source=AlertType.AUDIO,
                    event=(
                        "[Transcription unavailable — "
                        "Whisper could not be loaded]"
                    ),
                )
            )

        # -----------------------------------------------------
        # 3. KEYWORD SCAN
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
                "Speaker identification..."
            )

        sp = self._speaker_id(
            y,
            sr,
            duration
        )

        timeline.extend(
            sp["timeline"]
        )

        anomalies.extend(
            sp["anomalies"]
        )

        # -----------------------------------------------------
        # FINAL RESULT
        # -----------------------------------------------------
        return {
            "alerts": alerts,
            "timeline": timeline,
            "audio_anomalies": anomalies,
            "transcript": transcript,

            "summary_stats": {
                "audio_anomalies": len(anomalies),
                "speakers_detected": sp["count"],
                "duration_sec": round(
                    duration,
                    1
                ),
            },

            "capabilities": self.capabilities,
        }

    # ---------------------------------------------------------
    # GUNSHOT / IMPULSE DETECTION
    # ---------------------------------------------------------
    def _detect_impulses(
        self,
        y,
        sr
    ):

        import librosa

        alerts = []
        timeline = []
        anomalies = []

        # Make sure chunk is an integer.
        chunk = int(
            settings.AUDIO_CHUNK_SEC * sr
        )

        if chunk <= 0:
            chunk = int(sr)

        n_chunks = len(y) // chunk

        for i in range(n_chunks):

            start = i * chunk
            end = (i + 1) * chunk

            seg = y[start:end]

            t = (
                i *
                settings.AUDIO_CHUNK_SEC
            )

            ts = _ts(t)

            if len(seg) == 0:
                continue

            # -------------------------------------------------
            # RMS
            # -------------------------------------------------
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

            # -------------------------------------------------
            # ONSET
            # -------------------------------------------------
            try:

                onset_env = (
                    librosa.onset.onset_strength(
                        y=seg,
                        sr=sr
                    )
                )

                if len(onset_env) == 0:
                    continue

                peak_onset = float(
                    np.max(onset_env)
                )

            except Exception as e:

                logger.warning(
                    f"Onset detection failed: {e}"
                )

                continue

            # -------------------------------------------------
            # GUNSHOT DETECTION
            # -------------------------------------------------
            if (
                db >
                settings.GUNSHOT_DB_THRESHOLD
                and
                peak_onset >
                settings.GUNSHOT_ONSET_THRESHOLD
            ):

                try:

                    onset_frames = (
                        librosa.onset.onset_detect(
                            y=seg,
                            sr=sr
                        )
                    )

                except Exception:

                    onset_frames = []

                # Single sharp transient
                if len(onset_frames) <= 2:

                    conf = min(
                        0.95,
                        0.60
                        +
                        (
                            db
                            -
                            settings.GUNSHOT_DB_THRESHOLD
                        ) / 40
                        +
                        (
                            peak_onset
                            -
                            settings.GUNSHOT_ONSET_THRESHOLD
                        ) / 60
                    )

                    sev = (
                        SeverityLevel.HIGH
                        if conf > 0.80
                        else SeverityLevel.MEDIUM
                    )

                    alerts.append(
                        Alert(
                            timestamp=ts,
                            severity=sev,
                            alert_type=AlertType.AUDIO,
                            category="gunshot_detected",
                            description=(
                                f"Impulse event at {ts}: "
                                f"possible gunshot/explosion "
                                f"(dB {db:.1f}, "
                                f"onset {peak_onset:.1f}, "
                                f"conf {conf:.0%})"
                            ),
                            confidence=round(
                                conf,
                                2
                            ),
                        )
                    )

                    timeline.append(
                        TimelineEvent(
                            timestamp=ts,
                            source=AlertType.AUDIO,
                            event=(
                                "Acoustic impulse — "
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
                            timestamp=ts,
                            anomaly_type="gunshot",
                            confidence=round(
                                conf,
                                2
                            ),
                            detail=(
                                f"dB:{db:.1f} "
                                f"onset:{peak_onset:.1f} "
                                f"transients:"
                                f"{len(onset_frames)}"
                            ),
                        )
                    )

            # -------------------------------------------------
            # SUSTAINED LOUD AUDIO
            # -------------------------------------------------
            elif (
                db > 78
                and
                peak_onset > 6.0
            ):

                try:

                    onset_frames = (
                        librosa.onset.onset_detect(
                            y=seg,
                            sr=sr
                        )
                    )

                except Exception:

                    onset_frames = []

                if len(onset_frames) > 3:

                    anomalies.append(
                        AudioAnomaly(
                            timestamp=ts,
                            anomaly_type="elevated_audio",
                            confidence=0.55,
                            detail=(
                                f"Sustained loud audio "
                                f"at {ts} "
                                f"(dB {db:.1f})"
                            ),
                        )
                    )

        return {
            "alerts": alerts,
            "timeline": timeline,
            "anomalies": anomalies,
        }

    # ---------------------------------------------------------
    # WHISPER TRANSCRIPTION
    # ---------------------------------------------------------
    def _transcribe(
        self,
        path
    ):

        timeline = []

        if self.whisper is None:
            return "", timeline

        try:

            logger.info(
                "Starting Whisper tiny transcription..."
            )

            result = self.whisper.transcribe(
                path,

                # English is faster than automatic
                # language detection.
                language="en",

                word_timestamps=False,

                # Render CPU-safe setting.
                fp16=False,

                # Avoid unnecessary verbose output.
                verbose=False,
            )

            text = (
                result
                .get("text", "")
                .strip()
            )

            # -------------------------------------------------
            # Add real speech segments to timeline.
            # -------------------------------------------------
            for seg in result.get(
                "segments",
                []
            ):

                if (
                    seg.get(
                        "no_speech_prob",
                        1.0
                    )
                    < 0.5
                ):

                    ts = _ts(
                        seg["start"]
                    )

                    timeline.append(
                        TimelineEvent(
                            timestamp=ts,
                            source=AlertType.AUDIO,
                            event=(
                                f'Speech: '
                                f'"{seg["text"].strip()}"'
                            ),
                            confidence=round(
                                1 -
                                seg.get(
                                    "no_speech_prob",
                                    0
                                ),
                                2
                            ),
                        )
                    )

            logger.info(
                "Whisper transcription completed"
            )

            return text, timeline

        except Exception as e:

            logger.error(
                f"Transcription error: {e}"
            )

            return "", []

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

        words = (
            transcript
            .lower()
            .split()
        )

        matched = [
            kw
            for kw in settings.THREAT_KEYWORDS
            if kw in words
        ]

        if matched:

            n = len(matched)

            # 1 keyword = LOW
            # 2-3 keywords = MEDIUM
            # 4+ keywords = HIGH
            sev = (
                SeverityLevel.HIGH
                if n >= 4
                else
                SeverityLevel.MEDIUM
                if n >= 2
                else
                SeverityLevel.LOW
            )

            conf = min(
                0.92,
                0.50 + n * 0.10
            )

            alerts.append(
                Alert(
                    timestamp="00:00:00",
                    severity=sev,
                    alert_type=AlertType.AUDIO,
                    category="keyword_alert",
                    description=(
                        "Threat keywords in transcript: "
                        +
                        ", ".join(matched)
                    ),
                    confidence=conf,
                )
            )

            timeline.append(
                TimelineEvent(
                    timestamp="00:00:00",
                    source=AlertType.AUDIO,
                    event=(
                        "Keywords matched: "
                        +
                        ", ".join(matched)
                    ),
                )
            )

            anomalies.append(
                AudioAnomaly(
                    timestamp="00:00:00",
                    anomaly_type="keyword",
                    confidence=conf,
                    detail=(
                        f"Matched {n} keyword(s): "
                        +
                        ", ".join(matched)
                    ),
                )
            )

        return {
            "alerts": alerts,
            "timeline": timeline,
            "anomalies": anomalies,
        }

    # ---------------------------------------------------------
    # SPEAKER IDENTIFICATION
    # ---------------------------------------------------------
    def _speaker_id(
        self,
        y,
        sr,
        duration
    ):

        timeline = []
        anomalies = []

        try:

            import librosa

            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            # -------------------------------------------------
            # PERFORMANCE LIMIT
            #
            # Never process more than 5 minutes for speaker ID.
            # -------------------------------------------------
            max_seconds = 300

            max_samples = int(
                sr * max_seconds
            )

            if len(y) > max_samples:

                y_speaker = y[
                    :max_samples
                ]

            else:

                y_speaker = y

            # -------------------------------------------------
            # MFCC
            #
            # Larger hop_length = fewer frames = much faster.
            # -------------------------------------------------
            mfcc = librosa.feature.mfcc(
                y=y_speaker,
                sr=sr,
                n_mfcc=13,
                hop_length=2048
            )

            if mfcc.shape[1] < 10:

                return {
                    "timeline": [],
                    "anomalies": [],
                    "count": 1,
                }

            # -------------------------------------------------
            # MFCC DELTA
            # -------------------------------------------------
            delta = librosa.feature.delta(
                mfcc
            )

            feat = np.vstack(
                [
                    mfcc,
                    delta
                ]
            ).T

            # -------------------------------------------------
            # Determine number of speakers.
            #
            # Keep it conservative.
            # -------------------------------------------------
            n_sp = min(
                4,
                max(
                    2,
                    int(
                        duration // 30
                    )
                )
            )

            # Never ask KMeans for more clusters
            # than available samples.
            n_sp = min(
                n_sp,
                len(feat)
            )

            if n_sp < 2:

                return {
                    "timeline": [],
                    "anomalies": [],
                    "count": 1,
                }

            # -------------------------------------------------
            # SCALE FEATURES
            # -------------------------------------------------
            X = StandardScaler().fit_transform(
                feat
            )

            # -------------------------------------------------
            # KMEANS
            #
            # n_init=3 instead of 10 for speed.
            # -------------------------------------------------
            labels = KMeans(
                n_clusters=n_sp,
                random_state=42,
                n_init=3
            ).fit_predict(X)

            # -------------------------------------------------
            # DETECT SPEAKER CHANGES
            # -------------------------------------------------
            seen = set()

            hop = 2048

            start = 0.0
            prev = labels[0]

            for i, lbl in enumerate(
                labels
            ):

                if lbl != prev:

                    sid = (
                        f"SPK-{prev + 1:02d}"
                    )

                    if sid not in seen:

                        seen.add(
                            sid
                        )

                        timeline.append(
                            TimelineEvent(
                                timestamp=_ts(
                                    start
                                ),
                                source=AlertType.AUDIO,
                                event=(
                                    f"Speaker "
                                    f"{sid} identified"
                                ),
                            )
                        )

                        # New speaker after
                        # the first speaker.
                        if len(seen) > 1:

                            anomalies.append(
                                AudioAnomaly(
                                    timestamp=_ts(
                                        start
                                    ),
                                    anomaly_type=(
                                        "new_speaker"
                                    ),
                                    confidence=0.65,
                                    detail=(
                                        f"New speaker: "
                                        f"{sid}"
                                    ),
                                )
                            )

                    start = (
                        i *
                        hop /
                        sr
                    )

                    prev = lbl

            # Add the final speaker if one exists.
            final_sid = (
                f"SPK-{prev + 1:02d}"
            )

            if final_sid not in seen:

                seen.add(
                    final_sid
                )

                timeline.append(
                    TimelineEvent(
                        timestamp=_ts(
                            start
                        ),
                        source=AlertType.AUDIO,
                        event=(
                            f"Speaker "
                            f"{final_sid} identified"
                        ),
                    )
                )

            return {
                "timeline": timeline,
                "anomalies": anomalies,
                "count": len(seen),
            }

        except Exception as e:

            logger.warning(
                f"Speaker ID failed: {e}"
            )

            return {
                "timeline": [],
                "anomalies": [],
                "count": 0,
            }