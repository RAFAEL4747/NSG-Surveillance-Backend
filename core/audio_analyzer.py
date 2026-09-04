"""
Audio Analysis Engine
Optimized for low-resource cloud deployment.

Principle:
Only report real detections. If Whisper is unavailable,
no transcript is fabricated.
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
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class AudioAnalyzer:

    # IMPORTANT:
    # Keep Whisper loaded between jobs instead of loading it repeatedly.
    _whisper_model = None
    _whisper_attempted = False

    def __init__(self):
        self.whisper = None
        self.capabilities = {}
        self._load_whisper()

    def _load_whisper(self):
        """
        Load Whisper only once per running server.
        """
        if AudioAnalyzer._whisper_attempted:
            self.whisper = AudioAnalyzer._whisper_model

            if self.whisper is not None:
                self.capabilities["transcription"] = True
            else:
                self.capabilities["transcription"] = False

            return

        AudioAnalyzer._whisper_attempted = True

        try:
            import whisper

            logger.info("Loading Whisper tiny model...")

            AudioAnalyzer._whisper_model = whisper.load_model(
                "tiny",
                device="cpu"
            )

            self.whisper = AudioAnalyzer._whisper_model
            self.capabilities["transcription"] = True

            logger.info("Whisper tiny loaded successfully")

        except Exception as e:
            AudioAnalyzer._whisper_model = None
            self.capabilities["transcription"] = False
            self.capabilities["transcription_reason"] = str(e)

            logger.warning(f"Whisper unavailable: {e}")

    def analyse(
        self,
        audio_path: str,
        progress_cb: Callable | None = None
    ) -> dict:

        import librosa

        if progress_cb:
            progress_cb(62, "Loading audio analyser...")

        # ---------------------------------------------------------
        # Load audio at 16 kHz mono.
        # This is much smaller/faster than loading the original
        # sampling rate.
        # ---------------------------------------------------------

        y, sr = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )

        duration = len(y) / sr

        alerts = []
        timeline = []
        anomalies = []
        transcript = ""

        # ---------------------------------------------------------
        # 1. Gunshot / impulse detection
        # ---------------------------------------------------------

        if progress_cb:
            progress_cb(66, "Gunshot detection...")

        gs = self._detect_impulses(y, sr)

        alerts.extend(gs["alerts"])
        timeline.extend(gs["timeline"])
        anomalies.extend(gs["anomalies"])

        # ---------------------------------------------------------
        # 2. Speech transcription
        # ---------------------------------------------------------

        if progress_cb:
            progress_cb(73, "Speech transcription...")

        if self.whisper:

            transcript, t_tl = self._transcribe(audio_path)

            timeline.extend(t_tl)

        else:

            timeline.append(
                TimelineEvent(
                    timestamp="00:00:00",
                    source=AlertType.AUDIO,
                    event="[Transcription unavailable]"
                )
            )

        # ---------------------------------------------------------
        # 3. Keyword scan
        # ---------------------------------------------------------

        if progress_cb:
            progress_cb(82, "Keyword scan...")

        if transcript:

            kw = self._keyword_scan(transcript)

            alerts.extend(kw["alerts"])
            timeline.extend(kw["timeline"])
            anomalies.extend(kw["anomalies"])

        # ---------------------------------------------------------
        # 4. Speaker identification
        # ---------------------------------------------------------

        if progress_cb:
            progress_cb(88, "Speaker identification...")

        sp = self._speaker_id(y, sr, duration)

        timeline.extend(sp["timeline"])
        anomalies.extend(sp["anomalies"])

        return {
            "alerts": alerts,
            "timeline": timeline,
            "audio_anomalies": anomalies,
            "transcript": transcript,

            "summary_stats": {
                "audio_anomalies": len(anomalies),
                "speakers_detected": sp["count"],
                "duration_sec": round(duration, 1),
            },

            "capabilities": self.capabilities,
        }

    # =============================================================
    # GUNSHOT / IMPULSE DETECTION
    # =============================================================

    def _detect_impulses(self, y, sr):

        import librosa

        alerts = []
        timeline = []
        anomalies = []

        chunk_sec = float(settings.AUDIO_CHUNK_SEC)

        chunk_size = int(chunk_sec * sr)

        if chunk_size <= 0:
            chunk_size = sr

        n_chunks = len(y) // chunk_size

        for i in range(n_chunks):

            start = i * chunk_size
            end = start + chunk_size

            seg = y[start:end]

            if len(seg) < sr * 0.1:
                continue

            t = i * chunk_sec
            ts = _ts(t)

            # Fast RMS calculation
            rms = float(np.sqrt(np.mean(seg ** 2)))

            if rms < 1e-7:
                continue

            db = 20 * np.log10(rms) + 96

            # Don't run expensive onset analysis on quiet audio.
            if db < 65:
                continue

            onset_env = librosa.onset.onset_strength(
                y=seg,
                sr=sr,
                hop_length=1024
            )

            if len(onset_env) == 0:
                continue

            peak_onset = float(np.max(onset_env))

            # Conservative gunshot test
            if (
                db > settings.GUNSHOT_DB_THRESHOLD
                and peak_onset > settings.GUNSHOT_ONSET_THRESHOLD
            ):

                onset_frames = librosa.onset.onset_detect(
                    y=seg,
                    sr=sr,
                    hop_length=1024,
                    backtrack=False
                )

                if len(onset_frames) <= 2:

                    conf = min(
                        0.95,
                        0.60
                        + (db - settings.GUNSHOT_DB_THRESHOLD) / 40
                        + (peak_onset - settings.GUNSHOT_ONSET_THRESHOLD) / 60
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
                            confidence=round(conf, 2),
                        )
                    )

                    timeline.append(
                        TimelineEvent(
                            timestamp=ts,
                            source=AlertType.AUDIO,
                            event=(
                                f"Acoustic impulse — "
                                f"possible gunshot ({conf:.0%})"
                            ),
                            confidence=round(conf, 2),
                        )
                    )

                    anomalies.append(
                        AudioAnomaly(
                            timestamp=ts,
                            anomaly_type="gunshot",
                            confidence=round(conf, 2),
                            detail=(
                                f"dB:{db:.1f} "
                                f"onset:{peak_onset:.1f} "
                                f"transients:{len(onset_frames)}"
                            ),
                        )
                    )

            # Sustained loud audio
            elif db > 78 and peak_onset > 6.0:

                onset_frames = librosa.onset.onset_detect(
                    y=seg,
                    sr=sr,
                    hop_length=1024
                )

                if len(onset_frames) > 3:

                    anomalies.append(
                        AudioAnomaly(
                            timestamp=ts,
                            anomaly_type="elevated_audio",
                            confidence=0.55,
                            detail=(
                                f"Sustained loud audio at {ts} "
                                f"(dB {db:.1f})"
                            ),
                        )
                    )

        return {
            "alerts": alerts,
            "timeline": timeline,
            "anomalies": anomalies
        }

    # =============================================================
    # WHISPER
    # =============================================================

    def _transcribe(self, path):

        timeline = []

        try:

            result = self.whisper.transcribe(
                path,
                language="en",
                word_timestamps=False,

                # CPU-friendly settings
                fp16=False,
                temperature=0,
                condition_on_previous_text=False,
                verbose=False,
            )

            text = result.get("text", "").strip()

            for seg in result.get("segments", []):

                if seg.get("no_speech_prob", 1.0) < 0.5:

                    ts = _ts(seg["start"])

                    timeline.append(
                        TimelineEvent(
                            timestamp=ts,
                            source=AlertType.AUDIO,
                            event=f'Speech: "{seg["text"].strip()}"',
                            confidence=round(
                                1 - seg.get("no_speech_prob", 0),
                                2
                            ),
                        )
                    )

            return text, timeline

        except Exception as e:

            logger.error(f"Transcription error: {e}")

            return "", []

    # =============================================================
    # KEYWORD SCAN
    # =============================================================

    def _keyword_scan(self, transcript):

        alerts = []
        timeline = []
        anomalies = []

        words = transcript.lower().split()

        matched = [
            kw
            for kw in settings.THREAT_KEYWORDS
            if kw.lower() in words
        ]

        if matched:

            n = len(matched)

            sev = (
                SeverityLevel.HIGH
                if n >= 4
                else SeverityLevel.MEDIUM
                if n >= 2
                else SeverityLevel.LOW
            )

            conf = min(0.92, 0.50 + n * 0.10)

            alerts.append(
                Alert(
                    timestamp="00:00:00",
                    severity=sev,
                    alert_type=AlertType.AUDIO,
                    category="keyword_alert",
                    description=(
                        f"Threat keywords in transcript: "
                        f"{', '.join(matched)}"
                    ),
                    confidence=conf,
                )
            )

            timeline.append(
                TimelineEvent(
                    timestamp="00:00:00",
                    source=AlertType.AUDIO,
                    event=(
                        f"Keywords matched: "
                        f"{', '.join(matched)}"
                    )
                )
            )

            anomalies.append(
                AudioAnomaly(
                    timestamp="00:00:00",
                    anomaly_type="keyword",
                    confidence=conf,
                    detail=(
                        f"Matched {n} keyword(s): "
                        f"{', '.join(matched)}"
                    )
                )
            )

        return {
            "alerts": alerts,
            "timeline": timeline,
            "anomalies": anomalies
        }

    # =============================================================
    # SPEAKER IDENTIFICATION
    # =============================================================

    def _speaker_id(self, y, sr, duration):

        timeline = []
        anomalies = []

        try:

            import librosa

            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            # -----------------------------------------------------
            # Limit amount of audio used for speaker clustering.
            # This prevents huge KMeans operations.
            # -----------------------------------------------------

            max_duration = min(duration, 300)

            samples = int(max_duration * sr)

            y_sample = y[:samples]

            # Use a larger hop to drastically reduce feature count.
            hop = 2048

            mfcc = librosa.feature.mfcc(
                y=y_sample,
                sr=sr,
                n_mfcc=13,
                hop_length=hop
            )

            delta = librosa.feature.delta(mfcc)

            feat = np.vstack([
                mfcc,
                delta
            ]).T

            # Keep maximum number of clustering samples manageable.
            max_frames = 1500

            if len(feat) > max_frames:

                indexes = np.linspace(
                    0,
                    len(feat) - 1,
                    max_frames
                ).astype(int)

                feat = feat[indexes]

            if len(feat) < 20:

                return {
                    "timeline": [],
                    "anomalies": [],
                    "count": 0
                }

            n_sp = min(
                4,
                max(2, int(duration // 30))
            )

            # Don't ask KMeans for more clusters than data supports.
            n_sp = min(n_sp, len(feat))

            X = StandardScaler().fit_transform(feat)

            labels = KMeans(
                n_clusters=n_sp,
                random_state=42,
                n_init=3,
                max_iter=100
            ).fit_predict(X)

            seen = set()

            previous = labels[0]

            for i, label in enumerate(labels):

                if label != previous:

                    sid = f"SPK-{previous + 1:02d}"

                    if sid not in seen:

                        seen.add(sid)

                        timestamp = (
                            i * hop / sr
                        )

                        timeline.append(
                            TimelineEvent(
                                timestamp=_ts(timestamp),
                                source=AlertType.AUDIO,
                                event=f"Speaker {sid} identified"
                            )
                        )

                        if len(seen) > 1:

                            anomalies.append(
                                AudioAnomaly(
                                    timestamp=_ts(timestamp),
                                    anomaly_type="new_speaker",
                                    confidence=0.65,
                                    detail=f"New speaker: {sid}"
                                )
                            )

                    previous = label

            return {
                "timeline": timeline,
                "anomalies": anomalies,
                "count": n_sp
            }

        except Exception as e:

            logger.warning(
                f"Speaker ID failed: {e}"
            )

            return {
                "timeline": [],
                "anomalies": [],
                "count": 0
            }