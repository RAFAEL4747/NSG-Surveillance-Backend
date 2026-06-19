"""
Audio Analysis Engine
Handles: gunshot / impulse detection · speech transcription · keyword alerting ·
         speaker identification · sentiment scoring
"""

import os
import logging
import numpy as np
from typing import Callable
from pathlib import Path

import librosa
import soundfile as sf

from core.config import settings
from core.schemas import (
    Alert, AlertType, AudioAnomaly, SeverityLevel, TimelineEvent,
)

logger = logging.getLogger(__name__)

# ── Threat-level sentiment words (expand as needed) ───────────────────────────
NEGATIVE_WORDS = {
    "kill", "attack", "bomb", "destroy", "shoot", "fire", "breach",
    "danger", "threat", "emergency", "alarm",
}


def _sec_to_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class AudioAnalyzer:
    def __init__(self):
        self.whisper_model = None    # loaded lazily
        self._load_whisper()

    def _load_whisper(self):
        try:
            import whisper
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper ASR model loaded (base)")
        except ImportError:
            logger.warning("openai-whisper not installed — transcription disabled")
        except Exception as e:
            logger.warning(f"Whisper load failed ({e}) — transcription disabled")

    # ── Public entry point ────────────────────────────────────────────────────

    def analyse(
        self,
        audio_path: str,
        progress_cb: Callable[[int, str], None] | None = None,
    ) -> dict:
        """
        Full audio analysis pipeline.
        Returns: alerts, timeline, audio_anomalies, transcript, summary_stats.
        """
        if progress_cb:
            progress_cb(62, "Loading audio file…")

        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        logger.info(f"Audio: {duration:.1f}s @ {sr}Hz")

        alerts:          list[Alert]         = []
        timeline:        list[TimelineEvent] = []
        audio_anomalies: list[AudioAnomaly]  = []
        transcript_full: str                 = ""

        # ── 1. Gunshot / impulse detection ───────────────────────────────────
        if progress_cb:
            progress_cb(65, "Running gunshot detection…")
        gs_results = self._detect_impulse_events(y, sr)
        alerts.extend(gs_results["alerts"])
        timeline.extend(gs_results["timeline"])
        audio_anomalies.extend(gs_results["anomalies"])

        # ── 2. Speech transcription ───────────────────────────────────────────
        if progress_cb:
            progress_cb(72, "Transcribing speech…")
        if self.whisper_model:
            transcript_full, t_timeline = self._transcribe(audio_path)
            timeline.extend(t_timeline)
        else:
            transcript_full = "[Transcription unavailable — install openai-whisper]"
            timeline.append(TimelineEvent(
                timestamp="00:00:00", source=AlertType.AUDIO,
                event="Transcription skipped — Whisper model not available",
            ))

        # ── 3. Keyword & threat alerting ──────────────────────────────────────
        if progress_cb:
            progress_cb(80, "Scanning for threat keywords…")
        kw_results = self._scan_keywords(transcript_full)
        alerts.extend(kw_results["alerts"])
        timeline.extend(kw_results["timeline"])
        audio_anomalies.extend(kw_results["anomalies"])

        # ── 4. Sentiment scoring ──────────────────────────────────────────────
        if progress_cb:
            progress_cb(85, "Computing audio sentiment…")
        sentiment = self._score_sentiment(transcript_full)
        if sentiment["threat_score"] > 0.6:
            alerts.append(Alert(
                timestamp="00:00:00",
                severity=SeverityLevel.MEDIUM,
                alert_type=AlertType.AUDIO,
                category="sentiment_alert",
                description=f"High threat sentiment detected in audio — score {sentiment['threat_score']:.2f}",
                confidence=sentiment["threat_score"],
            ))

        # ── 5. Speaker identification ─────────────────────────────────────────
        if progress_cb:
            progress_cb(88, "Identifying speakers…")
        speaker_results = self._identify_speakers(y, sr, duration)
        timeline.extend(speaker_results["timeline"])
        audio_anomalies.extend(speaker_results["anomalies"])

        summary_stats = {
            "audio_anomalies": len(audio_anomalies),
            "transcript_length": len(transcript_full.split()),
            "speakers_detected": speaker_results["count"],
            "duration_sec": round(duration, 1),
        }

        return {
            "alerts":          alerts,
            "timeline":        timeline,
            "audio_anomalies": audio_anomalies,
            "transcript":      transcript_full,
            "sentiment":       sentiment,
            "summary_stats":   summary_stats,
        }

    # ── Gunshot / impulse detection ───────────────────────────────────────────

    def _detect_impulse_events(self, y: np.ndarray, sr: int) -> dict:
        """
        Detect sudden high-energy impulse events (gunshots, explosions).
        Method: short-time RMS energy + onset strength spike detection.
        """
        alerts    = []
        timeline  = []
        anomalies = []

        chunk = settings.AUDIO_CHUNK_SEC * sr
        n_chunks = len(y) // chunk

        for i in range(n_chunks):
            segment = y[i * chunk:(i + 1) * chunk]
            time_sec = i * settings.AUDIO_CHUNK_SEC
            ts = _sec_to_ts(time_sec)

            # RMS energy in dBFS
            rms = np.sqrt(np.mean(segment ** 2))
            if rms == 0:
                continue
            db = 20 * np.log10(rms + 1e-9) + 96  # approximate dBSPL

            # Onset strength — high value = sudden transient
            onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
            peak_onset = float(np.max(onset_env))

            # Gunshot heuristic: high dB + high onset within very short window
            if db > settings.GUNSHOT_DB_THRESHOLD and peak_onset > 15.0:
                conf = min(0.99, (db - settings.GUNSHOT_DB_THRESHOLD) / 30 + peak_onset / 60)
                severity = SeverityLevel.HIGH if conf > 0.75 else SeverityLevel.MEDIUM

                alerts.append(Alert(
                    timestamp=ts, severity=severity,
                    alert_type=AlertType.AUDIO,
                    category="gunshot_detected",
                    description=f"Acoustic impulse event at {ts} — possible gunshot/explosion (dB: {db:.1f}, onset: {peak_onset:.1f})",
                    confidence=round(conf, 2),
                ))
                timeline.append(TimelineEvent(
                    timestamp=ts, source=AlertType.AUDIO,
                    event=f"Impulse event detected — possible gunshot (confidence {conf:.0%})",
                    confidence=round(conf, 2),
                ))
                anomalies.append(AudioAnomaly(
                    timestamp=ts, anomaly_type="gunshot",
                    confidence=round(conf, 2),
                    detail=f"dB level: {db:.1f}, onset strength: {peak_onset:.1f}",
                ))

            # Scream / loud voice detection
            elif db > 75 and peak_onset > 8.0:
                alerts.append(Alert(
                    timestamp=ts, severity=SeverityLevel.MEDIUM,
                    alert_type=AlertType.AUDIO,
                    category="scream_detected",
                    description=f"Elevated vocal stress / possible scream detected at {ts}",
                    confidence=0.65,
                ))
                anomalies.append(AudioAnomaly(
                    timestamp=ts, anomaly_type="scream",
                    confidence=0.65, detail=f"dB: {db:.1f}",
                ))

        return {"alerts": alerts, "timeline": timeline, "anomalies": anomalies}

    # ── Speech transcription ──────────────────────────────────────────────────

    def _transcribe(self, audio_path: str) -> tuple[str, list[TimelineEvent]]:
        """Transcribe audio using OpenAI Whisper."""
        timeline = []
        try:
            result = self.whisper_model.transcribe(audio_path, language="en", word_timestamps=False)
            full_text = result.get("text", "").strip()

            for seg in result.get("segments", []):
                ts = _sec_to_ts(seg["start"])
                timeline.append(TimelineEvent(
                    timestamp=ts, source=AlertType.AUDIO,
                    event=f'Speech: "{seg["text"].strip()}"',
                    confidence=round(1 - seg.get("no_speech_prob", 0), 2),
                ))

            return full_text, timeline
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", []

    # ── Keyword scanning ──────────────────────────────────────────────────────

    def _scan_keywords(self, transcript: str) -> dict:
        alerts    = []
        timeline  = []
        anomalies = []

        words_lower = transcript.lower().split()
        matched = [kw for kw in settings.THREAT_KEYWORDS if kw in words_lower]

        if matched:
            severity = SeverityLevel.HIGH if len(matched) >= 3 else SeverityLevel.MEDIUM
            alerts.append(Alert(
                timestamp="00:00:00", severity=severity,
                alert_type=AlertType.AUDIO,
                category="keyword_alert",
                description=f"Threat keywords detected in transcript: {', '.join(matched)}",
                confidence=min(0.95, 0.5 + len(matched) * 0.1),
            ))
            timeline.append(TimelineEvent(
                timestamp="00:00:00", source=AlertType.AUDIO,
                event=f"Keyword match(es): {', '.join(matched)}",
            ))
            anomalies.append(AudioAnomaly(
                timestamp="00:00:00", anomaly_type="keyword",
                confidence=min(0.95, 0.5 + len(matched) * 0.1),
                detail=f"Matched: {', '.join(matched)}",
            ))

        return {"alerts": alerts, "timeline": timeline, "anomalies": anomalies}

    # ── Sentiment scoring ─────────────────────────────────────────────────────

    def _score_sentiment(self, transcript: str) -> dict:
        """
        Simple lexical threat-sentiment scorer.
        Production: replace with fine-tuned BERT threat classifier.
        """
        words  = transcript.lower().split()
        if not words:
            return {"threat_score": 0.0, "negative_count": 0}

        threat_words = settings.THREAT_KEYWORDS + list(NEGATIVE_WORDS)
        matches      = sum(1 for w in words if w in threat_words)
        score        = min(1.0, matches / max(len(words), 1) * 20)

        return {"threat_score": round(score, 3), "negative_count": matches}

    # ── Speaker identification ────────────────────────────────────────────────

    def _identify_speakers(self, y: np.ndarray, sr: int, duration: float) -> dict:
        """
        Speaker diarisation stub using MFCC clustering.
        Production: replace with pyannote.audio speaker diarization pipeline.
        """
        timeline  = []
        anomalies = []

        try:
            # Extract MFCCs and cluster into speaker segments
            mfcc         = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_delta   = librosa.feature.delta(mfcc)
            features     = np.vstack([mfcc, mfcc_delta]).T    # (frames, 26)

            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            n_speakers = min(4, max(2, int(duration // 30)))
            scaler     = StandardScaler()
            X          = scaler.fit_transform(features)
            km         = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
            labels     = km.fit_predict(X)

            # Map frame labels → time segments
            hop    = 512
            prev   = labels[0]
            start  = 0.0
            seen_speakers = set()
            for i, lbl in enumerate(labels):
                if lbl != prev:
                    t_start = _sec_to_ts(start)
                    speaker_id = f"SPK-{prev+1:02d}"
                    seen_speakers.add(speaker_id)
                    is_new = speaker_id not in seen_speakers
                    timeline.append(TimelineEvent(
                        timestamp=t_start, source=AlertType.AUDIO,
                        event=f"Speaker {speaker_id} active",
                    ))
                    if is_new:
                        anomalies.append(AudioAnomaly(
                            timestamp=t_start, anomaly_type="new_speaker",
                            confidence=0.70,
                            detail=f"New speaker identified: {speaker_id}",
                        ))
                    start = i * hop / sr
                    prev  = lbl

            return {"timeline": timeline, "anomalies": anomalies, "count": n_speakers}

        except ImportError:
            logger.warning("scikit-learn not installed — speaker ID disabled")
            return {"timeline": [], "anomalies": [], "count": 0}
        except Exception as e:
            logger.warning(f"Speaker ID failed: {e}")
            return {"timeline": [], "anomalies": [], "count": 0}
