"""
Audio Analysis Engine
KEY PRINCIPLE: Only report real detections. If Whisper is absent we transcribe
nothing — we do not invent transcript content.
All thresholds are set conservatively to minimise false positives.
"""
import os, logging, numpy as np
from typing import Callable
from core.config import settings
from core.schemas import Alert, AlertType, AudioAnomaly, SeverityLevel, TimelineEvent

logger = logging.getLogger(__name__)

def _ts(sec: float) -> str:
    h,m,s = int(sec//3600), int((sec%3600)//60), int(sec%60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class AudioAnalyzer:
    def __init__(self):
        self.whisper    = None
        self.capabilities = {}
        self._load_whisper()

    def _load_whisper(self):
        try:
            import whisper
            self.whisper = whisper.load_model("tiny")
            self.capabilities["transcription"] = True
            logger.info("Whisper loaded")
        except Exception as e:
            self.capabilities["transcription"] = False
            self.capabilities["transcription_reason"] = str(e)
            logger.warning(f"Whisper unavailable: {e}")

    def analyse(self, audio_path: str, progress_cb: Callable | None = None) -> dict:
        import librosa
        if progress_cb: progress_cb(62, "Loading audio…")

        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        alerts, timeline, anomalies = [], [], []
        transcript = ""

        # 1. Gunshot / impulse detection
        if progress_cb: progress_cb(66, "Gunshot detection…")
        gs = self._detect_impulses(y, sr)
        alerts.extend(gs["alerts"]); timeline.extend(gs["timeline"]); anomalies.extend(gs["anomalies"])

        # 2. Transcription
        if progress_cb: progress_cb(73, "Speech transcription…")
        if self.whisper:
            transcript, t_tl = self._transcribe(audio_path)
            timeline.extend(t_tl)
        else:
            timeline.append(TimelineEvent(
                timestamp="00:00:00", source=AlertType.AUDIO,
                event="[Transcription unavailable — install openai-whisper]"
            ))

        # 3. Keyword scan (only if we have a real transcript)
        if progress_cb: progress_cb(82, "Keyword scan…")
        if transcript:
            kw = self._keyword_scan(transcript)
            alerts.extend(kw["alerts"]); timeline.extend(kw["timeline"]); anomalies.extend(kw["anomalies"])

        # 4. Speaker identification
        if progress_cb: progress_cb(88, "Speaker identification…")
        sp = self._speaker_id(y, sr, duration)
        timeline.extend(sp["timeline"]); anomalies.extend(sp["anomalies"])

        return {
            "alerts": alerts, "timeline": timeline, "audio_anomalies": anomalies,
            "transcript": transcript,
            "summary_stats": {
                "audio_anomalies": len(anomalies),
                "speakers_detected": sp["count"],
                "duration_sec": round(duration, 1),
            },
            "capabilities": self.capabilities,
        }

    def _detect_impulses(self, y, sr):
        import librosa
        alerts, timeline, anomalies = [], [], []
        chunk = settings.AUDIO_CHUNK_SEC * sr
        n_chunks = len(y) // chunk

        for i in range(n_chunks):
            seg  = y[i*chunk:(i+1)*chunk]
            t    = i * settings.AUDIO_CHUNK_SEC
            ts   = _ts(t)

            rms  = float(np.sqrt(np.mean(seg**2)))
            if rms < 1e-7: continue
            db   = 20 * np.log10(rms) + 96

            onset_env   = librosa.onset.onset_strength(y=seg, sr=sr)
            peak_onset  = float(np.max(onset_env))

            # Gunshot: BOTH db AND onset must clear their thresholds
            # This prevents door slams, music beats, etc from triggering
            if db > settings.GUNSHOT_DB_THRESHOLD and peak_onset > settings.GUNSHOT_ONSET_THRESHOLD:
                # Additional check: the transient must be very short (< 0.1s)
                onset_frames = librosa.onset.onset_detect(y=seg, sr=sr)
                if len(onset_frames) <= 2:   # single sharp transient
                    conf = min(0.95, 0.60 + (db - settings.GUNSHOT_DB_THRESHOLD)/40
                                           + (peak_onset - settings.GUNSHOT_ONSET_THRESHOLD)/60)
                    sev  = SeverityLevel.HIGH if conf > 0.80 else SeverityLevel.MEDIUM
                    alerts.append(Alert(
                        timestamp=ts, severity=sev,
                        alert_type=AlertType.AUDIO, category="gunshot_detected",
                        description=f"Impulse event at {ts}: possible gunshot/explosion (dB {db:.1f}, onset {peak_onset:.1f}, conf {conf:.0%})",
                        confidence=round(conf,2),
                    ))
                    timeline.append(TimelineEvent(timestamp=ts, source=AlertType.AUDIO,
                        event=f"Acoustic impulse — possible gunshot ({conf:.0%})", confidence=round(conf,2)))
                    anomalies.append(AudioAnomaly(timestamp=ts, anomaly_type="gunshot",
                        confidence=round(conf,2), detail=f"dB:{db:.1f} onset:{peak_onset:.1f} transients:{len(onset_frames)}"))

            # Scream / sustained high-energy voice — looser check, lower severity
            elif db > 78 and peak_onset > 6.0:
                onset_frames = librosa.onset.onset_detect(y=seg, sr=sr)
                if len(onset_frames) > 3:   # sustained, not a single bang
                    anomalies.append(AudioAnomaly(timestamp=ts, anomaly_type="elevated_audio",
                        confidence=0.55, detail=f"Sustained loud audio at {ts} (dB {db:.1f})"))

        return {"alerts": alerts, "timeline": timeline, "anomalies": anomalies}

    def _transcribe(self, path):
        timeline = []
        try:
            result = self.whisper.transcribe(path, language="en", word_timestamps=False)
            text   = result.get("text","").strip()
            for seg in result.get("segments",[]):
                # Only add segment to timeline if speech probability is high
                if seg.get("no_speech_prob", 1.0) < 0.5:
                    ts = _ts(seg["start"])
                    timeline.append(TimelineEvent(
                        timestamp=ts, source=AlertType.AUDIO,
                        event=f'Speech: "{seg["text"].strip()}"',
                        confidence=round(1 - seg.get("no_speech_prob",0), 2),
                    ))
            return text, timeline
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", []

    def _keyword_scan(self, transcript):
        alerts, timeline, anomalies = [], [], []
        words   = transcript.lower().split()
        matched = [kw for kw in settings.THREAT_KEYWORDS if kw in words]
        if matched:
            # Weight by count — 1 keyword = LOW, 2+ = MEDIUM, 4+ = HIGH
            n    = len(matched)
            sev  = SeverityLevel.HIGH if n >= 4 else SeverityLevel.MEDIUM if n >= 2 else SeverityLevel.LOW
            conf = min(0.92, 0.50 + n * 0.10)
            alerts.append(Alert(
                timestamp="00:00:00", severity=sev,
                alert_type=AlertType.AUDIO, category="keyword_alert",
                description=f"Threat keywords in transcript: {', '.join(matched)}",
                confidence=conf,
            ))
            timeline.append(TimelineEvent(timestamp="00:00:00", source=AlertType.AUDIO,
                event=f"Keywords matched: {', '.join(matched)}"))
            anomalies.append(AudioAnomaly(timestamp="00:00:00", anomaly_type="keyword",
                confidence=conf, detail=f"Matched {n} keyword(s): {', '.join(matched)}"))
        return {"alerts": alerts, "timeline": timeline, "anomalies": anomalies}

    def _speaker_id(self, y, sr, duration):
        timeline, anomalies = [], []
        try:
            import librosa
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            feat   = np.vstack([mfcc, librosa.feature.delta(mfcc)]).T
            n_sp   = min(4, max(2, int(duration // 30)))
            X      = StandardScaler().fit_transform(feat)
            labels = KMeans(n_clusters=n_sp, random_state=42, n_init=10).fit_predict(X)
            seen   = set()
            hop    = 512; start = 0.0; prev = labels[0]
            for i,lbl in enumerate(labels):
                if lbl != prev:
                    sid = f"SPK-{prev+1:02d}"
                    if sid not in seen:
                        seen.add(sid)
                        timeline.append(TimelineEvent(
                            timestamp=_ts(start), source=AlertType.AUDIO,
                            event=f"Speaker {sid} identified"))
                        if len(seen) > 1:
                            anomalies.append(AudioAnomaly(
                                timestamp=_ts(start), anomaly_type="new_speaker",
                                confidence=0.65, detail=f"New speaker: {sid}"))
                    start = i*hop/sr; prev = lbl
            return {"timeline": timeline, "anomalies": anomalies, "count": n_sp}
        except Exception as e:
            logger.warning(f"Speaker ID failed: {e}")
            return {"timeline": [], "anomalies": [], "count": 0}
