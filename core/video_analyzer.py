"""
Video Analysis Engine
Handles: frame extraction · object/weapon detection · face detection ·
         activity analysis · loitering detection · crowd detection · heatmap
"""

import cv2
import numpy as np
import os
import logging
from pathlib import Path
from datetime import timedelta
from typing import Callable

from core.config import settings
from core.schemas import (
    Alert, SeverityLevel, AlertType, DetectedObject,
    DetectedFace, TimelineEvent, HeatmapData,
)

logger = logging.getLogger(__name__)

# ── Weapon / threat object labels from COCO that YOLO can detect ──────────────
WEAPON_LABELS = {"knife", "scissors", "gun", "pistol", "rifle", "sword"}
THREAT_LABELS = {"backpack", "handbag", "suitcase", "cell phone"}  # secondary
PERSON_LABEL  = "person"


def _frame_to_ts(frame_num: int, fps: float) -> str:
    """Convert frame number → HH:MM:SS string."""
    total_sec = int(frame_num / fps)
    return str(timedelta(seconds=total_sec)).zfill(8)[:8]


class VideoAnalyzer:
    def __init__(self):
        self.detector    = None   # YOLO model, loaded lazily
        self.face_cascade = None  # OpenCV Haar cascade for face detection
        self._load_models()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self):
        """Load detection models. Falls back gracefully if GPU/weights absent."""
        # Face detector — always available via OpenCV
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        logger.info("Face cascade loaded")

        # YOLO object detector — load if ultralytics available
        try:
            from ultralytics import YOLO
            model_path = os.path.join(settings.MODEL_DIR, settings.YOLO_MODEL)
            self.detector = YOLO(model_path)   # auto-downloads on first run
            logger.info(f"YOLO model loaded: {settings.YOLO_MODEL}")
        except ImportError:
            logger.warning("ultralytics not installed — object detection disabled")
        except Exception as e:
            logger.warning(f"YOLO load failed ({e}) — object detection disabled")

    # ── Public entry point ────────────────────────────────────────────────────

    def analyse(
        self,
        video_path: str,
        output_dir: str,
        progress_cb: Callable[[int, str], None] | None = None,
    ) -> dict:
        """
        Full video analysis pipeline.
        Returns a dict with keys: alerts, timeline, heatmap,
        detected_objects, detected_faces, summary_stats.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sample_every = max(1, int(fps * settings.FRAME_SAMPLE_RATE))

        logger.info(f"Video: {total_frames} frames @ {fps:.1f} fps  {width}x{height}")

        # Accumulators
        alerts:           list[Alert]          = []
        timeline:         list[TimelineEvent]  = []
        detected_objects: list[DetectedObject] = []
        detected_faces:   list[DetectedFace]   = []

        # Heatmap accumulator — N×N grid summing person presence
        G = settings.HEATMAP_GRID
        heatmap_acc = np.zeros((G, G), dtype=np.float32)

        # Loitering tracker: person_id → (last_bbox_centre, first_seen_frame)
        loiter_tracker: dict[int, tuple] = {}
        crowd_alerted_at: int = -999

        frame_num = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % sample_every == 0:
                ts = _frame_to_ts(frame_num, fps)
                pct = int(frame_num / max(total_frames, 1) * 60)  # video = 0–60%
                if progress_cb:
                    progress_cb(pct, f"Analysing frame {frame_num}/{total_frames}")

                # ── Object detection ──────────────────────────────────────
                persons_this_frame = []
                if self.detector:
                    obj_results = self._run_yolo(frame, frame_num, ts, width, height)
                    detected_objects.extend(obj_results["objects"])
                    alerts.extend(obj_results["alerts"])
                    timeline.extend(obj_results["timeline"])
                    persons_this_frame = obj_results["person_centres"]
                else:
                    # Fallback: simple background subtraction person count
                    persons_this_frame = self._bg_subtract_persons(frame, width, height)

                # ── Heatmap accumulation ──────────────────────────────────
                for cx, cy in persons_this_frame:
                    gx = min(G - 1, int(cx * G))
                    gy = min(G - 1, int(cy * G))
                    heatmap_acc[gy][gx] += 1.0

                # ── Crowd detection ───────────────────────────────────────
                if (len(persons_this_frame) >= settings.CROWD_THRESHOLD
                        and frame_num - crowd_alerted_at > fps * 30):
                    crowd_alerted_at = frame_num
                    alerts.append(Alert(
                        timestamp=ts,
                        severity=SeverityLevel.MEDIUM,
                        alert_type=AlertType.VIDEO,
                        category="crowd_formation",
                        description=f"Crowd of {len(persons_this_frame)} persons detected — anomalous gathering",
                        confidence=0.80,
                        frame_number=frame_num,
                    ))
                    timeline.append(TimelineEvent(
                        timestamp=ts, source=AlertType.VIDEO,
                        event=f"Crowd formation: {len(persons_this_frame)} persons converging",
                    ))

                # ── Loitering detection ───────────────────────────────────
                loiter_events = self._check_loitering(
                    persons_this_frame, loiter_tracker, frame_num, fps, ts
                )
                alerts.extend(loiter_events["alerts"])
                timeline.extend(loiter_events["timeline"])

                # ── Face detection ────────────────────────────────────────
                face_results = self._detect_faces(frame, frame_num, ts, output_dir)
                detected_faces.extend(face_results["faces"])
                alerts.extend(face_results["alerts"])
                timeline.extend(face_results["timeline"])

                processed += 1

            frame_num += 1

        cap.release()

        # ── Normalise heatmap ─────────────────────────────────────────────────
        max_val = heatmap_acc.max()
        if max_val > 0:
            heatmap_acc /= max_val
        heatmap_flat = heatmap_acc.flatten().tolist()

        heatmap = HeatmapData(grid_size=G, cells=heatmap_flat)

        # ── Summary stats ─────────────────────────────────────────────────────
        unique_persons = max(
            len([o for o in detected_objects if o.label == PERSON_LABEL]),
            len(detected_faces),
            1,
        )
        summary_stats = {
            "persons_detected":  unique_persons,
            "objects_flagged":   len([o for o in detected_objects if o.label in WEAPON_LABELS]),
            "faces_recognised":  len(detected_faces),
            "watchlist_matches": len([f for f in detected_faces if f.watchlist_match]),
            "frames_processed":  processed,
            "duration_analysed": _frame_to_ts(frame_num, fps),
        }

        return {
            "alerts":            alerts,
            "timeline":          timeline,
            "heatmap":           heatmap,
            "detected_objects":  detected_objects,
            "detected_faces":    detected_faces,
            "summary_stats":     summary_stats,
        }

    # ── YOLO inference ────────────────────────────────────────────────────────

    def _run_yolo(self, frame, frame_num, ts, width, height):
        results      = self.detector(frame, conf=settings.YOLO_CONFIDENCE, verbose=False)
        objects      = []
        alerts       = []
        timeline     = []
        person_centres = []

        for r in results:
            for box in r.boxes:
                label = self.detector.names[int(box.cls)]
                conf  = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Normalise bbox
                bbox = [x1/width, y1/height, x2/width, y2/height]

                objects.append(DetectedObject(
                    label=label, confidence=conf, bbox=bbox,
                    frame_number=frame_num, timestamp=ts,
                ))

                # Person centre for heatmap / crowd / loitering
                if label == PERSON_LABEL:
                    cx = ((x1 + x2) / 2) / width
                    cy = ((y1 + y2) / 2) / height
                    person_centres.append((cx, cy))

                # Weapon alert
                if label in WEAPON_LABELS:
                    severity = SeverityLevel.HIGH
                    alerts.append(Alert(
                        timestamp=ts, severity=severity,
                        alert_type=AlertType.VIDEO,
                        category="weapon_detected",
                        description=f"{label.title()} detected in frame (confidence {conf:.0%})",
                        confidence=conf, frame_number=frame_num,
                    ))
                    timeline.append(TimelineEvent(
                        timestamp=ts, source=AlertType.VIDEO,
                        event=f"Weapon detected: {label} ({conf:.0%})",
                        confidence=conf,
                    ))

                # Unattended bag — flag as medium
                elif label in {"backpack", "suitcase", "handbag"}:
                    alerts.append(Alert(
                        timestamp=ts, severity=SeverityLevel.MEDIUM,
                        alert_type=AlertType.VIDEO,
                        category="unattended_object",
                        description=f"Unattended {label} detected — verify ownership",
                        confidence=conf, frame_number=frame_num,
                    ))

        return {"objects": objects, "alerts": alerts, "timeline": timeline,
                "person_centres": person_centres}

    # ── Background subtraction fallback ───────────────────────────────────────

    def _bg_subtract_persons(self, frame, width, height):
        """
        Very lightweight fallback when YOLO is unavailable.
        Uses simple motion/edge cues to estimate person presence regions.
        Returns list of normalised (cx, cy) centres.
        """
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred  = cv2.GaussianBlur(gray, (21, 21), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centres = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:  # rough person-sized blob
                M  = cv2.moments(c)
                if M["m00"] != 0:
                    cx = (M["m10"] / M["m00"]) / width
                    cy = (M["m01"] / M["m00"]) / height
                    centres.append((cx, cy))
        return centres[:20]  # cap at 20

    # ── Loitering detection ───────────────────────────────────────────────────

    def _check_loitering(self, person_centres, tracker, frame_num, fps, ts):
        alerts   = []
        timeline = []
        threshold_frames = settings.LOITER_SECONDS * fps

        for i, (cx, cy) in enumerate(person_centres):
            pid = i  # simplified: use index as proxy ID
            if pid not in tracker:
                tracker[pid] = ((cx, cy), frame_num)
            else:
                prev_centre, first_frame = tracker[pid]
                dist = ((cx - prev_centre[0])**2 + (cy - prev_centre[1])**2) ** 0.5
                if dist < 0.05:   # person hasn't moved more than 5% of frame width
                    frames_stationary = frame_num - first_frame
                    if frames_stationary >= threshold_frames:
                        # Only alert once per 60-second window per person
                        if frames_stationary % (60 * fps) < (fps * settings.FRAME_SAMPLE_RATE + 1):
                            alerts.append(Alert(
                                timestamp=ts,
                                severity=SeverityLevel.MEDIUM,
                                alert_type=AlertType.VIDEO,
                                category="loitering",
                                description=f"Person stationary for >{settings.LOITER_SECONDS}s — possible loitering",
                                confidence=0.75,
                                frame_number=frame_num,
                            ))
                            timeline.append(TimelineEvent(
                                timestamp=ts, source=AlertType.VIDEO,
                                event=f"Loitering detected — person #{pid} stationary >{settings.LOITER_SECONDS}s",
                            ))
                else:
                    tracker[pid] = ((cx, cy), frame_num)   # reset on movement

        return {"alerts": alerts, "timeline": timeline}

    # ── Face detection ────────────────────────────────────────────────────────

    def _detect_faces(self, frame, frame_num, ts, output_dir):
        faces    = []
        alerts   = []
        timeline = []
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for i, (x, y, w, h) in enumerate(detections):
            face_id      = f"FACE-{frame_num:06d}-{i}"
            face_crop    = frame[y:y+h, x:x+w]
            snapshot_rel = f"face_{frame_num}_{i}.jpg"
            snapshot_abs = os.path.join(output_dir, snapshot_rel)
            cv2.imwrite(snapshot_abs, face_crop)

            # Watchlist comparison (stub — replace with ArcFace/DeepFace in prod)
            wl_match, wl_conf = self._compare_watchlist(face_crop)

            face = DetectedFace(
                face_id=face_id, timestamp=ts, frame_number=frame_num,
                watchlist_match=wl_match, match_confidence=wl_conf,
                snapshot_path=snapshot_rel,
            )
            faces.append(face)

            if wl_match:
                alerts.append(Alert(
                    timestamp=ts, severity=SeverityLevel.HIGH,
                    alert_type=AlertType.VIDEO,
                    category="watchlist_match",
                    description=f"Face matched watchlist entry {wl_match} (confidence {wl_conf:.0%})",
                    confidence=wl_conf, frame_number=frame_num,
                    snapshot_path=snapshot_rel,
                ))
                timeline.append(TimelineEvent(
                    timestamp=ts, source=AlertType.VIDEO,
                    event=f"Watchlist match: {wl_match} ({wl_conf:.0%})",
                    confidence=wl_conf,
                ))
            else:
                timeline.append(TimelineEvent(
                    timestamp=ts, source=AlertType.VIDEO,
                    event=f"Face detected (unregistered) — stored as {face_id}",
                ))

        return {"faces": faces, "alerts": alerts, "timeline": timeline}

    def _compare_watchlist(self, face_crop):
        """
        Stub watchlist comparator.
        Production: replace with ArcFace / InsightFace embedding comparison.
        Loads reference images from settings.WATCHLIST_DIR and computes cosine
        similarity against the incoming face embedding.
        """
        watchlist_dir = Path(settings.WATCHLIST_DIR)
        entries = list(watchlist_dir.glob("*.jpg")) + list(watchlist_dir.glob("*.png"))
        if not entries:
            return None, None

        # Placeholder: simulate no match (returns None)
        # In production:
        #   from deepface import DeepFace
        #   result = DeepFace.verify(face_crop, str(entry), model_name="ArcFace")
        #   if result["verified"]: return entry.stem, 1 - result["distance"]
        return None, None
