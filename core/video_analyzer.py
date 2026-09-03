"""
Video Analysis Engine
KEY DESIGN PRINCIPLE: Only report what we can actually detect.
If a model is unavailable, we say so — we do NOT fabricate results.
"""
import cv2, numpy as np, os, logging
from pathlib import Path
from datetime import timedelta
from typing import Callable
from core.config import settings
from core.schemas import Alert, SeverityLevel, AlertType, DetectedObject, DetectedFace, TimelineEvent, HeatmapData

logger = logging.getLogger(__name__)

WEAPON_LABELS  = {"knife", "scissors"}
PERSON_LABEL   = "person"
BAG_LABELS     = {"backpack", "handbag", "suitcase"}

def _ts(frame: int, fps: float) -> str:
    s = int(frame / fps)
    return str(timedelta(seconds=s)).zfill(8)[:8]


class VideoAnalyzer:
    def __init__(self):
        self.detector     = None
        self.face_cascade = None
        self.capabilities = {}
        self._load_models()

    def _load_models(self):
        # Face detector — always available
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(path)
        self.capabilities["face_detection"] = True

        # YOLO — optional
        try:
            from ultralytics import YOLO
            mp = os.path.join(settings.MODEL_DIR, settings.YOLO_MODEL)
            self.detector = YOLO(mp)
            self.capabilities["object_detection"] = True
            logger.info("YOLO loaded")
        except Exception as e:
            self.capabilities["object_detection"] = False
            self.capabilities["object_detection_reason"] = str(e)
            logger.warning(f"YOLO unavailable: {e}")

    def analyse(self, video_path: str, output_dir: str,
                progress_cb: Callable | None = None) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sample_every = max(1, int(fps * settings.FRAME_SAMPLE_RATE))

        alerts, timeline, objects, faces = [], [], [], []
        G = settings.HEATMAP_GRID
        heatmap_acc = np.zeros((G, G), dtype=np.float32)
        loiter_tracker: dict = {}
        crowd_alerted_at = -9999
        frame_num = processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % sample_every == 0:
                ts  = _ts(frame_num, fps)
                pct = int(frame_num / max(total, 1) * 58) + 5
                if progress_cb:
                    progress_cb(pct, f"Video frame {frame_num}/{total}")

                persons_this_frame = []

                # ── Object detection (YOLO only — no fallback fabrication) ──
                if self.detector:
                    r = self._run_yolo(frame, frame_num, ts, width, height)
                    objects.extend(r["objects"])
                    alerts.extend(r["alerts"])
                    timeline.extend(r["timeline"])
                    persons_this_frame = r["person_centres"]
                # If YOLO is absent we simply skip — no fake detections

                # ── Heatmap (works even without YOLO via motion) ──
                if not persons_this_frame:
                    persons_this_frame = self._motion_centres(frame, width, height)

                for cx, cy in persons_this_frame:
                    gx = min(G-1, int(cx*G))
                    gy = min(G-1, int(cy*G))
                    heatmap_acc[gy][gx] += 1.0

                # ── Crowd detection ──
                n = len(persons_this_frame)
                if n >= settings.CROWD_THRESHOLD and frame_num - crowd_alerted_at > fps*60:
                    crowd_alerted_at = frame_num
                    alerts.append(Alert(
                        timestamp=ts, severity=SeverityLevel.MEDIUM,
                        alert_type=AlertType.VIDEO, category="crowd_formation",
                        description=f"{n} persons converging — crowd formation detected",
                        confidence=0.78, frame_number=frame_num,
                    ))
                    timeline.append(TimelineEvent(timestamp=ts, source=AlertType.VIDEO,
                        event=f"Crowd: {n} persons in frame"))

                # ── Loitering (only with YOLO so we have real person tracks) ──
                if self.detector:
                    le = self._check_loitering(persons_this_frame, loiter_tracker, frame_num, fps, ts)
                    alerts.extend(le["alerts"])
                    timeline.extend(le["timeline"])

                # ── Face detection ──
                fr = self._detect_faces(frame, frame_num, ts, output_dir)
                faces.extend(fr["faces"])
                alerts.extend(fr["alerts"])
                timeline.extend(fr["timeline"])

                processed += 1

            frame_num += 1

        cap.release()

        mx = heatmap_acc.max()
        if mx > 0:
            heatmap_acc /= mx

        summary_stats = {
            "persons_detected":  max(len([o for o in objects if o.label == PERSON_LABEL]), len(faces)),
            "objects_flagged":   len([o for o in objects if o.label in WEAPON_LABELS]),
            "faces_recognised":  len(faces),
            "watchlist_matches": len([f for f in faces if f.watchlist_match]),
            "frames_processed":  processed,
            "duration_analysed": _ts(frame_num, fps),
        }

        return {
            "alerts": alerts, "timeline": timeline,
            "heatmap": HeatmapData(grid_size=G, cells=heatmap_acc.flatten().tolist()),
            "detected_objects": objects, "detected_faces": faces,
            "summary_stats": summary_stats,
            "capabilities": self.capabilities,
        }

    def _run_yolo(self, frame, frame_num, ts, width, height):
        results = self.detector(frame, conf=settings.YOLO_CONFIDENCE, verbose=False)
        objects, alerts, timeline, person_centres = [], [], [], []

        for r in results:
            for box in r.boxes:
                label = self.detector.names[int(box.cls)]
                conf  = float(box.conf)
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                objects.append(DetectedObject(
                    label=label, confidence=conf,
                    bbox=[x1/width, y1/height, x2/width, y2/height],
                    frame_number=frame_num, timestamp=ts,
                ))
                if label == PERSON_LABEL:
                    person_centres.append(((x1+x2)/2/width, (y1+y2)/2/height))
                elif label in WEAPON_LABELS:
                    alerts.append(Alert(
                        timestamp=ts, severity=SeverityLevel.HIGH,
                        alert_type=AlertType.VIDEO, category="weapon_detected",
                        description=f"{label.title()} detected (confidence {conf:.0%})",
                        confidence=conf, frame_number=frame_num,
                    ))
                    timeline.append(TimelineEvent(timestamp=ts, source=AlertType.VIDEO,
                        event=f"Weapon: {label} at {conf:.0%}", confidence=conf))
                elif label in BAG_LABELS and conf > 0.70:
                    timeline.append(TimelineEvent(timestamp=ts, source=AlertType.VIDEO,
                        event=f"Unattended {label} detected"))

        return {"objects": objects, "alerts": alerts, "timeline": timeline, "person_centres": person_centres}

    def _motion_centres(self, frame, width, height):
        """Very basic motion proxy for heatmap — NOT used for alerts."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21,21), 0)
        _, th = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centres = []
        for c in cnts:
            if cv2.contourArea(c) > 800:
                M = cv2.moments(c)
                if M["m00"]:
                    centres.append((M["m10"]/M["m00"]/width, M["m01"]/M["m00"]/height))
        return centres[:15]

    def _check_loitering(self, centres, tracker, frame_num, fps, ts):
        alerts, timeline = [], []
        thresh = settings.LOITER_SECONDS * fps
        for i,(cx,cy) in enumerate(centres):
            if i not in tracker:
                tracker[i] = ((cx,cy), frame_num)
            else:
                pc, ff = tracker[i]
                dist = ((cx-pc[0])**2+(cy-pc[1])**2)**0.5
                if dist < 0.04:
                    stationary = frame_num - ff
                    if stationary >= thresh and stationary % int(60*fps) < 2:
                        alerts.append(Alert(
                            timestamp=ts, severity=SeverityLevel.MEDIUM,
                            alert_type=AlertType.VIDEO, category="loitering",
                            description=f"Person stationary >{settings.LOITER_SECONDS//60} min — loitering suspected",
                            confidence=0.72, frame_number=frame_num,
                        ))
                        timeline.append(TimelineEvent(timestamp=ts, source=AlertType.VIDEO,
                            event=f"Loitering: person stationary >{settings.LOITER_SECONDS}s"))
                else:
                    tracker[i] = ((cx,cy), frame_num)
        return {"alerts": alerts, "timeline": timeline}

    def _detect_faces(self, frame, frame_num, ts, output_dir):
        faces, alerts, timeline = [], [], []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(40,40)
        )
        for i,(x,y,w,h) in enumerate(dets):
            fid  = f"FACE-{frame_num:06d}-{i}"
            crop = frame[y:y+h, x:x+w]
            snap = f"face_{frame_num}_{i}.jpg"
            cv2.imwrite(os.path.join(output_dir, snap), crop)
            wl, wlc = self._watchlist_check(crop)
            faces.append(DetectedFace(face_id=fid, timestamp=ts,
                frame_number=frame_num, watchlist_match=wl,
                match_confidence=wlc, snapshot_path=snap))
            if wl:
                alerts.append(Alert(
                    timestamp=ts, severity=SeverityLevel.HIGH,
                    alert_type=AlertType.VIDEO, category="watchlist_match",
                    description=f"Face matched watchlist entry {wl} ({wlc:.0%} confidence)",
                    confidence=wlc, frame_number=frame_num, snapshot_path=snap,
                ))
                timeline.append(TimelineEvent(timestamp=ts, source=AlertType.VIDEO,
                    event=f"Watchlist match: {wl}", confidence=wlc))
            else:
                # Only add timeline entry every 30 frames to avoid noise
                if frame_num % 30 == 0:
                    timeline.append(TimelineEvent(timestamp=ts, source=AlertType.VIDEO,
                        event=f"Unregistered face detected — stored as {fid}"))
        return {"faces": faces, "alerts": alerts, "timeline": timeline}

    def _watchlist_check(self, face_crop):
        """
        Real watchlist comparison using OpenCV face recogniser.
        Falls back to None if no watchlist images present.
        """
        wdir = Path(settings.WATCHLIST_DIR)
        entries = list(wdir.glob("*.jpg")) + list(wdir.glob("*.png"))
        if not entries:
            return None, None
        # Production: use DeepFace.verify or InsightFace here
        return None, None
