# NSG AI Surveillance Analysis System — Backend

AI/ML-powered analysis of MP4 video and MP3 audio for threat detection,
built with FastAPI. Runs fully on-premise, no internet required after setup.

---

## Project Structure

```
nsg_backend/
├── main.py                   ← FastAPI app entry point
├── requirements.txt
├── .env                      ← optional overrides
├── api/
│   ├── routes.py             ← REST endpoints
│   └── websocket.py          ← real-time progress stream
├── core/
│   ├── config.py             ← all settings / thresholds
│   ├── schemas.py            ← Pydantic data models
│   ├── video_analyzer.py     ← object detection, face, heatmap
│   ├── audio_analyzer.py     ← gunshot, transcription, keywords
│   ├── report_generator.py   ← PDF incident report
│   └── job_manager.py        ← background job tracking
├── models/
│   └── watchlist/            ← add face images here (JPG/PNG)
├── uploads/                  ← incoming files (auto-created)
├── outputs/                  ← results + PDFs (auto-created)
└── tests/
    └── test_smoke.py         ← smoke tests
```

---

## Quick Start

### 1. Install dependencies

```bash
cd nsg_backend
pip install -r requirements.txt
```

### 2. (Optional) Enable full AI models

Uncomment in `requirements.txt` and reinstall:

```bash
# Full object/weapon detection (requires ~6 MB model download on first run)
pip install ultralytics

# Speech-to-text transcription (~150 MB model)
pip install openai-whisper

# Face recognition against watchlist
pip install deepface
```

### 3. Run the server

```bash
python main.py
```

Server starts at: **http://localhost:8000**
API docs at:      **http://localhost:8000/docs**

### 4. Run smoke tests

```bash
python tests/test_smoke.py
```

---

## API Usage

### Upload files and start analysis

```bash
curl -X POST http://localhost:8000/api/v1/analyse \
  -F "video=@/path/to/footage.mp4" \
  -F "audio=@/path/to/audio.mp3"
```

Response:
```json
{ "job_id": "A3F7C2B1", "status": "processing", "message": "..." }
```

### Poll progress

```bash
curl http://localhost:8000/api/v1/jobs/A3F7C2B1
```

```json
{ "job_id": "A3F7C2B1", "progress": 72, "phase": "Transcribing speech…", "status": "processing" }
```

### Fetch full results

```bash
curl http://localhost:8000/api/v1/results/A3F7C2B1
```

### Download PDF report

```bash
curl -O http://localhost:8000/api/v1/report/A3F7C2B1
```

### Real-time progress via WebSocket

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/progress/A3F7C2B1");
ws.onmessage = (e) => console.log(JSON.parse(e.data));
// { progress: 45, phase: "Running gunshot detection…", status: "processing" }
```

---

## Adding Watchlist Faces

Drop reference face images into `models/watchlist/`:

```
models/watchlist/
├── WL-0047.jpg     ← filename becomes the watchlist ID
├── WL-0048.png
```

When a face is matched, the alert will reference the filename (without extension).

---

## Configuration

Edit `core/config.py` or create a `.env` file:

```env
FRAME_SAMPLE_RATE=1          # frames to extract per second
YOLO_CONFIDENCE=0.45         # detection confidence threshold
GUNSHOT_DB_THRESHOLD=85.0    # dB level to flag as gunshot
LOITER_SECONDS=120           # seconds before loitering alert
CROWD_THRESHOLD=6            # persons to trigger crowd alert
```

---

## Detection Capabilities

| Module | Method | Requires |
|---|---|---|
| Object & weapon detection | YOLOv8 | `ultralytics` |
| Person detection (fallback) | OpenCV contours | built-in |
| Face detection | Haar cascade | built-in |
| Face recognition | ArcFace/DeepFace | `deepface` |
| Loitering detection | Centroid tracking | built-in |
| Crowd detection | Person count threshold | built-in |
| Gunshot detection | RMS + onset strength | `librosa` |
| Speech transcription | OpenAI Whisper | `openai-whisper` |
| Keyword alerting | Lexical match | built-in |
| Speaker identification | MFCC + KMeans | `scikit-learn` |
| Heatmap generation | Grid accumulation | built-in |
| PDF report | ReportLab | `reportlab` |

---

## Production Hardening (next steps)

- Replace in-memory job store with **Redis** or **PostgreSQL**
- Add **JWT authentication** on all endpoints
- Use **NVIDIA TensorRT** for GPU-accelerated inference
- Replace KMeans speaker ID with **pyannote.audio**
- Add **encrypted storage** for face snapshots
- Deploy behind **Nginx** reverse proxy with TLS
