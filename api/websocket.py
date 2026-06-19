"""
WebSocket endpoint for real-time job progress streaming.
Connect to: ws://localhost:8000/ws/progress/{job_id}
Receives JSON messages: { "progress": 42, "phase": "...", "status": "..." }
"""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from core import job_manager
from core.schemas import AnalysisStatus

ws_router = APIRouter(tags=["WebSocket"])
logger = logging.getLogger(__name__)


@ws_router.websocket("/progress/{job_id}")
async def progress_stream(websocket: WebSocket, job_id: str):
    await websocket.accept()
    logger.info(f"WS connected: job {job_id}")

    try:
        while True:
            job = job_manager.get_job(job_id)
            if not job:
                await websocket.send_text(json.dumps({"error": "Job not found"}))
                break

            payload = {
                "job_id":   job_id,
                "progress": job["progress"],
                "phase":    job["phase"],
                "status":   job["status"],
            }
            await websocket.send_text(json.dumps(payload))

            if job["status"] in (AnalysisStatus.COMPLETE, AnalysisStatus.FAILED):
                break

            await asyncio.sleep(0.8)   # push update every 800 ms

    except WebSocketDisconnect:
        logger.info(f"WS disconnected: job {job_id}")
    except Exception as e:
        logger.error(f"WS error for job {job_id}: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
