import asyncio, json, logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from core import job_manager
from core.schemas import AnalysisStatus

ws_router = APIRouter(tags=["WebSocket"])
logger = logging.getLogger(__name__)

@ws_router.websocket("/progress/{job_id}")
async def progress_stream(ws: WebSocket, job_id: str):
    await ws.accept()
    try:
        while True:
            job = job_manager.get_job(job_id)
            if not job:
                await ws.send_text(json.dumps({"error":"Job not found"})); break
            await ws.send_text(json.dumps({
                "job_id": job_id, "progress": job["progress"],
                "phase": job["phase"], "status": job["status"],
            }))
            if job["status"] in (AnalysisStatus.COMPLETE, AnalysisStatus.FAILED): break
            await asyncio.sleep(0.8)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try: await ws.send_text(json.dumps({"error": str(e)}))
        except: pass
    finally:
        try: await ws.close()
        except: pass
