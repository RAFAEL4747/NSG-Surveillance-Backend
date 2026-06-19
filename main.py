"""
NSG AI Surveillance Analysis System
FastAPI Backend — Entry Point
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import router
from api.websocket import ws_router
from core.config import settings

app = FastAPI(
    title="NSG Surveillance Analysis API",
    description="AI/ML-powered video and audio analysis for NSG surveillance systems",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.include_router(ws_router, prefix="/ws")

app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")

@app.get("/health")
async def health():
    return {"status": "operational", "system": "NSG Surveillance Analysis", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
