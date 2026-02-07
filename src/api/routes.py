"""
HTTP API Routes
"""

import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.config import WHISPER_MODEL


def create_api_routes(app: FastAPI, start_time: float, gpu: str = "A100"):
    """Add HTTP routes to FastAPI app"""
    
    @app.get("/api/status")
    async def status():
        return JSONResponse({
            "status": "online",
            "model": f"WhisperX {WHISPER_MODEL}",
            "gpu": gpu,
            "uptime": int(time.time() - start_time),
        })
    
    @app.get("/api/health")
    async def health():
        return JSONResponse({"healthy": True})
