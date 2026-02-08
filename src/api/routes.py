"""
HTTP API Routes
"""

import time
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from src.config import WHISPER_MODEL

logger = logging.getLogger(__name__)


class ExpandKeywordsRequest(BaseModel):
    topic: str
    language: str = "vi"


def create_api_routes(app: FastAPI, start_time: float, gpu: str = "A100", service=None):
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
    
    @app.post("/api/expand-keywords")
    async def expand_keywords(req: ExpandKeywordsRequest):
        """Expand a lecture topic into technical keywords via Groq LLM"""
        if not service or not service.groq or not service.groq.is_available:
            return JSONResponse(
                {"error": "LLM service not available"},
                status_code=503,
            )
        
        if not req.topic.strip():
            return JSONResponse(
                {"error": "Topic is required"},
                status_code=400,
            )
        
        try:
            keywords = await service.groq.expand_keywords(
                req.topic.strip(), language=req.language
            )
            return JSONResponse({"keywords": keywords})
        except Exception as e:
            logger.error(f"[API] expand-keywords error: {e}")
            return JSONResponse(
                {"error": "Keyword expansion failed"},
                status_code=500,
            )
