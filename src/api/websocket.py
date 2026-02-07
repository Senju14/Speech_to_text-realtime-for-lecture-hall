"""
WebSocket Handler for Real-time Transcription
"""

import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from src.session import ASRService


async def handle_websocket(websocket: WebSocket, service: ASRService, conn_id: int):
    """
    Handle WebSocket connection for real-time transcription
    
    Args:
        websocket: FastAPI WebSocket connection
        service: Shared ASR service instance
        conn_id: Connection ID for logging
    """
    await websocket.accept()
    print(f"[WS #{conn_id}] Connected")
    
    session = service.create_session()
    
    async def recv():
        """Receive and process messages from client"""
        try:
            while True:
                msg = await websocket.receive_text()
                await session.handle_incoming(msg)
        except WebSocketDisconnect:
            print(f"[WS #{conn_id}] Disconnected")
        except Exception as e:
            print(f"[WS #{conn_id}] Error: {e}")
    
    async def send():
        """Send results back to client"""
        try:
            while True:
                msg = await session.out_queue.get()
                await websocket.send_text(msg)
        except Exception:
            pass
    
    try:
        await asyncio.gather(recv(), send())
    finally:
        await session.cleanup()
        print(f"[WS #{conn_id}] Closed")
