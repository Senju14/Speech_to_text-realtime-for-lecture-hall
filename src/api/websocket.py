"""
WebSocket Handler for Real-time Transcription

Supports both:
- Binary messages: Raw Int16 PCM audio bytes (optimized path)
- Text messages: JSON commands (start, stop, ping)
"""

import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
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
                # Receive either bytes or text
                message = await websocket.receive()
                
                if message["type"] == "websocket.receive":
                    if "bytes" in message and message["bytes"]:
                        # Binary audio data - fast path
                        await session.handle_binary_audio(message["bytes"])
                    elif "text" in message and message["text"]:
                        # JSON command (start/stop/ping)
                        await session.handle_incoming(message["text"])
                elif message["type"] == "websocket.disconnect":
                    break
                        
        except WebSocketDisconnect:
            print(f"[WS #{conn_id}] Disconnected")
        except RuntimeError as e:
            # Gracefully handle "Cannot call receive() once a disconnect has occurred"
            print(f"[WS #{conn_id}] Client gone: {e}")
        except Exception as e:
            print(f"[WS #{conn_id}] Error: {e}")
    
    async def send():
        """Send results back to client"""
        try:
            while True:
                msg = await session.out_queue.get()
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(msg)
        except RuntimeError:
            # Client disconnected during send
            pass
        except Exception:
            pass
    
    try:
        await asyncio.gather(recv(), send())
    finally:
        await session.cleanup()
        print(f"[WS #{conn_id}] Closed")
