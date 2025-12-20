# chrome_audio_bridge_fixed.py
import asyncio
import json
import struct
import websockets
import numpy as np
from queue import Queue, Empty
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json as json_lib

class ChromeAudioBridge:
    def __init__(self, ws_port=8765, target_ws_url=None):
        self.ws_port = ws_port
        self.target_ws_url = target_ws_url
        self.audio_queue = Queue()
        self.clients = set()
        self.is_forwarding = False
        self.forward_task = None
        self.target_ws = None
        self.loop = None
        
    def set_event_loop(self, loop):
        """Lưu event loop chính"""
        self.loop = loop
        
    async def handle_chrome_client(self, websocket):
        """Xử lý kết nối từ Chrome extension"""
        self.clients.add(websocket)
        print(f"📥 Chrome client connected. Total: {len(self.clients)}")
        
        # TỰ ĐỘNG BẬT FORWARDING KHI CÓ CLIENT ĐẦU TIÊN
        if not self.is_forwarding and self.target_ws_url:
            print("🚀 Tự động bật forwarding đến ASR server...")
            await self.start_forwarding_async()
        
        try:
            async for message in websocket:
                # Nhận audio data từ Chrome
                audio_data = np.frombuffer(message, dtype=np.int16)
                if len(audio_data) > 0:
                    self.audio_queue.put(audio_data)
                    
        except websockets.exceptions.ConnectionClosed:
            print("📤 Chrome client disconnected")
        finally:
            self.clients.remove(websocket)
            
    async def forward_to_asr(self):
        """Chuyển audio đến hệ thống ASR Modal"""
        if not self.target_ws_url:
            print("❌ Chưa cấu hình ASR server URL")
            return
            
        print(f"🔄 Đang kết nối đến ASR server: {self.target_ws_url}")
        
        try:
            async with websockets.connect(self.target_ws_url, max_size=None) as ws:
                print("✅ Đã kết nối đến ASR server")
                self.target_ws = ws
                
                # Metadata cho server
                meta = json.dumps({"sr": 16000, "source": "chrome_tab"}).encode()
                header = struct.pack("<I", len(meta)) + meta
                
                while self.is_forwarding:
                    try:
                        # Lấy audio từ queue
                        audio_chunk = self.audio_queue.get(timeout=0.1)
                        
                        # Gửi đến ASR server
                        await ws.send(header + audio_chunk.tobytes())
                            
                    except Empty:
                        await asyncio.sleep(0.01)
                    except Exception as e:
                        print(f"⚠️ Lỗi khi forward audio: {e}")
                        
        except Exception as e:
            print(f"❌ Lỗi kết nối đến ASR server: {e}")
        finally:
            self.target_ws = None
            
    async def start_forwarding_async(self):
        """Async version để gọi từ within event loop"""
        if not self.is_forwarding:
            self.is_forwarding = True
            self.forward_task = asyncio.create_task(self.forward_to_asr())
            print("▶️ Bắt đầu chuyển audio đến ASR")
            
    def start_forwarding(self):
        """Sync version để gọi từ HTTP handler"""
        if not self.is_forwarding and self.loop:
            self.is_forwarding = True
            # Tạo task trong event loop chính
            asyncio.run_coroutine_threadsafe(
                self.forward_to_asr(), 
                self.loop
            )
            print("▶️ Bắt đầu chuyển audio đến ASR")
            
    def stop_forwarding(self):
        """Dừng chuyển audio"""
        if self.is_forwarding:
            self.is_forwarding = False
            if self.forward_task:
                self.forward_task.cancel()
            print("⏹️ Dừng chuyển audio")
            
    async def websocket_server(self):
        """Chạy WebSocket server cho Chrome extension"""
        print(f"🌐 Chrome Audio Bridge đang chạy tại ws://localhost:{self.ws_port}")
        print("📌 Mở Chrome extension để bắt đầu capture audio từ tab")
        
        async with websockets.serve(
            self.handle_chrome_client, 
            "localhost", 
            self.ws_port
        ):
            await asyncio.Future()  # Chạy vô hạn

    class ControlHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/status':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                status = {
                    'clients': len(self.server.bridge.clients),
                    'forwarding': self.server.bridge.is_forwarding,
                    'queue_size': self.server.bridge.audio_queue.qsize()
                }
                self.wfile.write(json_lib.dumps(status).encode())
                
            elif self.path == '/start':
                self.server.bridge.start_forwarding()
                self.send_response(200)
                self.end_headers()
                
            elif self.path == '/stop':
                self.server.bridge.stop_forwarding()
                self.send_response(200)
                self.end_headers()
                
            else:
                self.send_response(404)
                self.end_headers()
                
        def log_message(self, format, *args):
            pass
            
    async def control_server(self):
        """HTTP server đơn giản để điều khiển từ xa"""
        server = HTTPServer(('localhost', 8766), self.ControlHandler)
        server.bridge = self
        
        print(f"🎛️ Control server tại http://localhost:8766")
        print("📊 Kiểm tra trạng thái: http://localhost:8766/status")
        print("🚀 Bật forwarding: http://localhost:8766/start")
        
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        while True:
            await asyncio.sleep(1)
        
    def run(self):
        """Chạy bridge"""
        asyncio.run(self.main())
        
    async def main(self):
        """Hàm chính"""
        # Lưu event loop chính
        self.loop = asyncio.get_running_loop()
        
        print("=" * 60)
        print("CHROME AUDIO BRIDGE - FIXED VERSION")
        print("=" * 60)
        print("📖 Extension đã cài đặt thành công!")
        print("👉 Mở YouTube tab và click extension để bắt đầu")
        print("=" * 60)
        
        ws_task = asyncio.create_task(self.websocket_server())
        control_task = asyncio.create_task(self.control_server())
        
        await asyncio.gather(ws_task, control_task)

if __name__ == "__main__":
    # DÙNG URL TỪ DEPLOYMENT VỪA RỒI
    TARGET_WS = "wss://ricky13170--asr-whisper-streaming-web-app.modal.run/ws"
    
    print(f"🔗 ASR Server URL: {TARGET_WS}")
    
    bridge = ChromeAudioBridge(
        ws_port=8765,
        target_ws_url=TARGET_WS
    )
    
    try:
        bridge.run()
    except KeyboardInterrupt:
        print("\n👋 Đang dừng Chrome Audio Bridge...")
