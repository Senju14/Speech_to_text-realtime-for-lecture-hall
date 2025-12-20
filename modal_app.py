import modal
import asyncio
import json
import struct
import time
from typing import Dict

import numpy as np

# ================= MODAL APP =================
app = modal.App("asr-whisper-streaming")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "torchaudio",
        "transformers",
        "accelerate",
        "fastapi",
        "uvicorn",
        "websockets",
        "numpy",
        "deep-translator",
    )
)

# ================= CONFIG =================
SAMPLE_RATE = 16000
MAX_BUFFER_SEC = 10
MAX_BUFFER_SAMPLES = SAMPLE_RATE * MAX_BUFFER_SEC
SILENCE_LIMIT = 0.6
PARTIAL_INTERVAL = 0.35

MODEL_ID = "openai/whisper-large-v3"

# ================= ASR SERVICE (RPC) =================
@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    scaledown_window=300,
)
class ASRService:

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        from deep_translator import GoogleTranslator

        print("🔄 Loading Whisper...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        self.translator = GoogleTranslator(source="vi", target="en")
        print("✅ Whisper ready")

    def _transcribe(self, audio: np.ndarray) -> str:
        import torch

        if len(audio) < int(0.5 * SAMPLE_RATE):
            return ""

        inputs = self.processor(
            audio=audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        feats = inputs.input_features.to(self.device, dtype=self.dtype)

        with torch.no_grad():
            ids = self.model.generate(
                feats,
                max_new_tokens=128,
                num_beams=1,
                do_sample=False,
                language="vi",
                task="transcribe",
            )

        return self.processor.batch_decode(
            ids, skip_special_tokens=True
        )[0].strip()

    @modal.method()
    def process_audio(self, audio_bytes: bytes) -> dict:
        pcm = (
            np.frombuffer(audio_bytes, np.int16)
            .astype(np.float32) / 32768.0
        )

        text = self._transcribe(pcm)
        trans = self.translator.translate(text) if text else ""

        return {"text": text, "trans": trans}


# ================= FASTAPI (ASGI) =================
@app.function(image=image, timeout=3600)
@modal.asgi_app()
def web_app():
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse

    fastapi_app = FastAPI()
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    asr = ASRService()
    viewers: Dict[str, WebSocket] = {}

    class State:
        def __init__(self, ws: WebSocket):
            self.ws = ws
            self.buffer = np.zeros(0, np.float32)
            self.speech = False
            self.last_voice = 0.0
            self.partial_task = None
            self.lock = asyncio.Lock()

    async def broadcast(obj: dict):
        dead = []
        payload = json.dumps(obj)
        for cid, ws in viewers.items():
            try:
                await ws.send_text(payload)
            except:
                dead.append(cid)
        for cid in dead:
            viewers.pop(cid, None)

    async def do_transcribe(state: State, final=False):
        async with state.lock:
            audio = state.buffer.copy()
            if final:
                state.buffer = np.zeros(0, np.float32)

        if audio.size == 0:
            return

        audio_bytes = (audio * 32768).astype(np.int16).tobytes()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: asr.process_audio.remote(audio_bytes)
        )

        if not result.get("text"):
            return

        payload = {
            "type": "fullSentence" if final else "realtime",
            "text": result["text"],
            "trans": result.get("trans", ""),
        }

        try:
            await state.ws.send_text(json.dumps(payload))
        except:
            pass

        await broadcast(payload)

    async def partial_loop(state: State):
        try:
            while state.speech:
                await do_transcribe(state, final=False)
                await asyncio.sleep(PARTIAL_INTERVAL)
        except asyncio.CancelledError:
            pass

    @fastapi_app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        cid = str(id(ws))
        role = ws.query_params.get("role", "broadcaster")

        if role == "viewer":
            viewers[cid] = ws
            try:
                while True:
                    await ws.receive_text()
            except WebSocketDisconnect:
                pass
            viewers.pop(cid, None)
            return

        state = State(ws)

        try:
            while True:
                data = await ws.receive_bytes()

                meta_len = struct.unpack_from("<I", data, 0)[0]
                pcm_bytes = data[4 + meta_len:]

                pcm = (
                    np.frombuffer(pcm_bytes, np.int16)
                    .astype(np.float32) / 32768.0
                )

                rms = float(np.sqrt(np.mean(pcm ** 2)))
                now = time.time()

                async with state.lock:
                    state.buffer = np.concatenate([state.buffer, pcm])
                    if len(state.buffer) > MAX_BUFFER_SAMPLES:
                        state.buffer = state.buffer[-MAX_BUFFER_SAMPLES:]

                if rms > 0.01:
                    if not state.speech:
                        state.speech = True
                        state.last_voice = now
                        await ws.send_text(json.dumps({"type": "vad_start"}))
                        await broadcast({"type": "vad_start"})
                        state.partial_task = asyncio.create_task(partial_loop(state))
                    else:
                        state.last_voice = now

                elif state.speech and (now - state.last_voice) > SILENCE_LIMIT:
                    state.speech = False
                    if state.partial_task:
                        state.partial_task.cancel()
                        state.partial_task = None

                    await ws.send_text(json.dumps({"type": "vad_stop"}))
                    await broadcast({"type": "vad_stop"})
                    await do_transcribe(state, final=True)

        except WebSocketDisconnect:
            pass
        finally:
            if state.partial_task:
                state.partial_task.cancel()

    @fastapi_app.get("/")
    async def root():
        return {"status": "ok", "ws": "/ws"}
    

    from fastapi.responses import HTMLResponse

    @fastapi_app.get("/viewer")
    async def viewer_page():
        return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
            <meta charset="utf-8"/>
            <title>ASR Viewer</title>
            </head>
            <body>
            <h2>ASR Live Viewer</h2>
            <pre id="log"></pre>
            <script>
            const ws = new WebSocket(
            (location.protocol === "https:" ? "wss://" : "ws://")
            + location.host + "/ws?role=viewer"
            );

            ws.onmessage = (e) => {
                const d = JSON.parse(e.data);
                document.getElementById("log").textContent +=
                JSON.stringify(d) + "\\n";
            };
            </script>
            </body>
            </html>
            """)

    return fastapi_app


# import modal
# import asyncio
# import json
# import struct
# import time
# from typing import Dict

# import numpy as np

# # ================= MODAL APP =================
# app = modal.App("asr-whisper-streaming")

# image = (
#     modal.Image.debian_slim(python_version="3.11")
#     .apt_install("ffmpeg")
#     .pip_install(
#         "torch",
#         "torchaudio",
#         "transformers",
#         "accelerate",
#         "fastapi",
#         "uvicorn",
#         "websockets",
#         "numpy",
#         "deep-translator",
#         "onnxruntime",  # For Silero VAD
#         "silero-vad",
#     )
# )

# # ================= CONFIG =================
# SAMPLE_RATE = 16000
# MAX_BUFFER_SEC = 10
# MAX_BUFFER_SAMPLES = SAMPLE_RATE * MAX_BUFFER_SEC
# SILENCE_LIMIT = 1.2  # Increased for better stability
# PARTIAL_INTERVAL = 0.35

# # VAD Configuration
# VAD_ON_THRESHOLD = 0.3      # Speech probability threshold to turn ON
# VAD_OFF_THRESHOLD = 0.25    # Speech probability threshold to turn OFF
# ON_DEBOUNCE_FRAMES = 2      # Frames to confirm speech start
# OFF_DEBOUNCE_FRAMES = 5     # Frames to confirm speech end
# MIN_SPEECH_DURATION = 0.3   # Minimum speech duration in seconds

# MODEL_ID = "openai/whisper-large-v3"

# # ================= VAD UTILITIES =================
# def load_silero_vad():
#     """Load Silero VAD model"""
#     from silero_vad import load_silero_vad
#     return load_silero_vad()

# def get_vad_probability(audio_bytes: bytes) -> float:
#     """Get speech probability for audio chunk using Silero VAD"""
#     import torch
    
#     # Convert bytes to numpy array
#     pcm = (
#         np.frombuffer(audio_bytes, np.int16)
#         .astype(np.float32) / 32768.0
#     )
    
#     if len(pcm) == 0:
#         return 0.0
    
#     # Convert to torch tensor
#     tensor = torch.from_numpy(pcm).float()
    
#     # Load VAD model (will be cached)
#     vad_model = load_silero_vad()
    
#     # Get VAD probability
#     with torch.no_grad():
#         probability = vad_model(tensor, SAMPLE_RATE)
    
#     return float(probability)

# # ================= ASR SERVICE (RPC) =================
# @app.cls(
#     image=image,
#     gpu="A10G",
#     timeout=600,
#     scaledown_window=300,
# )
# class ASRService:

#     @modal.enter()
#     def load(self):
#         import torch
#         from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
#         from deep_translator import GoogleTranslator

#         print("🔄 Loading Whisper...")
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.dtype = torch.float16 if self.device == "cuda" else torch.float32

#         self.processor = AutoProcessor.from_pretrained(MODEL_ID)
#         self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
#             MODEL_ID,
#             torch_dtype=self.dtype,
#             low_cpu_mem_usage=True,
#         ).to(self.device)
#         self.model.eval()

#         self.translator = GoogleTranslator(source="vi", target="en")
#         print("✅ Whisper ready")

#     def _transcribe(self, audio: np.ndarray) -> str:
#         import torch

#         if len(audio) < int(0.5 * SAMPLE_RATE):
#             return ""

#         inputs = self.processor(
#             audio=audio,
#             sampling_rate=SAMPLE_RATE,
#             return_tensors="pt"
#         )

#         feats = inputs.input_features.to(self.device, dtype=self.dtype)

#         with torch.no_grad():
#             ids = self.model.generate(
#                 feats,
#                 max_new_tokens=128,
#                 num_beams=1,
#                 do_sample=False,
#                 language="vi",
#                 task="transcribe",
#             )

#         return self.processor.batch_decode(
#             ids, skip_special_tokens=True
#         )[0].strip()

#     @modal.method()
#     def process_audio(self, audio_bytes: bytes) -> dict:
#         pcm = (
#             np.frombuffer(audio_bytes, np.int16)
#             .astype(np.float32) / 32768.0
#         )

#         text = self._transcribe(pcm)
#         trans = self.translator.translate(text) if text else ""

#         return {"text": text, "trans": trans}


# # ================= FASTAPI (ASGI) =================
# @app.function(image=image, timeout=3600)
# @modal.asgi_app()
# def web_app():
#     from fastapi import FastAPI, WebSocket, WebSocketDisconnect
#     from fastapi.middleware.cors import CORSMiddleware
#     from fastapi.responses import HTMLResponse

#     fastapi_app = FastAPI()
#     fastapi_app.add_middleware(
#         CORSMiddleware,
#         allow_origins=["*"],
#         allow_methods=["*"],
#         allow_headers=["*"],
#     )

#     # Initialize ASR service
#     asr = ASRService()
    
#     viewers: Dict[str, WebSocket] = {}

#     class VADController:
#         """Stateful VAD controller with debouncing"""
#         def __init__(self):
#             self.speech_active = False
#             self.on_counter = 0
#             self.off_counter = 0
#             self.last_speech_time = 0.0
#             self.speech_start_time = 0.0
            
#         def update(self, vad_prob: float, current_time: float) -> str:
#             """
#             Returns: "start", "continue", "stop", or None
#             """
#             # Update last speech time if probability is high
#             if vad_prob > VAD_OFF_THRESHOLD:
#                 self.last_speech_time = current_time
            
#             # Speech not active
#             if not self.speech_active:
#                 if vad_prob >= VAD_ON_THRESHOLD:
#                     self.on_counter += 1
#                 else:
#                     self.on_counter = 0
                
#                 # Debounce: need consecutive frames above threshold
#                 if self.on_counter >= ON_DEBOUNCE_FRAMES:
#                     self.speech_active = True
#                     self.speech_start_time = current_time
#                     self.on_counter = 0
#                     self.off_counter = 0
#                     return "start"
            
#             # Speech active
#             else:
#                 if vad_prob >= VAD_OFF_THRESHOLD:
#                     self.off_counter = 0
#                     return "continue"
#                 else:
#                     self.off_counter += 1
                    
#                     # Check if speech has ended
#                     if self.off_counter >= OFF_DEBOUNCE_FRAMES:
#                         # Ensure minimum speech duration
#                         speech_duration = current_time - self.speech_start_time
#                         if speech_duration >= MIN_SPEECH_DURATION:
#                             # Check silence duration
#                             silence_duration = current_time - self.last_speech_time
#                             if silence_duration > SILENCE_LIMIT:
#                                 self.speech_active = False
#                                 self.off_counter = 0
#                                 return "stop"
            
#             return None

#     class ConnectionState:
#         def __init__(self, ws: WebSocket):
#             self.ws = ws
#             self.buffer = np.zeros(0, np.float32)
#             self.speech = False
#             self.last_voice_time = 0.0
#             self.partial_task = None
#             self.lock = asyncio.Lock()
#             self.vad_controller = VADController()

#     async def broadcast(obj: dict):
#         dead = []
#         payload = json.dumps(obj)
#         for cid, ws in viewers.items():
#             try:
#                 await ws.send_text(payload)
#             except:
#                 dead.append(cid)
#         for cid in dead:
#             viewers.pop(cid, None)

#     async def do_transcribe(state: ConnectionState, final=False):
#         async with state.lock:
#             audio = state.buffer.copy()
#             if final:
#                 state.buffer = np.zeros(0, np.float32)

#         if audio.size == 0:
#             return

#         audio_bytes = (audio * 32768).astype(np.int16).tobytes()

#         loop = asyncio.get_event_loop()
#         result = await loop.run_in_executor(
#             None,
#             lambda: asr.process_audio.remote(audio_bytes)
#         )

#         if not result.get("text"):
#             return

#         payload = {
#             "type": "fullSentence" if final else "realtime",
#             "text": result["text"],
#             "trans": result.get("trans", ""),
#         }

#         try:
#             await state.ws.send_text(json.dumps(payload))
#         except:
#             pass

#         await broadcast(payload)

#     async def partial_loop(state: ConnectionState):
#         try:
#             while state.speech:
#                 await do_transcribe(state, final=False)
#                 await asyncio.sleep(PARTIAL_INTERVAL)
#         except asyncio.CancelledError:
#             pass

#     @fastapi_app.websocket("/ws")
#     async def ws_endpoint(ws: WebSocket):
#         await ws.accept()
#         cid = str(id(ws))
#         role = ws.query_params.get("role", "broadcaster")

#         if role == "viewer":
#             viewers[cid] = ws
#             try:
#                 while True:
#                     await ws.receive_text()
#             except WebSocketDisconnect:
#                 pass
#             viewers.pop(cid, None)
#             return

#         state = ConnectionState(ws)

#         try:
#             while True:
#                 data = await ws.receive_bytes()

#                 # Parse audio data
#                 meta_len = struct.unpack_from("<I", data, 0)[0]
#                 pcm_bytes = data[4 + meta_len:]

#                 if len(pcm_bytes) == 0:
#                     continue

#                 current_time = time.time()

#                 # Get VAD probability using the utility function
#                 vad_prob = await asyncio.get_event_loop().run_in_executor(
#                     None,
#                     lambda: get_vad_probability(pcm_bytes)
#                 )

#                 # Convert to numpy for buffer
#                 pcm = (
#                     np.frombuffer(pcm_bytes, np.int16)
#                     .astype(np.float32) / 32768.0
#                 )

#                 # Update audio buffer
#                 async with state.lock:
#                     state.buffer = np.concatenate([state.buffer, pcm])
#                     if len(state.buffer) > MAX_BUFFER_SAMPLES:
#                         state.buffer = state.buffer[-MAX_BUFFER_SAMPLES:]

#                 # Update VAD controller
#                 vad_event = state.vad_controller.update(vad_prob, current_time)

#                 # Handle VAD events
#                 if vad_event == "start":
#                     state.speech = True
#                     state.last_voice_time = current_time
#                     await ws.send_text(json.dumps({"type": "vad_start"}))
#                     await broadcast({"type": "vad_start"})
#                     state.partial_task = asyncio.create_task(partial_loop(state))
                    
#                 elif vad_event == "stop":
#                     state.speech = False
#                     if state.partial_task:
#                         state.partial_task.cancel()
#                         state.partial_task = None
                    
#                     await ws.send_text(json.dumps({"type": "vad_stop"}))
#                     await broadcast({"type": "vad_stop"})
#                     await do_transcribe(state, final=True)

#         except WebSocketDisconnect:
#             pass
#         finally:
#             if state.partial_task:
#                 state.partial_task.cancel()

#     @fastapi_app.get("/")
#     async def root():
#         return {"status": "ok", "ws": "/ws"}

#     @fastapi_app.get("/health")
#     async def health():
#         return {
#             "status": "healthy",
#             "services": {
#                 "asr": "ready",
#                 "vad": "ready"
#             }
#         }

#     @fastapi_app.get("/viewer")
#     async def viewer_page():
#         # Serve the actual overlay_client.html
#         html_content = """
# <!DOCTYPE html>
# <html lang="vi">
# <head>
#     <meta charset="utf-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1">
#     <title>ASR Live Captions</title>
#     <style>
#         :root {
#             --bg: #fff;
#             --text: #000;
#             --muted: #666;
#             --border: #e0e0e0;
#         }
#         * { margin: 0; padding: 0; box-sizing: border-box; }
#         body {
#             font-family: 'Segoe UI', sans-serif;
#             background: var(--bg);
#             color: var(--text);
#             height: 100vh;
#             display: flex;
#             flex-direction: column;
#         }
#         .container {
#             max-width: 900px;
#             margin: 0 auto;
#             padding: 20px;
#             flex: 1;
#             display: flex;
#             flex-direction: column;
#         }
#         header {
#             display: flex;
#             justify-content: space-between;
#             align-items: center;
#             border-bottom: 2px solid var(--text);
#             padding-bottom: 10px;
#             margin-bottom: 20px;
#         }
#         h1 { font-size: 24px; text-transform: uppercase; letter-spacing: 1px; }
#         .status { font-size: 14px; color: var(--muted); }
#         .display {
#             background: #f5f5f5;
#             padding: 30px;
#             border-radius: 8px;
#             border: 1px solid var(--border);
#             min-height: 150px;
#             text-align: center;
#             margin-bottom: 20px;
#         }
#         #vad { font-size: 14px; color: var(--muted); text-transform: uppercase; margin-bottom: 10px; }
#         #final { font-size: 32px; font-weight: 600; }
#         #final-trans { font-size: 24px; color: #444; font-style: italic; margin-top: 5px; }
#         #partial { font-size: 28px; color: var(--muted); margin-top: 15px; }
#         #partial-trans { font-size: 20px; color: #888; font-style: italic; }
#         .controls { display: flex; gap: 10px; margin-bottom: 20px; }
#         button {
#             background: var(--text);
#             color: var(--bg);
#             border: none;
#             padding: 10px 20px;
#             border-radius: 4px;
#             cursor: pointer;
#             font-weight: 600;
#         }
#         button:hover { opacity: 0.8; }
#         button.secondary { background: transparent; border: 1px solid var(--text); color: var(--text); }
#         .history { flex: 1; overflow-y: auto; border-top: 1px solid var(--border); padding-top: 20px; }
#         .history-item {
#             padding: 8px 0;
#             border-bottom: 1px solid var(--border);
#             display: flex;
#             font-size: 16px;
#         }
#         .time { font-family: monospace; margin-right: 15px; font-weight: 600; min-width: 80px; }
#         .trans { font-size: 14px; color: #888; font-style: italic; }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <header>
#             <h1>Live Captions</h1>
#             <div id="status" class="status">Disconnected</div>
#         </header>
#         <div class="display">
#             <div id="vad"></div>
#             <div id="final"></div>
#             <div id="final-trans"></div>
#             <div id="partial"></div>
#             <div id="partial-trans"></div>
#         </div>
#         <div class="controls">
#             <button id="clear">Clear</button>
#             <button id="fullscreen" class="secondary">Fullscreen</button>
#         </div>
#         <div class="history" id="history"></div>
#     </div>
#     <script>
#         const $ = id => document.getElementById(id);
#         const statusEl = $('status');
#         const vadEl = $('vad');
#         const finalEl = $('final');
#         const finalTransEl = $('final-trans');
#         const partialEl = $('partial');
#         const partialTransEl = $('partial-trans');
#         const historyEl = $('history');

#         $('clear').onclick = () => {
#             finalEl.textContent = '';
#             finalTransEl.textContent = '';
#             partialEl.textContent = '';
#             partialTransEl.textContent = '';
#         };

#         $('fullscreen').onclick = () => {
#             if (!document.fullscreenElement) document.documentElement.requestFullscreen();
#             else document.exitFullscreen();
#         };

#         const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws?role=viewer`;
#         let reconnectDelay = 1000;

#         function connect() {
#             statusEl.textContent = 'Connecting...';
#             const ws = new WebSocket(wsUrl);

#             ws.onopen = () => {
#                 statusEl.textContent = 'Connected';
#                 statusEl.style.color = 'green';
#                 reconnectDelay = 1000;
#             };

#             ws.onclose = () => {
#                 statusEl.textContent = 'Disconnected';
#                 statusEl.style.color = 'red';
#                 setTimeout(connect, reconnectDelay);
#                 reconnectDelay = Math.min(60000, reconnectDelay * 1.5);
#             };

#             ws.onerror = () => ws.close();

#             ws.onmessage = e => {
#                 const data = JSON.parse(e.data);
#                 if (data.type === 'realtime') {
#                     partialEl.textContent = data.text || '';
#                     partialTransEl.textContent = data.trans || '';
#                 } else if (data.type === 'fullSentence') {
#                     finalEl.textContent = data.text || '';
#                     finalTransEl.textContent = data.trans || '';
#                     partialEl.textContent = '';
#                     partialTransEl.textContent = '';
#                     if (data.text?.trim()) {
#                         const time = new Date().toLocaleTimeString('vi-VN', {hour:'2-digit', minute:'2-digit', second:'2-digit'});
#                         const item = document.createElement('div');
#                         item.className = 'history-item';
#                         item.innerHTML = `<span class="time">${time}</span><div><div>${data.text}</div><div class="trans">${data.trans || ''}</div></div>`;
#                         historyEl.prepend(item);
#                     }
#                 } else if (data.type === 'vad_start') {
#                     vadEl.textContent = 'Listening...';
#                 } else if (data.type === 'vad_stop') {
#                     vadEl.textContent = '';
#                 }
#             };
#         }

#         connect();
#     </script>
# </body>
# </html>
#         """
#         return HTMLResponse(html_content)

#     return fastapi_app


# @app.local_entrypoint()
# def main():
#     print("Deploy: modal deploy modal_app.py")
#     print("Dev: modal serve modal_app.py")
