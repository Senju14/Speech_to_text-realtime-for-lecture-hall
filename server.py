import asyncio
import json
import struct
import time
from typing import Dict, Any

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# --- existing imports ---
from vad_silero import vad_prob_for_buffer
from utils.audio_utils import highpass_filter, normalize_audio
from caption import update_overlay_text
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from caption import update_overlay_text 

import noisereduce as nr


#  Denoiser helpers
def process_realtime_chunk(x):
    """Realtime pipeline: highpass + normalize"""
    try:
        x = highpass_filter(x)
    except:
        pass
    try:
        x = normalize_audio(x)
    except:
        pass
    return x

def process_final_sentence(x, sr=16000):
    """Final sentence denoise: apply realtime pipeline + offline noise reduction"""
    x = process_realtime_chunk(x)
    try:
        x = nr.reduce_noise(y=x, sr=sr, prop_decrease=0.5)
    except:
        pass
    return x

# ----------------- CONFIG -----------------
MODEL_ID = "vinai/PhoWhisper-tiny"
SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_BUFFER_SEC = 10 #giữ 10s audio buffer
MAX_BUFFER = MAX_BUFFER_SEC * SR

print(f"🔄 Loading PhoWhisper on {DEVICE}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
print("✅ PhoWhisper ready")

# ----------------- FASTAPI -----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- CONNECTION REGISTRIES -----------------
broadcasters: Dict[str, "ConnectionState"] = {}
viewers: Dict[str, WebSocket] = {}

# ----------------- CONSTANTS -----------------
CHUNK_SAMPLES = 512
PARTIAL_INTERVAL = 0.25
VAD_ON_THRESH = 0.4
MAX_SILENCE_AFTER_SPEECH = 0.6

# ----------------- CONNECTION STATE -----------------
class ConnectionState:
    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.buffer = np.zeros(0, dtype=np.float32)
        self.partial_history = []
        self.agreed_text = ""
        self.speech_active = False
        self.last_speech_ts = 0.0
        self.lock = asyncio.Lock()
        self.partial_task = None
        self.current_segment_start = None
        self.segments = []
        self.futures = []

    async def send_json(self, obj: Dict[str, Any]):
        try:
            await self.ws.send_text(json.dumps(obj))
        except Exception as e:
            print("⚠️ WebSocket send failed:", e)

# ----------------- BROADCAST -----------------
async def broadcast_to_viewers(obj: Dict[str, Any]):
    if not viewers:
        return
    dead = []
    payload = json.dumps(obj)
    for cid, ws in list(viewers.items()):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(cid)
    for cid in dead:
        viewers.pop(cid, None)

# ----------------- LOCAL AGREEMENT -----------------
def apply_local_agreement(state: ConnectionState, new_text: str):
    state.partial_history.append(new_text)
    if len(state.partial_history) > 6:
        state.partial_history.pop(0)
    votes = {}
    for t in state.partial_history:
        votes[t] = votes.get(t, 0) + 1
    best, count = max(votes.items(), key=lambda x: x[1])
    if count >= 3 and len(best) > len(state.agreed_text):
        state.agreed_text = best
        return best
    return None

# ----------------- TRANSCRIPTION -----------------
async def do_transcribe_and_send(state: ConnectionState, final=False):
    async with state.lock:
        audio = state.buffer.copy()
    if audio.size == 0:
        return

    # Realtime partial: only highpass + normalize
    if not final:
        audio_proc = process_realtime_chunk(audio)
    else:
        audio_proc = process_final_sentence(audio, sr=SR)

    # avoid blocking asyncio
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, lambda: transcribe_audio(audio_proc))

    if not text:
        return

    now = time.time()

    if final:
        final_text = state.agreed_text or text
        seg_start = state.current_segment_start or now
        seg_end = now
        state.segments.append((seg_start, seg_end, final_text))
        await state.send_json({"type": "fullSentence", "text": final_text})
        await broadcast_to_viewers({"type": "fullSentence", "text": final_text})
        update_overlay_text("fullSentence", final_text)
        async with state.lock:
            state.buffer = np.zeros(0, dtype=np.float32)
            state.partial_history = []
            state.agreed_text = ""
            state.current_segment_start = None
        return

    # Local agreement for partial
    committed = apply_local_agreement(state, text)
    text_out = committed or text
    await state.send_json({"type": "realtime", "text": text_out})
    await broadcast_to_viewers({"type": "realtime", "text": text_out})
    update_overlay_text("realtime", text_out)

def transcribe_audio(audio_np):
    if audio_np is None or len(audio_np) < int(0.5 * SR):
        return ""
    try:
        inputs = processor(audio=audio_np, sampling_rate=SR, return_tensors="pt")
        input_features = inputs.input_features.to(DEVICE)

        attention_mask = inputs.attention_mask.to(DEVICE) if hasattr(inputs, 'attention_mask') else None

        with torch.no_grad():
            ids = model.generate(input_features, max_new_tokens=128)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text.strip()
    except Exception as e:
        print("⚠️ Transcription failed:", e)
        return ""

async def periodic_partial_sender(state: ConnectionState):
    try:
        while state.speech_active:
            await do_transcribe_and_send(state, final=False)
            await asyncio.sleep(PARTIAL_INTERVAL)
    except asyncio.CancelledError:
        return

# ----------------- WEBSOCKET -----------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    cid = str(id(ws))
    role = ws.query_params.get("role", "broadcaster")

    if role == "viewer":
        viewers[cid] = ws
        print(f"👁️ Viewer connected ID={cid} total_viewers={len(viewers)}")
        try:
            while True:
                try:
                    await ws.receive_text()
                except WebSocketDisconnect:
                    break
                except Exception:
                    await asyncio.sleep(0.1)
        finally:
            viewers.pop(cid, None)
            print(f"❌ Viewer disconnected ID={cid} total_viewers={len(viewers)}")
        return

    state = ConnectionState(ws)
    broadcasters[cid] = state
    print(f"✅ Broadcaster connected ID={cid}")

    try:
        while True:
            data = await ws.receive_bytes()
            meta_len = struct.unpack_from("<I", data, 0)[0]
            audio_bytes = data[4 + meta_len:]
            if len(audio_bytes) == 0:
                continue
            pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            #async with state.lock:
                #state.buffer = np.concatenate([state.buffer, pcm])
            async with state.lock:
                state.buffer = np.concatenate([state.buffer, pcm])
                if len(state.buffer) > MAX_BUFFER:
                    state.buffer = state.buffer[-MAX_BUFFER:]
                    print(f"🔧 Buffer trimmed: {len(state.buffer)} samples")  # D\ebug4
                # Debug: in buffer size mỗi 100 chunks
                if len(state.buffer) % (100 * CHUNK_SAMPLES) < CHUNK_SAMPLES:
                    buffer_seconds = len(state.buffer) / SR
                    print(f"📊 Buffer size: {buffer_seconds:.1f}s ({len(state.buffer)} samples)")

            prob = vad_prob_for_buffer(pcm)
            now = time.time()

            if prob >= VAD_ON_THRESH:
                if not state.speech_active:
                    state.speech_active = True
                    state.last_speech_ts = now
                    state.current_segment_start = now
                    await state.send_json({"type": "vad_start"})
                    await broadcast_to_viewers({"type": "vad_start"})
                    update_overlay_text("vad_start", "")
                    print("🎤 Speech start")
                    state.partial_task = asyncio.create_task(periodic_partial_sender(state))
                else:
                    state.last_speech_ts = now
            elif state.speech_active and (now - state.last_speech_ts) > MAX_SILENCE_AFTER_SPEECH:
                state.speech_active = False
                if state.partial_task:
                    state.partial_task.cancel()
                    state.partial_task = None
                print("🔇 Speech end — finalizing")
                await state.send_json({"type": "vad_stop"})
                await broadcast_to_viewers({"type": "vad_stop"})
                update_overlay_text("vad_stop", "")
                await do_transcribe_and_send(state, final=True)

    except WebSocketDisconnect:
        print(f"⚠️ Broadcaster disconnected ID={cid}")
    finally:
        if state.partial_task:
            state.partial_task.cancel()
        broadcasters.pop(cid, None)

# ----------------- VIEWER HTML -----------------
VIEWER_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Realtime Captions Viewer</title>
  <style>
    :root{
      --bg:#0f1722;
      --panel: rgba(255,255,255,0.04);
      --accent: #7cff7c;
      --muted: #aab3bd;
      --partial: #ffffff;
      --final: #7cff7c;
    }
    html,body{height:100%;margin:0;background:var(--bg);font-family:Inter,system-ui,Arial; color:var(--partial);}
    .container{max-width:1000px;margin:20px auto;padding:18px;}
    h1{margin:0 0 12px;font-size:20px;color:var(--final)}
    .caption-box{background:var(--panel);padding:16px;border-radius:12px;box-shadow:0 6px 18px rgba(0,0,0,0.5);}
    #vad{color: #ffd166; font-weight:600; margin-bottom:8px;}
    #partial{color:var(--partial); font-size:22px; line-height:1.3; min-height:48px; white-space:pre-wrap; word-break:break-word;}
    #final{color:var(--final); font-size:20px; margin-top:8px; opacity:0.95; white-space:pre-wrap; word-break:break-word;}
    .controls{margin-top:12px; display:flex; gap:8px; align-items:center;}
    .btn{background:#111827;color:white;padding:8px 12px;border-radius:8px;border:1px solid rgba(255,255,255,0.04);cursor:pointer;}
    .status{margin-left:auto;color:var(--muted);font-size:13px}
    .history{margin-top:14px; font-size:14px; color:var(--muted); max-height:220px; overflow:auto;}
    .history div{padding:6px 8px;border-bottom:1px dashed rgba(255,255,255,0.02);}
    @media (max-width:600px){
      .container{padding:12px}
      #partial{font-size:18px}
      #final{font-size:18px}
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Realtime Captions Viewer</h1>
    <div class="caption-box" id="box">
      <div id="vad"></div>
      <div id="partial"></div>
      <div id="final"></div>
      <div class="controls">
        <button id="clearBtn" class="btn">Clear final</button>
        <button id="fullscreenBtn" class="btn">Fullscreen</button>
        <div class="status" id="stat">Not connected</div>
      </div>
      <div class="history" id="history"></div>
    </div>
  </div>

<script>
(function(){
  const stat = document.getElementById('stat');
  const partialEl = document.getElementById('partial');
  const finalEl = document.getElementById('final');
  const vadEl = document.getElementById('vad');
  const historyEl = document.getElementById('history');
  const clearBtn = document.getElementById('clearBtn');
  const fullscreenBtn = document.getElementById('fullscreenBtn');

  clearBtn.onclick = () => { finalEl.innerText = ''; };
  fullscreenBtn.onclick = () => {
    const el = document.documentElement;
    if (el.requestFullscreen) el.requestFullscreen();
  };

  // build ws URL that matches server host + role=viewer
  const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl = `${wsProto}://${location.host}/ws?role=viewer`;
  let ws;
  let reconnectDelay = 1000;

  function connect(){
    stat.innerText = 'Connecting...';
    ws = new WebSocket(wsUrl);
    ws.onopen = () => {
      stat.innerText = 'Connected';
      reconnectDelay = 1000;
      console.log('[viewer] ws open', wsUrl);
    };
    ws.onclose = (ev) => {
      stat.innerText = 'Disconnected — retrying...';
      setTimeout(connect, reconnectDelay);
      reconnectDelay = Math.min(60000, reconnectDelay * 1.5);
    };
    ws.onerror = (e) => {
      ws.close();
    };
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        const t = data.type;
        if (t === 'realtime'){
          partialEl.innerText = data.text || '';
        } else if (t === 'fullSentence'){
          const txt = data.text || '';
          finalEl.innerText = txt;
          partialEl.innerText = '';
          const row = document.createElement('div');
          row.innerText = (new Date()).toLocaleTimeString() + ' — ' + txt;
          historyEl.prepend(row);
        } else if (t === 'vad_start'){
          vadEl.innerText = '🎤 Listening...';
        } else if (t === 'vad_stop'){
          vadEl.innerText = '';
        }
      } catch(e){
        console.error('bad msg', e, ev.data);
      }
    };
  }

  connect();
})();
</script>
</body>
</html>
"""

@app.get("/")
async def serve_viewer():
    return HTMLResponse(VIEWER_HTML)

# ----------------- RUN SERVER -----------------
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8012, log_level="info")
