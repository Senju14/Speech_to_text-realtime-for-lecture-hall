# 🎤 Real-time Vietnamese-English Speech Translation

Hệ thống dịch tiếng nói Việt-Anh thời gian thực, sử dụng cho giảng đường.

## 🚀 Quick Start

```bash
# 1. Clone và cài đặt
git clone https://github.com/your-username/asr-thesis.git
cd asr-thesis

# 2. Setup Modal CLI
pip install modal
modal token new

# 3. Deploy
modal secret create groq-api-key GROQ_API_KEY=gsk_your_key_here
modal deploy main.py
```

Truy cập URL được in ra sau khi deploy.

## 📐 Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Browser)                       │
├─────────────────────────────────────────────────────────────────┤
│  🎤 Microphone ──► AudioWorklet ──► Resample 16kHz ──► Base64   │
│                                                          │       │
│  📊 UI Manager ◄── WebSocket ◄───────────────────────────┘       │
│     └── Transcript Display (Vi + En)                             │
└─────────────────────────────────────────────────────────────────┘
                              │ WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (Modal A100 GPU)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Audio ──► Silero VAD ──► Buffer ──► WhisperX ──► Text          │
│                                           │                       │
│                           ┌───────────────┘                       │
│                           ▼                                       │
│                 Hallucination Filter                              │
│                 (Pattern matching)                                │
│                           │                                       │
│                           ▼                                       │
│                 NLLB 3.3B Translator ──► English Text            │
│                           │                                       │
│                           ▼                                       │
│                 WebSocket Response                                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| ASR | WhisperX (large-v3) | Vietnamese speech recognition + word alignment |
| Translation | NLLB 3.3B | Vi→En neural machine translation |
| VAD | Silero VAD | Neural voice activity detection |
| Streaming | WebSocket | Real-time bidirectional |
| Backend | Modal + FastAPI | Serverless GPU compute |
| Frontend | Vanilla JS + CSS | Lightweight UI |

## 📁 Cấu trúc dự án

```
├── main.py                 # Modal entry point
├── src/
│   ├── config/
│   │   └── settings.py     # Runtime configuration
│   ├── asr/
│   │   └── whisperx_asr.py # WhisperX ASR wrapper
│   ├── vad/
│   │   └── silero_vad.py   # Silero VAD
│   ├── translation/
│   │   └── nllb_translator.py # NLLB translator
│   ├── session/
│   │   ├── handler.py      # WebSocket session handler
│   │   └── filters.py      # Hallucination filters
│   ├── api/
│   │   ├── routes.py       # HTTP endpoints
│   │   └── websocket.py    # WebSocket handler
│   └── utils/
│       ├── audio.py        # Audio processing
│       └── torch_patch.py  # PyTorch compatibility
└── frontend/
    ├── index.html          # Main UI
    ├── style.css           # Styling
    └── js/
        ├── main.js         # App controller
        ├── audio.js        # Audio capture
        ├── socket.js       # WebSocket client
        └── ui.js           # UI manager
```

## ⚙️ Cấu hình

Chỉnh sửa `src/config/settings.py`:

```python
WHISPER_MODEL = "large-v3"      # Model size
WHISPER_LANGUAGE = "vi"         # Force Vietnamese
VAD_THRESHOLD = 0.5             # Voice detection sensitivity
MAX_BUFFER_DURATION = 6.0       # Max audio buffer (seconds)
MIN_SILENCE_DURATION = 0.6      # Silence to trigger finalize
```

Modal config trong `main.py`:

```python
MODAL_GPU = "A100"              # GPU type
MODAL_MEMORY = 24576            # Memory (MB)
```

## 🔬 Phương pháp chính

1. **Streaming ASR Pipeline**
   - Chunk-based processing với Silero VAD
   - WhisperX cho batched inference + word alignment

2. **Hallucination Detection**
   - Pattern matching (YouTube artifacts, sign-offs)
   - Minimum length validation

3. **Cascade Translation**
   - NLLB 3.3B với float16
   - Async translation pipeline

4. **Real-time WebSocket Protocol**
   - Base64 audio streaming
   - JSON transcript responses with word timestamps

## 📊 Metrics

- **Latency**: ~0.5-1s (transcription + translation)
- **GPU**: A100 40GB
- **Model load**: ~30s cold start

## 📈 Evaluation Results

Streaming ASR benchmark on 100 samples per dataset:

| Model | Dataset | GPU | WER | CER | TTFT | RTF |
|-------|---------|-----|-----|-----|------|-----|
| Whisper | vlsp2020 | A100 | 26.94% | 22.19% | 4ms | 0.070x |
| Whisper | earnings22 | A100 | 25.44% | 19.65% | 1ms | 0.060x |
| PhoWhisper | vlsp2020 | A100 | **16.16%** | **14.82%** | 4ms | 0.081x |
| PhoWhisper | earnings22 | A100 | 29.59% | 21.80% | 2ms | 0.088x |

**Key findings:**
- PhoWhisper achieves **16.16% WER** on Vietnamese (VLSP2020) - 40% better than Whisper
- Whisper performs better on English (earnings22)
- RTF ~0.06-0.09x = **~11-17x faster than real-time**

See `test/README.md` for running evaluations.

## 📝 License

MIT License