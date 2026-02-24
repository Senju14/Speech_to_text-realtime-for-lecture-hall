<p align="center">
  <h1 align="center">🎤 Real-time Vietnamese Speech Translation</h1>
  <p align="center">
    <em>Low-latency Vietnamese → English speech translation system for lecture halls</em>
  </p>
  <p align="center">
    <a href="#-features"><img src="https://img.shields.io/badge/ASR-WhisperX-blue?style=for-the-badge" alt="ASR"></a>
    <a href="#-features"><img src="https://img.shields.io/badge/NMT-NLLB%203.3B-green?style=for-the-badge" alt="NMT"></a>
    <a href="#-features"><img src="https://img.shields.io/badge/GPU-A100-orange?style=for-the-badge" alt="GPU"></a>
    <a href="#-features"><img src="https://img.shields.io/badge/Deploy-Modal-purple?style=for-the-badge" alt="Deploy"></a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.11-blue.svg?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
    <img src="https://img.shields.io/badge/status-Active-brightgreen.svg" alt="Status">
  </p>
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🗣️ **Real-time ASR** | WhisperX (Faster-Whisper large-v3) with streaming chunk-based processing |
| 🧠 **Neural VAD** | Pyannote segmentation model for accurate speech/silence detection |
| ✍️ **Post-processing** | BARTpho syllable-level correction with English-aware preservation |
| 🌐 **Translation** | NLLB-200 3.3B for high-quality Vietnamese → English translation |
| 🤖 **LLM Context** | Groq (LLaMA 3.1) for keyword expansion and lecture summarization |
| 🔁 **Local Agreement** | Multi-pass decoding consensus for stable partial transcriptions |
| 🛡️ **Hallucination Filter** | Pattern matching + WPS + confidence thresholding |
| ⚡ **Serverless GPU** | Modal cloud deployment with A100 GPU and auto-scaling |

---

## 📐 System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        FRONTEND  (Browser)                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  🎤 Microphone ──► AudioWorklet ──► Resample 16 kHz ──► Base64       │
│                                                             │        │
│  📊 UI Manager ◄──── WebSocket ◄────────────────────────────┘        │
│     ├── Transcript Display (VI + EN)                                 │
│     ├── Lecture Summary Panel                                        │
│     └── Session Recordings                                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                              │ WebSocket (JSON)
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   BACKEND  (Modal · A100 GPU)                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Audio Chunk                                                         │
│      │                                                               │
│      ▼                                                               │
│  ┌────────────┐    ┌──────────────────┐    ┌─────────────────────┐   │
│  │ Pyannote   │    │ Speech Segment   │    │ WhisperX ASR        │   │
│  │ Neural VAD │──►│ Buffer           │──►│ (Faster-Whisper)    │   │
│  └────────────┘    │ (overlap+merge)  │    │ + Local Agreement   │   │
│                    └──────────────────┘    └─────────┬───────────┘   │
│                                                      │               │
│                                    ┌─────────────────┤               │
│                                    ▼                 ▼               │
│                          ┌──────────────┐   ┌────────────────┐       │
│                          │ Hallucination│   │ Groq LLM       │       │
│                          │ Filter       │   │ (context prime) │       │
│                          └──────┬───────┘   └────────────────┘       │
│                                 │                                    │
│                                 ▼                                    │
│                          ┌──────────────┐                            │
│                          │ BARTpho      │                            │
│                          │ Correction   │                            │
│                          └──────┬───────┘                            │
│                                 │                                    │
│                                 ▼                                    │
│                          ┌──────────────┐                            │
│                          │ NLLB 3.3B    │                            │
│                          │ Translation  │──► WebSocket Response      │
│                          └──────────────┘                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Details |
|-------|-----------|---------|
| **ASR** | [WhisperX](https://github.com/m-bain/whisperX) / Faster-Whisper | `large-v3` · Vietnamese-optimized streaming |
| **VAD** | [Pyannote](https://github.com/pyannote/pyannote-audio) | Neural segmentation with 2s rolling context |
| **Post-processing** | [BARTpho](https://github.com/VinAIResearch/BARTpho) + LoRA | Syllable-level Vietnamese error correction |
| **Translation** | [NLLB-200 3.3B](https://huggingface.co/facebook/nllb-200-3.3B) | Vi → En neural machine translation |
| **LLM** | [Groq](https://groq.com/) (LLaMA 3.1 8B) | Keyword expansion & lecture summarization |
| **Backend** | [Modal](https://modal.com/) + FastAPI | Serverless GPU (A100) · WebSocket API |
| **Frontend** | Vanilla JS + CSS | AudioWorklet · Responsive UI |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Modal](https://modal.com/) account (free tier available)
- [Groq](https://console.groq.com/) API key *(optional, for LLM features)*

### Setup & Deploy

```bash
# 1. Clone repository
git clone https://github.com/Senju14/Speech_to_text-realtime-for-lecture-hall.git
cd Speech_to_text-realtime-for-lecture-hall

# 2. Install Modal CLI
pip install modal
modal token new

# 3. (Optional) Set Groq API key for LLM features
modal secret create groq-api-key GROQ_API_KEY=gsk_xxxxx

# 4. Deploy to Modal
modal deploy main.py
```

> 🌐 After deployment, access the URL printed in the terminal to open the web interface.

---

## 📁 Project Structure

```
.
├── main.py                          # Modal entry point & ASGI app
├── requirements.txt                 # Python dependencies
├── backend/
│   ├── config.py                    # All configuration parameters
│   ├── handler.py                   # WebSocket session management
│   ├── asr.py                       # WhisperX / Faster-Whisper ASR
│   ├── vad.py                       # Pyannote neural VAD
│   ├── speech_segment_buffer.py     # Audio segmentation with overlap
│   ├── local_agreement.py           # Multi-pass decoding consensus
│   ├── hallucination_filter.py      # ASR hallucination detection
│   ├── bartpho_corrector.py         # BARTpho post-processing (English-aware)
│   ├── translation.py               # NLLB translation service
│   ├── groq_service.py              # Groq LLM integration
│   ├── audio_buffer.py              # Audio buffer management
│   ├── audio_normalizer.py          # Audio normalization utilities
│   ├── torch_patch.py               # PyTorch compatibility patches
│   └── download_models.py           # Model download utilities
├── frontend/
│   ├── index.html                   # Main application page
│   ├── recordings.html              # Session recordings page
│   ├── style.css                    # UI styling
│   └── js/
│       ├── main.js                  # Application controller
│       ├── audio.js                 # Microphone capture & AudioWorklet
│       ├── socket.js                # WebSocket client
│       ├── ui.js                    # UI rendering & state management
│       ├── recorder.worklet.js      # Audio processing worklet
│       ├── recordings.js            # Recording playback
│       ├── export.js                # Transcript export
│       └── utils.js                 # Shared utilities
└── test/
    ├── README.md                    # Evaluation documentation
    ├── streaming_eval.py            # Benchmark evaluation script
    └── generate_tables.py           # Results table generator
```

---

## ⚙️ Configuration

All parameters are centralized in [`backend/config.py`](backend/config.py):

```python
# Modal Deployment
MODAL_GPU = "A100"                      # GPU type (A100 / H100)
MODAL_MEMORY = 24576                    # Container memory (MB)

# ASR
WHISPER_MODEL = "large-v3"              # Whisper model size
WHISPER_LANGUAGE = "vi"                 # Force Vietnamese

# VAD (Pyannote Neural)
VAD_BASE_THRESHOLD = 0.015              # Speech detection threshold

# Segmentation
MAX_SEGMENT_SEC = 5.0                   # Max segment duration
OVERLAP_SEC = 0.8                       # Segment overlap for context

# BARTpho Post-processing
ENABLE_BARTPHO = True                   # Toggle Vietnamese correction

# Translation
NLLB_MODEL = "facebook/nllb-200-3.3B"  # NLLB model
NLLB_SRC_LANG = "vie_Latn"             # Vietnamese
NLLB_TGT_LANG = "eng_Latn"             # English
```

---

## 📊 Evaluation Results

Streaming ASR benchmark on **100 samples per dataset**, evaluated on VLSP2020 (Vietnamese) and Earnings22 (English):

| Model | Dataset | GPU | WER ↓ | CER ↓ | TTFT | RTF |
|:------|:--------|:---:|------:|------:|-----:|----:|
| Whisper large-v3 | VLSP2020 | A100 | 26.94% | 22.19% | 4 ms | 0.070x |
| Whisper large-v3 | Earnings22 | A100 | 25.44% | 19.65% | 1 ms | 0.060x |
| **PhoWhisper** | **VLSP2020** | **A100** | **16.16%** | **14.82%** | 4 ms | 0.081x |
| PhoWhisper | Earnings22 | A100 | 29.59% | 21.80% | 2 ms | 0.088x |

**Key findings:**
- 🏆 PhoWhisper achieves **16.16% WER** on Vietnamese — **40% improvement** over Whisper
- Whisper outperforms on English (Earnings22) as expected
- RTF 0.06–0.09x → **11–17× faster than real-time**
- TTFT < 5 ms for all configurations

> 📖 See [`test/README.md`](test/README.md) for evaluation scripts and detailed methodology.

---

## 🔬 Key Methods

<details>
<summary><b>1. Streaming ASR Pipeline</b></summary>

- Chunk-based audio processing (~250ms chunks) with Pyannote neural VAD
- Speech segments buffered with configurable overlap for context continuity
- WhisperX (Faster-Whisper) for low-latency transcription on GPU
- Local Agreement algorithm ensures stable partial transcriptions across decode passes

</details>

<details>
<summary><b>2. Hallucination Detection</b></summary>

- Pattern matching for known artifacts (YouTube intros, repeated phrases)
- Words-per-second (WPS) validation — rejects unnaturally fast outputs
- Confidence score thresholding from Whisper decoder

</details>

<details>
<summary><b>3. BARTpho Post-processing (English-Aware)</b></summary>

- Three-layer English detection: diacritics/caps rules, common word list, consecutive ASCII heuristic
- Splits mixed Vietnamese-English text → corrects only Vietnamese portions
- Preserves English technical terms, abbreviations, and proper nouns
- Length safety check prevents over-correction

</details>

<details>
<summary><b>4. Neural Machine Translation</b></summary>

- NLLB-200 3.3B with safetensors for async Vi → En translation
- Concurrent translation pipeline — doesn't block ASR processing

</details>

<details>
<summary><b>5. LLM-powered Context Priming</b></summary>

- User provides lecture topic → Groq LLM expands into domain keywords
- Keywords injected as Whisper initial prompt for domain-adapted decoding
- Auto-summarization of lecture transcripts after configurable duration

</details>

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Partial transcript latency | ~0.5 – 1s |
| Final transcript + translation | ~2 – 3s |
| Cold start (model loading) | ~25s |
| GPU | NVIDIA A100 40GB |
| Supported languages | Vietnamese (ASR) → English (translation) |

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Built as part of a graduation thesis on real-time speech translation for lecture halls.</sub>
</p>
