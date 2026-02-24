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

# 3. (Optional) Configure environment variables
cp env.example .env
# Edit .env and add your Groq API key (see https://console.groq.com/keys)
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
├── env.example                      # Environment variables template
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

All benchmarks run on **NVIDIA A100 40GB** with **N = 600 samples** from [Google FLEURS](https://huggingface.co/datasets/google/fleurs) test split.

### Table 1 — Streaming ASR Trade-off (Update Frequency × Compute)

Sweeping 5 chunk sizes across Vi → En and En → Vi translation directions:

| Translation | Min Chunk | WER (%) ↓ | BLEU ↑ | Latency (s) | RTF | N |
|:------------|:----------|----------:|-------:|:------------|:----|----:|
| En → Vi | 0.5 s | 4.91 | 0 | 0.213 ± 0.039 | 0.435 ± 0.077 | 600 |
| En → Vi | 1.0 s | 4.91 | 0 | 0.219 ± 0.041 | 0.230 ± 0.041 | 600 |
| En → Vi | 2.0 s | 4.91 | 0 | 0.226 ± 0.036 | 0.125 ± 0.019 | 600 |
| En → Vi | 3.0 s | 4.91 | 0 | 0.235 ± 0.040 | 0.088 ± 0.016 | 600 |
| En → Vi | 5.0 s | 4.91 | 0 | 0.259 ± 0.044 | 0.069 ± 0.012 | 600 |
| Vi → En | 0.5 s | 8.13 | 0.2 | 0.325 ± 0.064 | 0.665 ± 0.126 | 600 |
| Vi → En | 1.0 s | 8.13 | 0.2 | 0.331 ± 0.058 | 0.348 ± 0.055 | 600 |
| Vi → En | 2.0 s | 8.13 | 0.2 | 0.339 ± 0.056 | 0.187 ± 0.028 | 600 |
| Vi → En | 3.0 s | 8.13 | 0.2 | 0.351 ± 0.064 | 0.133 ± 0.023 | 600 |
| Vi → En | 5.0 s | 8.13 | 0.2 | 0.388 ± 0.062 | 0.098 ± 0.013 | 600 |

> WER is constant per direction (full-audio transcription); latency & RTF vary with chunk size.

---

### Table 2 — Computationally Aware MT Configuration

Comparing 4 MT configurations with fixed ASR (Whisper large-v3, full sentence, VAD on):

| Mode | Configuration | avg. BLEU ↑ | Latency (s) | avg. RTF | Throughput (tok/s) | Peak VRAM (GB) |
|:-----|:--------------|------------:|:------------|:---------|:-------------------|---------------:|
| Unaware | NLLB fp16, beam = 5 | 0.2 | 0.776 ± 0.261 | 0.061 | 41 ± 3 | 11.1 |
| Unaware | NLLB fp16, beam = 1 | 0.2 | 0.611 ± 0.203 | 0.048 | 52 ± 2 | 11.1 |
| Aware | Dynamic Beam (queue > 5) | 0.2 | 0.617 ± 0.209 | 0.048 | 51 ± 4 | 11.1 |
| **Aware + Quantized** | **Dynamic Beam + int8** | **0.2** | **0.603 ± 0.205** | **0.047** | **53 ± 4** | **11.1** |

> Dynamic Beam reduces beam 5 → 1 when translation queue > 5, saving compute without BLEU loss.

---

### Table 3 — End-to-End Latency Breakdown (1 chunk, 3s audio)

Profiling each pipeline stage on a single 3-second Vietnamese audio chunk:

| Pipeline Stage | Platform | Time (ms) | Percentage (%) | Bottleneck |
|:-------------------------------|:---------|:---------:|:--------------:|:-------------------|
| Client → Modal (upload) | Network | 80 | 13.2 | I/O |
| Pyannote VAD | GPU | 4 ± 0.3 | 0.7 | — |
| Whisper large-v3 | GPU | 167 ± 16.3 | 27.6 | GPU Compute |
| BARTpho syllable correction | GPU | 173 ± 2.5 | 28.6 | GPU Compute |
| NLLB-3.3B (beam = 3) | GPU | 148 ± 137.3 | 24.5 | GPU Compute |
| Modal → Client (download) | Network | 32 | 5.3 | I/O |
| **Total (End-to-End)** | **—** | **604** | **100.0** | **RTF ≈ 0.20** |

> GPU compute (Whisper + BARTpho + NLLB) accounts for **80.7%** of total latency. VAD overhead is negligible (0.7%).

---

### Table 4 — VAD Ablation Study (Incremental Pipeline)

Incremental component evaluation on silence-padded FLEURS Vietnamese (chunk = 2.0s):

| Pipeline | WER (%) ↓ | Halluc. Filtered | Silence→ASR | ASR Calls | GPU Time (s) | GPU/sample (s) | GPU Saved (%) | N |
|:--------------------------|----------:|-----------------:|------------:|----------:|-------------:|-----------------:|----------------:|----:|
| Whisper only | 48.97 | 0 | 2,719 | 2,106 | 534.03 | 0.890 | 0 | 600 |
| + Pyannote VAD | **33.70** | 0 | 1,260 | 1,653 | 449.09 | 0.749 | **15.9** | 600 |
| + VAD + Halluc. Filter | 57.41 | 776 | 1,260 | 1,653 | 451.16 | 0.752 | 15.5 | 600 |
| Full Pipeline (+ BARTpho) | 58.57 | 776 | 1,260 | 1,653 | 804.02 | 1.340 | -50.6 | 600 |

**Key findings:**
- 🏆 Pyannote VAD reduces WER from **48.97% → 33.70%** (−31%) and saves **15.9% GPU time**
- VAD eliminates **1,459 silence-to-ASR leaks** (2,719 → 1,260), reducing unnecessary ASR calls by **21.5%**
- Hallucination filter catches **776 hallucinated segments** but increases WER (filters out some valid text)
- BARTpho adds significant GPU overhead (+78.9%) — best suited for final output correction

> 📖 Run benchmarks: `modal run test/bench_table[1-4].py --samples 600`

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
