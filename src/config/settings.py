"""
Configuration settings for ASR Thesis

NOTE: Modal infrastructure config (MODAL_APP_NAME, MODAL_GPU, etc.) is in main.py
because Modal needs those values at deploy time before the container starts.
This file contains runtime configuration only.

API keys are loaded from environment variables (set via .env file or Modal Secrets).
"""

import os

# =============================================================================
# WhisperX ASR
# =============================================================================
WHISPER_MODEL = "large-v3"
WHISPER_LANGUAGE = None  # None = multilingual (auto-detect per session)
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_BATCH_SIZE = 16
ASR_DEVICE = "cuda"

# =============================================================================
# Voice Activity Detection (Silero VAD)
# =============================================================================
VAD_THRESHOLD = 0.5

# =============================================================================
# Segmentation (SpeechSegmentBuffer — Ricky's overlap-based architecture)
# =============================================================================
MAX_SEGMENT_SEC = 5.0         # Max segment duration before force finalize
OVERLAP_SEC = 0.8             # Overlap between consecutive segments
SILENCE_LIMIT = 0.3           # Silence to trigger FINAL (lower = faster response)
MIN_DECODE_SEC = 1.0          # Min audio duration for partial decode

# Legacy aliases (used by Senju14 handler code)
MIN_SILENCE_DURATION = SILENCE_LIMIT
MAX_BUFFER_DURATION = MAX_SEGMENT_SEC
MIN_SEGMENT_DURATION = 0.3

# =============================================================================
# Local Agreement (stabilizes partial transcripts — from Ricky)
# =============================================================================
AGREEMENT_N = 3               # Number of consecutive agreements needed

# =============================================================================
# Hallucination Filter
# =============================================================================
HALLUCINATION_HISTORY_SIZE = 5

# =============================================================================
# NLLB Translation
# =============================================================================
NLLB_MODEL = "facebook/nllb-200-1.3B" 
NLLB_USE_8BIT = True  # 8-bit quantization for 3.3B: ~3.5GB VRAM vs ~6.6GB

NLLB_SRC_LANG = "vie_Latn"
NLLB_TGT_LANG = "eng_Latn"
NLLB_DEVICE = "cuda"
NLLB_MAX_LENGTH = 128  # Increased from 64 for longer sentences
NLLB_NUM_BEAMS = 2     # Greedy decoding for speed (beam=2 is ~2x slower)

# =============================================================================
# Audio Format
# =============================================================================
SAMPLE_RATE = 16000
# Note: Frontend sends audio every 250ms (see frontend/js/audio.js SEND_INTERVAL_MS)

# =============================================================================
# Cache Paths (Modal volume)
# =============================================================================
CACHE_DIR = "/cache"
HF_CACHE_DIR = "/cache/huggingface"
NLLB_CACHE_DIR = "/cache/nllb"

# =============================================================================
# Post-processing (BARTpho syllable correction)
# =============================================================================
ENABLE_BARTPHO = True   # BARTpho syllable correction for Vietnamese (Ricky)
BARTPHO_ADAPTER = "522H0134-NguyenNhatHuy/bartpho-syllable-correction"
BARTPHO_DEVICE = "cuda"

# =============================================================================
# Groq LLM (for context priming & auto-summary)
# =============================================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # From .env or Modal Secret
GROQ_MODEL = "llama-3.1-8b-instant"  # Fast & cheap
GROQ_TIMEOUT = 10  
AUTO_SUMMARY_MIN_DURATION = 120  # 2 minutes
