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
WHISPER_LANGUAGE = "vi"
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_BATCH_SIZE = 16
ASR_DEVICE = "cuda"

# =============================================================================
# Voice Activity Detection
# =============================================================================
VAD_THRESHOLD = 0.5
MIN_SILENCE_DURATION = 0.3   # Seconds of silence to finalize segment (tuned for real-time)
MAX_BUFFER_DURATION = 4.0    # Max buffer duration before force finalize (tuned for real-time)
MIN_SEGMENT_DURATION = 0.3   # Minimum segment duration to process

# =============================================================================
# NLLB Translation
# =============================================================================
# Model options (choose ONE):
# - "facebook/nllb-200-3.3B"        : Best quality, ~13GB VRAM, ~2s/sentence
# - "facebook/nllb-200-distilled-600M" : Fast, ~2.5GB VRAM, ~0.3s/sentence (RECOMMENDED)
# - "facebook/nllb-200-1.3B"        : Balance, ~5GB VRAM, ~0.8s/sentence
NLLB_MODEL = "facebook/nllb-200-distilled-600M"  # Optimized for real-time

# 8-bit quantization: Reduces VRAM by ~50%, slight quality loss
# Requires: pip install bitsandbytes
NLLB_USE_8BIT = False  # Set True if using 3.3B model to reduce VRAM

NLLB_SRC_LANG = "vie_Latn"
NLLB_TGT_LANG = "eng_Latn"
NLLB_DEVICE = "cuda"
NLLB_MAX_LENGTH = 128  # Increased from 64 for longer sentences
NLLB_NUM_BEAMS = 1     # Greedy decoding for speed (beam=2 is ~2x slower)

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
BARTPHO_ADAPTER = "522H0134-NguyenNhatHuy/bartpho-syllable-correction"
BARTPHO_DEVICE = "cuda"

# =============================================================================
# Groq LLM (for context priming & auto-summary)
# =============================================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # From .env or Modal Secret
GROQ_MODEL = "llama-3.1-8b-instant"  # Fast & cheap
GROQ_TIMEOUT = 10  
AUTO_SUMMARY_MIN_DURATION = 120  # 2 minutes
