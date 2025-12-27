# import numpy as np
# import time
# import torch
# from collections import deque
# import hashlib

# from backend.config import SAMPLE_RATE, WHISPER_DEVICE
# from backend.audio.sliding_window import SlidingWindow

# # E2E Model
# WHISPER_E2E_MODEL = "openai/whisper-large-v3"

# class HallucinationFilter:
#     """
#     Smart hallucination detection (no hardcoding!)
#     Uses statistical patterns to detect model hallucinations.
#     """
    
#     def __init__(self, history_size=5):
#         self.recent_hashes = deque(maxlen=history_size)
#         self.recent_texts = deque(maxlen=history_size)
    
#     def _hash_text(self, text: str) -> str:
#         """Create hash of normalized text"""
#         normalized = text.lower().strip()
#         return hashlib.md5(normalized.encode()).hexdigest()[:8]
    
#     def _get_repetition_ratio(self, text: str) -> float:
#         """
#         Calculate how repetitive the text is.
#         High ratio = many repeated words = likely hallucination
#         """
#         words = text.lower().split()
#         if len(words) < 3:
#             return 0.0
        
#         unique_words = set(words)
#         # Ratio: 1.0 = all unique, closer to 0 = very repetitive
#         uniqueness = len(unique_words) / len(words)
#         return 1.0 - uniqueness  # Invert: higher = more repetitive
    
#     def _is_similar_to_recent(self, text: str) -> bool:
#         """Check if text is too similar to recent outputs"""
#         text_hash = self._hash_text(text)
        
#         # Exact match check
#         if text_hash in self.recent_hashes:
#             return True
        
#         # Fuzzy similarity check (word overlap)
#         current_words = set(text.lower().split())
#         for recent in self.recent_texts:
#             recent_words = set(recent.lower().split())
#             if len(current_words) > 0 and len(recent_words) > 0:
#                 overlap = len(current_words & recent_words)
#                 max_len = max(len(current_words), len(recent_words))
#                 similarity = overlap / max_len
#                 if similarity > 0.8:  # 80% overlap = too similar
#                     return True
        
#         return False
    
#     def is_hallucination(self, text: str, audio_amplitude: float) -> tuple[bool, str]:
#         """
#         Detect if output is likely a hallucination.
#         Returns: (is_hallucination, reason)
#         """
#         if not text or len(text.strip()) < 2:
#             return True, "empty"
        
#         words = text.split()
#         word_count = len(words)
        
#         # 1. Energy-Length Ratio Check
#         # If audio is quiet but text is long, suspicious
#         # Threshold: quiet audio (amp < 0.05) should not produce > 5 words
#         if audio_amplitude < 0.05 and word_count > 5:
#             return True, "quiet_audio_long_text"
        
#         # 2. Very quiet audio (< 0.02) should produce max 3 words
#         if audio_amplitude < 0.02 and word_count > 3:
#             return True, "very_quiet_long_text"
        
#         # 3. Repetition Detection
#         repetition_ratio = self._get_repetition_ratio(text)
#         if repetition_ratio > 0.5 and word_count > 4:  # >50% repeated words
#             return True, "high_repetition"
        
#         # 4. Similar to recent outputs (likely stuck in loop)
#         if self._is_similar_to_recent(text):
#             return True, "similar_to_recent"
        
#         # Passed all checks - add to history
#         self.recent_hashes.append(self._hash_text(text))
#         self.recent_texts.append(text)
        
#         return False, "ok"
    
#     def reset(self):
#         """Reset filter state for new session"""
#         self.recent_hashes.clear()
#         self.recent_texts.clear()


# class WhisperStreaming:
#     """
#     E2E Streaming ASR + Translation using Whisper Large V3
#     With intelligent hallucination filtering
#     """
    
#     def __init__(self):
#         self.model = None
#         self.processor = None
#         self.device = WHISPER_DEVICE
#         self._loaded = False
#         self.segment_id = 0
        
#         # Window 2s, Overlap 1s
#         self.sliding_window = SlidingWindow(window_sec=2.0, overlap_sec=1.0)
#         self.context_prompt = None
        
#         # Anti-hallucination filter
#         self.hallucination_filter = HallucinationFilter(history_size=5)
        
#     def load(self):
#         if self._loaded: 
#             return
            
#         from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
#         print(f"[ASR] Loading E2E model: {WHISPER_E2E_MODEL}...")
#         start = time.time()
        
#         self.processor = WhisperProcessor.from_pretrained(WHISPER_E2E_MODEL)
#         self.model = WhisperForConditionalGeneration.from_pretrained(
#             WHISPER_E2E_MODEL,
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True,
#         )
#         self.model.to(self.device)
#         self.model.eval()
        
#         print(f"[ASR] E2E Model loaded in {time.time() - start:.1f}s")
#         self._loaded = True
    
#     def set_context(self, context=None):
#         if context: 
#             self.context_prompt = context[-200:]
    
#     def add_audio(self, audio_chunk: np.ndarray):
#         self.sliding_window.add_audio(audio_chunk)
    
#     @property
#     def buffer(self):
#         return self.sliding_window.buffer
    
#     def should_transcribe(self) -> bool:
#         return self.sliding_window.has_window()
    
#     def transcribe(self, timestamp: str) -> dict:
#         """
#         E2E Transcription + Translation with Hallucination Filtering
#         """
#         if self.sliding_window.total_samples < int(SAMPLE_RATE * 0.1):
#             return {}
        
#         audio = self.sliding_window.get_window()
        
#         max_amp = np.max(np.abs(audio))
#         print(f"[E2E] Buffer: {len(audio)} samples | Amp: {max_amp:.6f}")
        
#         # Skip very quiet audio
#         if max_amp < 0.001: 
#             print("[E2E] ...Too quiet (Skipping)")
#             return {}
            
#         start = time.time()
#         vi_text = ""
#         en_text = ""
        
#         try:
#             # Process audio with attention_mask
#             inputs = self.processor(
#                 audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
#             )
#             input_features = inputs.input_features.to(self.device).to(torch.float16)
#             attention_mask = torch.ones(input_features.shape[:2], dtype=torch.long, device=self.device)
            
#             # 1. TRANSCRIBE Vietnamese - Using language/task params instead of deprecated forced_decoder_ids
#             with torch.no_grad():
#                 vi_ids = self.model.generate(
#                     input_features,
#                     attention_mask=attention_mask,
#                     language="vi",
#                     task="transcribe",
#                     max_length=128,
#                     num_beams=1,
#                     do_sample=False
#                 )
#             vi_text = self.processor.batch_decode(vi_ids, skip_special_tokens=True)[0].strip()
            
#             # 🔥 HALLUCINATION CHECK for Vietnamese
#             is_hallucination, reason = self.hallucination_filter.is_hallucination(vi_text, max_amp)
            
#             if is_hallucination:
#                 print(f"⚠️ [HALLUCINATION] Filtered: '{vi_text[:50]}...' (reason: {reason})")
#                 self.sliding_window.slide()
#                 return {}
            
#             # 2. TRANSLATE to English - Using language/task params
#             with torch.no_grad():
#                 en_ids = self.model.generate(
#                     input_features,
#                     attention_mask=attention_mask,
#                     language="vi",
#                     task="translate",
#                     max_length=128,
#                     num_beams=1,
#                     do_sample=False
#                 )
#             en_text = self.processor.batch_decode(en_ids, skip_special_tokens=True)[0].strip()
            
#             self.sliding_window.slide()
            
#         except Exception as e:
#             print(f"[E2E] Error: {e}")
#             import traceback
#             traceback.print_exc()
#             return {}
        
#         print(f"🇻🇳 [VI] '{vi_text}'")
#         print(f"🇬🇧 [EN] '{en_text}'")

#         self.segment_id += 1
#         self.set_context(vi_text)
        
#         return {
#             "segment_id": self.segment_id,
#             "source": vi_text,
#             "target": en_text,
#             "timestamp": timestamp,
#             "processing_ms": int((time.time() - start) * 1000)
#         }
    
#     def finalize(self) -> str:
#         return ""
    
#     def reset(self):
#         self.sliding_window.clear()
#         self.segment_id = 0
#         self.context_prompt = None
#         self.hallucination_filter.reset()  # Reset filter too

import numpy as np
import time
import torch
import hashlib
from collections import deque

from backend.config import SAMPLE_RATE, WHISPER_DEVICE

WHISPER_E2E_MODEL = "openai/whisper-large-v3"


# ======================= Hallucination Filter =======================

class HallucinationFilter:
    def __init__(self, history_size=5):
        self.recent_hashes = deque(maxlen=history_size)
        self.recent_texts = deque(maxlen=history_size)

    def _hash(self, text):
        return hashlib.md5(text.lower().strip().encode()).hexdigest()[:8]

    def is_hallucination(self, text: str, amp: float):
        if not text or len(text.split()) < 2:
            return True, "empty"

        words = text.split()

        if amp < 0.02 and len(words) > 4:
            return True, "quiet_audio_long_text"

        h = self._hash(text)
        if h in self.recent_hashes:
            return True, "repeat"

        self.recent_hashes.append(h)
        self.recent_texts.append(text)
        return False, "ok"

    def reset(self):
        self.recent_hashes.clear()
        self.recent_texts.clear()


# ======================= Whisper Segment ASR =======================

class WhisperStreaming:
    """
    Segment-based Whisper ASR (VAD-driven)
    Stateless audio, only text context
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = WHISPER_DEVICE
        self.segment_id = 0
        self.context_prompt = None
        self.filter = HallucinationFilter()
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        print(f"[ASR] Loading {WHISPER_E2E_MODEL}...")
        start = time.time()

        self.processor = WhisperProcessor.from_pretrained(WHISPER_E2E_MODEL)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_E2E_MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.model.eval()
        self._loaded = True
        print(f"[ASR] Loaded in {time.time() - start:.1f}s")

    def reset(self):
        self.segment_id = 0
        self.context_prompt = None
        self.filter.reset()

    def set_context(self, text: str):
        if text:
            self.context_prompt = text[-200:]

    def transcribe(self, audio: np.ndarray, timestamp: str, final: bool):
        max_amp = float(np.max(np.abs(audio)))
        if max_amp < 0.002:
            return {}

        start = time.time()

        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        input_features = inputs.input_features.to(self.device).to(torch.float16)

        with torch.no_grad():
            vi_ids = self.model.generate(
                input_features,
                language="vi",
                task="transcribe",
                max_length=128,
                do_sample=False,
                num_beams=1,
            )

        vi_text = self.processor.batch_decode(
            vi_ids, skip_special_tokens=True
        )[0].strip()

        is_hallu, reason = self.filter.is_hallucination(vi_text, max_amp)
        if is_hallu:
            print(f"⚠️ Hallucination filtered ({reason})")
            return {}

        with torch.no_grad():
            en_ids = self.model.generate(
                input_features,
                language="vi",
                task="translate",
                max_length=128,
                do_sample=False,
                num_beams=1,
            )

        en_text = self.processor.batch_decode(
            en_ids, skip_special_tokens=True
        )[0].strip()

        self.segment_id += 1

        return {
            "segment_id": self.segment_id,
            "source": vi_text,
            "target": en_text,
            "timestamp": timestamp,
            "processing_ms": int((time.time() - start) * 1000)
        }
