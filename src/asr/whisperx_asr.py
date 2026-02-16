"""
WhisperX ASR Backend — Merged Best-of-Both

Senju14: multi-language, word-level alignment, align model caching
Ricky13170: AdaptiveNormalizer (highpass + soft clip + adaptive noise floor)
"""

import numpy as np
import logging
from dataclasses import replace as dataclass_replace
from src.config import (
    WHISPER_MODEL, WHISPER_LANGUAGE, WHISPER_COMPUTE_TYPE, ASR_DEVICE,
    supports_alignment,
)
logger = logging.getLogger(__name__)


# ============================================================
# Adaptive Audio Normalizer (from Ricky13170)
# ============================================================

class AdaptiveNormalizer:
    """Audio normalization with highpass filter, soft clipping, adaptive noise floor."""

    def __init__(self, target_dB: float = -20.0, sample_rate: int = 16000):
        self.target_dB = target_dB
        self.target_rms = 10 ** (target_dB / 20)
        self.sample_rate = sample_rate
        self.min_gain = 0.3
        self.max_gain = 5.0
        self.noise_floor = 0.001
        self.rms_history = []
        self.max_history = 50

    def _highpass(self, audio: np.ndarray, cutoff: int = 80) -> np.ndarray:
        from scipy.signal import butter, lfilter
        nyq = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyq
        if normal_cutoff >= 1.0:
            return audio
        b, a = butter(2, normal_cutoff, btype="high")
        return lfilter(b, a, audio)

    @staticmethod
    def _rms_speech(audio: np.ndarray, threshold: float = 0.005) -> float:
        speech = audio[np.abs(audio) > threshold]
        return np.sqrt(np.mean(speech ** 2)) if len(speech) else 1e-6

    @staticmethod
    def _soft_clip(audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        if np.max(np.abs(audio)) > threshold:
            audio = np.tanh(audio * 0.8)
        return audio

    def _update_noise_floor(self, audio: np.ndarray):
        rms = np.sqrt(np.mean(audio ** 2))
        self.rms_history.append(rms)
        if len(self.rms_history) > self.max_history:
            self.rms_history.pop(0)
        if len(self.rms_history) >= 10 and len(self.rms_history) % 10 == 0:
            sorted_rms = sorted(self.rms_history)
            self.noise_floor = sorted_rms[len(sorted_rms) // 5]

    def normalize(self, audio: np.ndarray) -> tuple[np.ndarray, dict]:
        if len(audio) < self.sample_rate * 0.5:
            return audio, {"skipped": True}
        self._update_noise_floor(audio)
        audio = self._highpass(audio, cutoff=80)
        rms_original = self._rms_speech(audio)
        gain = np.clip(self.target_rms / rms_original, self.min_gain, self.max_gain)
        audio = audio * gain
        audio = self._soft_clip(audio, threshold=0.95)
        rms_after = self._rms_speech(audio)
        audio = audio * (self.target_rms / rms_after)
        audio = audio.astype(np.float32, copy=False)
        stats = {"original_dB": 20 * np.log10(rms_original + 1e-10), "gain": gain}
        return audio, stats


class WhisperXASR:
    """WhisperX ASR with word-level alignment and multi-language support"""
    
    def __init__(
        self,
        model_size: str = WHISPER_MODEL,
        language: str = WHISPER_LANGUAGE,
        device: str = ASR_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        self.model_size = model_size
        self.default_language = language  # Default lang; None = multilingual
        self.device = device
        self.compute_type = compute_type
        
        self.model = None
        # Lazy-loaded alignment models: {lang_code: (model, metadata)}
        self._align_cache = {}
        
        # Ricky: adaptive normalizer for lecture hall audio
        self.normalizer = AdaptiveNormalizer(target_dB=-20.0, sample_rate=16000)
        logger.info("[ASR] WhisperX with adaptive normalization enabled")
    
    def load_model(self):
        """Load WhisperX model (multilingual) and default alignment model"""
        if self.model is not None:
            return
        
        # Apply torch.load patch before importing whisperx
        from src.utils import apply_torch_load_patch, suppress_stdout
        apply_torch_load_patch()
        
        import whisperx
        
        logger.info(f"Loading WhisperX: {self.model_size} (default lang: {self.default_language or 'multilingual'})")
        
        # Suppress noisy print() from pyannote/pytorch_lightning during loading
        with suppress_stdout():
            self.model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type=self.compute_type,
                language=self.default_language,
            )
        
        # Pre-load alignment model for default language
        if self.default_language and supports_alignment(self.default_language):
            self._load_align_model(self.default_language)
        
        logger.info("WhisperX ready")
    
    def _load_align_model(self, language_code: str):
        """Load and cache alignment model for a specific language"""
        if language_code in self._align_cache:
            return self._align_cache[language_code]
        
        if not supports_alignment(language_code):
            logger.debug(f"[Align] No alignment support for '{language_code}'")
            return None, None
        
        try:
            import whisperx
            from src.utils import suppress_stdout
            
            logger.info(f"[Align] Loading alignment model for '{language_code}'")
            with suppress_stdout():
                model, metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=self.device,
                )
            self._align_cache[language_code] = (model, metadata)
            logger.info(f"[Align] Cached alignment model for '{language_code}'")
            return model, metadata
        except Exception as e:
            logger.warning(f"[Align] Failed to load alignment for '{language_code}': {e}")
            self._align_cache[language_code] = (None, None)
            return None, None
    
    def transcribe(
        self,
        audio: np.ndarray,
        batch_size: int = 16,
        initial_prompt: str = None,
        skip_align: bool = False,
        language: str = None,
    ) -> dict:
        """
        Transcribe audio with optional word-level alignment
        
        Args:
            audio: float32 audio at 16kHz
            batch_size: Batch size for inference
            initial_prompt: Context keywords to prime the model
            skip_align: If True, skip whisperx.align() for faster streaming
            language: Override language for this call (None = use default or auto-detect)
            
        Returns:
            dict with 'segments' and 'language' (detected language code)
        """
        if self.model is None:
            self.load_model()
        
        import whisperx
        
        # Determine language for this transcription
        lang = language if language is not None else self.default_language
        
        # Ricky: normalize audio before transcription
        normalized, _ = self.normalizer.normalize(audio)
        if normalized.dtype != np.float32:
            normalized = normalized.astype(np.float32)
        
        # Set initial_prompt via model options (WhisperX uses TranscriptionOptions,
        # NOT a transcribe() keyword argument)
        original_options = None
        if initial_prompt and hasattr(self.model, 'options'):
            original_options = self.model.options
            self.model.options = dataclass_replace(
                self.model.options, initial_prompt=initial_prompt
            )
        
        try:
            # Transcribe (lang=None enables auto-detect)
            result = self.model.transcribe(
                normalized,
                batch_size=batch_size,
                language=lang,
            )
        finally:
            # Restore original options (model is shared across sessions)
            if original_options is not None:
                self.model.options = original_options
        
        # Get detected language (WhisperX returns it when auto-detecting)
        detected_lang = result.get("language", lang)
        
        # Alignment: use detected language to pick the right model
        if not skip_align and detected_lang:
            align_model, align_metadata = self._load_align_model(detected_lang)
            if align_model is not None:
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    normalized,
                    self.device,
                    return_char_alignments=False,
                )
        
        # Always include detected language in result
        result["language"] = detected_lang
        return result
    
    def transcribe_segment(
        self,
        audio: np.ndarray,
        initial_prompt: str = None,
        skip_align: bool = True,
        language: str = None,
    ) -> dict:
        """
        Transcribe a single audio segment (optimized for streaming)
        
        Args:
            audio: float32 audio at 16kHz (typically 0.5-10 seconds)
            initial_prompt: Context keywords for domain accuracy
            skip_align: Skip word-level alignment for faster streaming (default: True)
            language: Override language (None = use default or auto-detect)
            
        Returns:
            dict with 'text', 'words', 'language' (detected)
        """
        if len(audio) < 8000:  # < 0.5 seconds
            return {"text": "", "words": [], "segments": [], "language": language or self.default_language}
        
        result = self.transcribe(
            audio, batch_size=1,
            initial_prompt=initial_prompt,
            skip_align=skip_align,
            language=language,
        )
        
        text_parts = []
        all_words = []
        
        for seg in result.get("segments", []):
            text_parts.append(seg.get("text", "").strip())
            
            for word in seg.get("words", []):
                all_words.append({
                    "word": word.get("word", ""),
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                    "score": word.get("score", 0.0),
                })
        
        return {
            "text": " ".join(text_parts).strip(),
            "words": all_words,
            "segments": result.get("segments", []),
            "language": result.get("language", language or self.default_language),
        }
