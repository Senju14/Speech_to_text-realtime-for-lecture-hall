"""
WhisperX ASR Backend

WhisperX provides batched inference with:
- Faster-whisper backend
- Built-in Pyannote VAD
- Word-level alignment

Reference: https://github.com/m-bain/whisperX
"""

import numpy as np
import logging
from src.config import WHISPER_MODEL, WHISPER_LANGUAGE, WHISPER_COMPUTE_TYPE, ASR_DEVICE
logger = logging.getLogger(__name__)


class WhisperXASR:
    """WhisperX ASR with word-level alignment"""
    
    def __init__(
        self,
        model_size: str = WHISPER_MODEL,
        language: str = WHISPER_LANGUAGE,
        device: str = ASR_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        
        self.model = None
        self.align_model = None
        self.align_metadata = None
    
    def load_model(self):
        """Load WhisperX and alignment models"""
        if self.model is not None:
            return
        
        # Apply torch.load patch before importing whisperx
        from src.utils import apply_torch_load_patch
        apply_torch_load_patch()
        
        import whisperx
        
        logger.info(f"Loading WhisperX: {self.model_size}")
        
        self.model = whisperx.load_model(
            self.model_size,
            self.device,
            compute_type=self.compute_type,
        )
        
        logger.info(f"Loading alignment model for '{self.language}'")
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=self.language,
            device=self.device,
        )
        
        logger.info("WhisperX ready")
    
    def transcribe(self, audio: np.ndarray, batch_size: int = 16) -> dict:
        """
        Transcribe audio with word-level alignment
        
        Args:
            audio: float32 audio at 16kHz
            batch_size: Batch size for inference
            
        Returns:
            dict with 'segments' containing text and word timestamps
        """
        if self.model is None:
            self.load_model()
        
        import whisperx
        
        # Transcribe
        result = self.model.transcribe(
            audio,
            batch_size=batch_size,
            language=self.language,
        )
        
        # Align for word-level timestamps
        result = whisperx.align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        
        return result
    
    def transcribe_segment(self, audio: np.ndarray) -> dict:
        """
        Transcribe a single audio segment
        
        Args:
            audio: float32 audio at 16kHz (typically 0.5-6 seconds)
            
        Returns:
            dict with 'text' and 'words' list
        """
        if len(audio) < 8000:  # < 0.5 seconds
            return {"text": "", "words": [], "segments": []}
        
        result = self.transcribe(audio, batch_size=1)
        
        text_parts = []
        all_words = []
        
        for seg in result.get("segments", []):
            text_parts.append(seg.get("text", "").strip())
            
            for word in seg.get("words", []):
                all_words.append({
                    "word": word.get("word", ""),
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                })
        
        return {
            "text": " ".join(text_parts).strip(),
            "words": all_words,
            "segments": result.get("segments", []),
        }
