"""
Silero VAD - Neural Voice Activity Detection

Used for streaming VAD to detect speech/silence boundaries
in real-time before sending to WhisperX.

Reference: https://github.com/snakers4/silero-vad
"""

import numpy as np
import logging

from src.config import VAD_THRESHOLD, SAMPLE_RATE

logger = logging.getLogger(__name__)


class SileroVAD:
    """Silero VAD for real-time speech detection"""
    
    def __init__(self, threshold: float = VAD_THRESHOLD, sample_rate: int = SAMPLE_RATE):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model = None
        self._utils = None
    
    def load_model(self):
        """Load Silero VAD from torch hub"""
        if self.model is not None:
            return
        
        import torch
        
        logger.info("Loading Silero VAD...")
        
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.model.eval()
        self._utils = utils
        
        logger.info("Silero VAD loaded")
    
    def reset_state(self):
        """Reset internal state for new audio stream"""
        if self.model is not None:
            self.model.reset_states()
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Check if audio chunk contains speech
        
        Args:
            audio: float32 audio at 16kHz
            
        Returns:
            True if speech detected
        """
        if self.model is None:
            self.load_model()
        
        import torch
        
        # Silero requires 512 samples per chunk for 16kHz
        chunk_size = 512
        
        if len(audio) < chunk_size:
            audio = np.pad(audio, (0, chunk_size - len(audio)), mode='constant')
        
        # Check each 512-sample window
        num_chunks = len(audio) // chunk_size
        if num_chunks == 0:
            return False
        
        for i in range(num_chunks):
            chunk = audio[i * chunk_size : (i + 1) * chunk_size]
            audio_tensor = torch.from_numpy(chunk).float()
            
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
            if speech_prob > self.threshold:
                return True
        
        return False
    
    def get_speech_probability(self, audio: np.ndarray) -> float:
        """Get max speech probability across audio"""
        if self.model is None:
            self.load_model()
        
        import torch
        
        chunk_size = 512
        
        if len(audio) < chunk_size:
            audio = np.pad(audio, (0, chunk_size - len(audio)), mode='constant')
        
        num_chunks = len(audio) // chunk_size
        if num_chunks == 0:
            return 0.0
        
        max_prob = 0.0
        for i in range(num_chunks):
            chunk = audio[i * chunk_size : (i + 1) * chunk_size]
            audio_tensor = torch.from_numpy(chunk).float()
            
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
            max_prob = max(max_prob, speech_prob)
        
        return max_prob
