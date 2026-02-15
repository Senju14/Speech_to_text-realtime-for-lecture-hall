"""
SpeechSegmentBuffer — overlap-based segment management (from Ricky13170).

Segments audio by VAD boundaries with overlap for context continuity.
"""

import numpy as np
from src.config import SAMPLE_RATE, MAX_SEGMENT_SEC, OVERLAP_SEC, SILENCE_LIMIT


class SpeechSegmentBuffer:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        max_sec: float = MAX_SEGMENT_SEC,
        overlap_sec: float = OVERLAP_SEC,
        silence_limit: float = SILENCE_LIMIT,
    ):
        self.sr = sample_rate
        self.max_sec = max_sec
        self.overlap_sec = overlap_sec
        self.silence_limit = silence_limit
        self.reset()

    def reset(self):
        self.in_speech = False
        self.segment = []
        self.overlap = np.zeros(0, dtype=np.float32)
        self.last_voice_ts = 0.0

    def process(self, audio: np.ndarray, is_speech: bool, now_ts: float):
        """
        Process an audio chunk.
        
        Returns:
            None — still accumulating
            ("final", audio_chunk) — segment ready for transcription
        """
        if is_speech:
            if not self.in_speech:
                self.in_speech = True
                self.segment = [self.overlap] if len(self.overlap) > 0 else []

            self.last_voice_ts = now_ts
            self.segment.append(audio)

            total_samples = sum(len(c) for c in self.segment)
            total_sec = total_samples / self.sr

            if total_sec >= self.max_sec:
                chunk = np.concatenate(self.segment)
                self._update_overlap(chunk)
                self.segment = []
                self.in_speech = False
                return "final", chunk

            return None

        if not self.in_speech:
            return None

        silence_duration = now_ts - self.last_voice_ts
        self.segment.append(audio)

        total_samples = sum(len(c) for c in self.segment)
        total_sec = total_samples / self.sr

        if silence_duration >= self.silence_limit:
            chunk = np.concatenate(self.segment)
            self._update_overlap(chunk)
            self.reset()
            return "final", chunk

        if total_sec >= self.max_sec:
            chunk = np.concatenate(self.segment)
            self._update_overlap(chunk)
            self.segment = []
            self.in_speech = False
            return "final", chunk

        return None

    def _update_overlap(self, chunk: np.ndarray):
        n_samples = int(self.overlap_sec * self.sr)
        self.overlap = chunk[-n_samples:] if len(chunk) >= n_samples else chunk.copy()

    def get_current_duration(self) -> float:
        if not self.segment:
            return 0.0
        return sum(len(c) for c in self.segment) / self.sr

    def get_current_audio(self) -> np.ndarray:
        if not self.segment:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.segment)
