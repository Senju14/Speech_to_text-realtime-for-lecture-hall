"""
HallucinationFilter — Multi-layer filtering (from Ricky13170).

Layers:
- MD5 repetition detection 
- Exact hallucination set (filler words, music markers)
- Forbidden substrings (YouTube artifacts)
- Confidence threshold
- RMS-text mismatch (quiet audio → long text)
- Repeated single-word detection
"""

import hashlib
from collections import deque
import logging

logger = logging.getLogger(__name__)


class HallucinationFilter:
    """Multi-layer hallucination filter for Whisper ASR output."""

    def __init__(self, history_size: int = 5):
        self.recent_hashes = deque(maxlen=history_size)
        self.recent_texts = deque(maxlen=history_size)

        self.exact_hallucinations = {
            "ừ", "à", "ờ", "ồ", "ừm", "hử", "hả",
            "music", "♪", "nhạc", "[music]", "[âm nhạc]",
            "uh", "um", "hmm",
        }

        self.forbidden_substrings = [
            "subscribe", "đăng ký kênh", "like and subscribe",
            "ghiền mì gõ", "la la school",
            "hẹn gặp lại", "cảm ơn đã xem", "cảm ơn các bạn đã theo dõi",
            "thank you for watching", "thanks for watching",
            "trong video tiếp theo", "video trước", "phần tiếp theo",
            "[", "]", ">>", "<<",
            "♪♪", "nhạc nhạc",
        ]

    def _hash_text(self, text: str) -> str:
        normalized = text.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:8]

    def is_hallucination(
        self, text: str, audio_rms: float = 1.0, confidence: float = 1.0
    ) -> tuple[bool, str]:
        """
        Multi-layer hallucination check.
        
        Returns:
            (is_hallucination, reason)
        """
        if not text or not text.strip():
            return True, "empty"

        text = text.strip()
        text_lower = text.lower()
        words = text.split()

        if len(text) < 2:
            return True, "too_short"

        # RMS-text mismatch: quiet audio shouldn't produce long text
        if audio_rms < 0.02 and len(words) > 6:
            return True, "quiet_audio_long_text"

        if text_lower in self.exact_hallucinations:
            return True, f"exact:{text}"

        for pattern in self.forbidden_substrings:
            if pattern in text_lower:
                return True, f"forbidden:{pattern}"

        text_hash = self._hash_text(text)
        if text_hash in self.recent_hashes:
            return True, "repetition"

        if confidence < 0.15:
            return True, "low_confidence"

        if len(set(words)) == 1 and len(words) >= 3:
            return True, "repeated_word"

        self.recent_hashes.append(text_hash)
        self.recent_texts.append(text)
        return False, "ok"

    def reset(self):
        self.recent_hashes.clear()
        self.recent_texts.clear()

    def get_history(self) -> list[str]:
        return list(self.recent_texts)
