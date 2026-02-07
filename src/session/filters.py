"""
Hallucination Filter

Detects and filters common hallucination patterns from Whisper models.
These patterns often appear when:
- Audio is mostly silence/noise
- Model is uncertain
- Training data leakage (YouTube artifacts)
"""

import re
from typing import List, Set


# Common hallucination patterns for Vietnamese ASR
HALLUCINATION_PATTERNS: List[str] = [
    # YouTube artifacts
    "subscribe",
    "đăng ký kênh",
    "like",
    "comment",
    "share",
    "bell",
    "notification",
    
    # Channel names (training leakage)
    "ghiền mì gõ",
    "la la school",
    "kênh youtube",
    "channel",
    
    # Sign-offs
    "hẹn gặp lại",
    "cảm ơn đã xem",
    "cảm ơn các bạn đã theo dõi",
    "thank you for watching",
    "thanks for watching",
    "bye bye",
    "see you",
    "goodbye",
    
    # Video transitions
    "trong video tiếp theo",
    "video trước",
    "phần tiếp theo",
    
    # Music markers
    "[music]",
    "[nhạc]",
    "♪",
    "(music)",
    "(nhạc)",
    
    # Silence markers
    "[im lặng]",
    "[silence]",
    "...",
]

# Patterns that are suspicious if they appear alone
SUSPICIOUS_SHORT_PATTERNS: Set[str] = {
    "ờ",
    "ừ", 
    "à",
    "uh",
    "um",
    "ah",
}


class HallucinationFilter:
    """Filter for detecting Whisper hallucinations"""
    
    def __init__(self, patterns: List[str] = None, min_length: int = 2):
        self.patterns = [p.lower() for p in (patterns or HALLUCINATION_PATTERNS)]
        self.min_length = min_length
    
    def is_hallucination(self, text: str) -> bool:
        """
        Check if text is likely a hallucination
        
        Args:
            text: Transcribed text
            
        Returns:
            True if text matches hallucination patterns
        """
        if not text:
            return True
        
        text_lower = text.lower().strip()
        
        # Too short
        if len(text_lower) < self.min_length:
            return True
        
        # Check patterns
        for pattern in self.patterns:
            if pattern in text_lower:
                return True
        
        # Only filler words
        words = text_lower.split()
        if len(words) <= 2:
            non_filler = [w for w in words if w not in SUSPICIOUS_SHORT_PATTERNS]
            if len(non_filler) == 0:
                return True
        
        return False
    
    def filter_text(self, text: str) -> str:
        """
        Remove hallucination patterns from text
        
        Args:
            text: Original text
            
        Returns:
            Cleaned text or empty string if all hallucination
        """
        if self.is_hallucination(text):
            return ""
        
        return text.strip()
