"""
LocalAgreement Policy for Streaming ASR

Based on SimulStreaming paper techniques:
- Wait until consecutive outputs agree before committing
- Reduces flickering and improves accuracy
- Confidence-adaptive commit thresholds
"""

from typing import Optional, List
from difflib import SequenceMatcher


class LocalAgreementTracker:
    """
    Tracks partial outputs and commits when stable.
    
    Algorithm:
    1. Collect consecutive partial outputs
    2. Find longest common prefix between last N outputs
    3. If common prefix is longer than committed text, commit it
    4. High confidence outputs can bypass agreement check
    """
    
    def __init__(self, min_agreement: int = 2, high_conf_threshold: float = 0.85):
        self.min_agreement = min_agreement
        self.high_conf_threshold = high_conf_threshold
        self.history: List[str] = []
        self.committed_text = ""
        self.committed_length = 0
    
    def reset(self):
        """Reset tracker for new segment"""
        self.history.clear()
        self.committed_text = ""
        self.committed_length = 0
    
    def add_partial(self, text: str, confidence: float = 0.0) -> Optional[str]:
        """
        Add a partial output and check if we should commit.
        
        Returns:
            New committed text if ready, None otherwise
        """
        if not text or not text.strip():
            return None
        
        text = text.strip()
        
        # High confidence: commit immediately
        if confidence >= self.high_conf_threshold:
            if len(text) > self.committed_length:
                self.committed_text = text
                self.committed_length = len(text)
                self.history.clear()
                return text
        
        # Add to history
        self.history.append(text)
        
        # Keep only last N for memory efficiency
        if len(self.history) > self.min_agreement + 2:
            self.history = self.history[-(self.min_agreement + 2):]
        
        # Check for agreement
        if len(self.history) >= self.min_agreement:
            common = self._find_common_prefix(self.history[-self.min_agreement:])
            
            if common and len(common) > self.committed_length:
                self.committed_text = common
                self.committed_length = len(common)
                return common
        
        return None
    
    def _find_common_prefix(self, texts: List[str]) -> str:
        """Find the longest common prefix among texts"""
        if not texts:
            return ""
        
        if len(texts) == 1:
            return texts[0]
        
        # Word-level comparison for better accuracy
        words_list = [t.split() for t in texts]
        min_len = min(len(w) for w in words_list)
        
        common_words = []
        for i in range(min_len):
            word = words_list[0][i]
            if all(w[i] == word for w in words_list):
                common_words.append(word)
            else:
                break
        
        return " ".join(common_words)
    
    def get_committed(self) -> str:
        """Get the current committed text"""
        return self.committed_text
    
    def get_pending(self) -> str:
        """Get text that's pending agreement"""
        if self.history:
            latest = self.history[-1]
            if len(latest) > self.committed_length:
                return latest[self.committed_length:].strip()
        return ""


class ConfidenceAdaptiveCommit:
    """
    Adaptive commit policy based on confidence levels.
    
    High confidence (>0.85): Commit immediately
    Medium confidence (0.5-0.85): Wait for LocalAgreement
    Low confidence (<0.5): Wait longer or discard
    """
    
    def __init__(self):
        self.high_threshold = 0.85
        self.medium_threshold = 0.5
        self.low_threshold = 0.3
    
    def get_policy(self, confidence: float) -> str:
        """
        Returns policy: 'commit', 'agree', 'wait', or 'discard'
        """
        if confidence >= self.high_threshold:
            return 'commit'
        elif confidence >= self.medium_threshold:
            return 'agree'
        elif confidence >= self.low_threshold:
            return 'wait'
        else:
            return 'discard'
