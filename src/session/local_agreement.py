"""
LocalAgreement — partial text stabilization algorithm (from Ricky13170).

Compares N consecutive transcriptions and outputs only the common prefix
as "stable". Reduces flickering text in real-time display.
"""

import logging

logger = logging.getLogger(__name__)


class LocalAgreement:
    def __init__(self, n: int = 2):
        self.n = n
        self.history = []

    def process(self, new_text: str) -> tuple[str, str]:
        """Returns (stable_text, unstable_text)."""
        if not new_text or not new_text.strip():
            return "", ""

        words = new_text.strip().split()
        self.history.append(words)

        if len(self.history) > self.n:
            self.history = self.history[-self.n:]

        if len(self.history) < self.n:
            return "", new_text

        common_words = self._find_common_prefix(self.history[-self.n:])

        if common_words:
            stable_text = " ".join(common_words)
            unstable_text = (
                " ".join(words[len(common_words):])
                if len(common_words) < len(words)
                else ""
            )
        else:
            stable_text = ""
            unstable_text = new_text

        return stable_text, unstable_text

    def _find_common_prefix(self, word_lists: list[list[str]]) -> list[str]:
        if not word_lists:
            return []
        if len(word_lists) == 1:
            return word_lists[0]

        min_len = min(len(wl) for wl in word_lists)
        common = []
        for i in range(min_len):
            words_at_i = [wl[i] for wl in word_lists]
            if all(w == words_at_i[0] for w in words_at_i):
                common.append(words_at_i[0])
            else:
                break
        return common

    def reset(self):
        self.history = []

    def get_current_stable(self) -> str:
        if len(self.history) < self.n:
            return ""
        common = self._find_common_prefix(self.history[-self.n:])
        return " ".join(common)
