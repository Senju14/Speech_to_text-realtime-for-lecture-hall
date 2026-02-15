"""
BARTpho Syllable Correction — ASR Post-processing

Merged: Ricky13170's 3-layer English detection (diacritics → abbreviations → word list)
+ safety-length check + EN/VI chunk splitting.
"""

import re
import logging
from typing import Optional, List, Tuple
import torch

logger = logging.getLogger(__name__)

BARTPHO_ADAPTER = "522H0134-NguyenNhatHuy/bartpho-syllable-correction"
BARTPHO_BASE = "vinai/bartpho-syllable"

# ============================================================
# English Detection — Three Layers (from Ricky13170)
# ============================================================

_VIET_CHARS = set(
    "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợ"
    "ùúủũụưứừửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ"
    "ÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"
)

_ASCII_WORD = re.compile(r"^[A-Za-z0-9]+$")
_PUNCT = re.compile(r"^[^\w\s]+|[^\w\s]+$")

_COMMON_EN_ABBREV = {
    "AI", "ML", "NLP", "GPU", "CPU", "API", "LLM", "CNN", "RNN", "GAN",
    "IoT", "SQL", "LSTM", "BERT", "GPT", "RAM", "SSD", "HDD",
    "USB", "HTTP", "HTTPS", "URL", "HTML", "CSS", "JSON", "XML", "REST",
    "OK", "IT", "CV", "IP", "OS", "UI", "UX", "ID", "ASIC", "FPGA",
    "PDF", "SDK", "CLI", "GUI", "OOP", "MVP", "POC", "SaaS", "PaaS",
    "AWS", "GCP", "CUDA", "TPU", "VRAM", "FLOPS", "FPS",
}

_COMMON_EN_WORDS = {
    "machine", "learning", "deep", "network", "neural", "computer",
    "vision", "processing", "natural", "language", "algorithm",
    "training", "dataset", "feature", "transformer", "attention",
    "encoder", "decoder", "embedding", "classification", "regression",
    "clustering", "segmentation", "detection", "recognition",
    "generation", "reinforcement", "supervised", "unsupervised",
    "optimization", "gradient", "backpropagation", "overfitting",
    "convolution", "pooling", "recurrent", "generative", "discriminative",
    "pretrained", "fine", "tuning", "inference", "prediction",
    "benchmark", "baseline", "hyperparameter", "parameter", "weight",
    "batch", "epoch", "loss", "accuracy", "precision", "recall",
    "audio", "video", "online", "offline", "download", "upload",
    "streaming", "podcast", "channel", "content", "creator",
    "homework", "deadline", "project", "assignment", "presentation",
    "slide", "demo", "tutorial", "workshop", "feedback", "review",
    "paper", "conference", "journal", "thesis", "abstract",
    "research", "experiment", "evaluation", "metric",
    "software", "hardware", "server", "database", "cloud", "framework",
    "code", "debug", "deploy", "compile", "runtime", "interface",
    "function", "variable", "string", "array", "module", "library",
    "frontend", "backend", "fullstack", "container", "docker",
    "microservice", "pipeline", "workflow", "script", "repository",
}

_MIN_LENGTH_RATIO = 0.5


def _strip_punct(word: str) -> str:
    return _PUNCT.sub("", word)


def _is_english_word(word: str) -> bool:
    if not word:
        return False
    if any(c in _VIET_CHARS for c in word):
        return False
    if not _ASCII_WORD.match(word):
        return False
    if word in _COMMON_EN_ABBREV or word.upper() in _COMMON_EN_ABBREV:
        return True
    if len(word) >= 2 and word.isupper():
        return True
    if any(c.isupper() for c in word[1:]):
        return True
    if word[0].isupper() and len(word) >= 4:
        return True
    if word.lower() in _COMMON_EN_WORDS:
        return True
    return False


def split_en_vi(text: str) -> List[Tuple[str, bool]]:
    """Split text into (text, is_english) chunks with 3-layer detection."""
    words = text.split()
    if not words:
        return []

    classifications = []
    for word in words:
        clean = _strip_punct(word)
        is_en = _is_english_word(clean)
        is_ascii = bool(_ASCII_WORD.match(clean)) if clean else False
        classifications.append((word, is_en, is_ascii))

    # Consecutive ASCII heuristic
    i = 0
    while i < len(classifications):
        word, is_en, is_ascii = classifications[i]
        if is_ascii and not is_en:
            j = i
            while j < len(classifications) and classifications[j][2]:
                j += 1
            if j - i >= 2:
                for k in range(i, j):
                    w, _, a = classifications[k]
                    classifications[k] = (w, True, a)
        i += 1

    # Group contiguous chunks
    chunks = []
    cur_words = [classifications[0][0]]
    cur_is_en = classifications[0][1]

    for word, is_en, _ in classifications[1:]:
        if is_en == cur_is_en:
            cur_words.append(word)
        else:
            chunks.append((" ".join(cur_words), cur_is_en))
            cur_words = [word]
            cur_is_en = is_en

    chunks.append((" ".join(cur_words), cur_is_en))
    return chunks


class BARTphoCorrector:
    """Vietnamese syllable-level error correction using BARTpho + LoRA"""

    def __init__(
        self,
        adapter_id: str = BARTPHO_ADAPTER,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        self.adapter_id = adapter_id
        self.device = device
        self.cache_dir = cache_dir

        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def load_model(self):
        """Load BARTpho base + LoRA adapter"""
        if self.is_loaded:
            return

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from peft import PeftModel
        from src.utils import suppress_stdout

        logger.info(f"Loading BARTpho corrector: {self.adapter_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_id,
            cache_dir=self.cache_dir,
        )

        with suppress_stdout():
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                BARTPHO_BASE,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16,
            )
            self.model = PeftModel.from_pretrained(
                base_model,
                self.adapter_id,
                cache_dir=self.cache_dir,
            )

        self.model = self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        logger.info("BARTpho corrector ready")

    def correct(self, text: str, max_length: int = 256) -> str:
        """
        Correct Vietnamese ASR transcription errors.
        Preserves English chunks using split_en_vi().
        """
        if not text or not text.strip():
            return ""

        if not self.is_loaded:
            self.load_model()

        try:
            chunks = split_en_vi(text)
            result_parts = []

            for chunk_text, is_english in chunks:
                if is_english:
                    result_parts.append(chunk_text)
                else:
                    corrected = self._infer(chunk_text, max_length)
                    # Safety: reject if output too short
                    in_words = len(chunk_text.split())
                    out_words = len(corrected.split()) if corrected else 0
                    if in_words > 0 and out_words / in_words < _MIN_LENGTH_RATIO:
                        logger.warning(f"BARTpho output too short, using original")
                        result_parts.append(chunk_text)
                    else:
                        result_parts.append(corrected or chunk_text)

            return " ".join(result_parts).strip()

        except Exception as e:
            logger.error(f"BARTpho correction error: {e}")
            return text

    def _infer(self, text: str, max_length: int = 256) -> str:
        """Run BARTpho inference on a single Vietnamese chunk."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
