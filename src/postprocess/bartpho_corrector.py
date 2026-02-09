"""
BARTpho Syllable Correction — ASR Post-processing

Fine-tuned LoRA adapter on vinai/bartpho-syllable for correcting
Vietnamese ASR transcription errors at the syllable level.

Model: 522H0134-NguyenNhatHuy/bartpho-syllable-correction
Base:  vinai/bartpho-syllable (MBartForConditionalGeneration)
Type:  LoRA adapter (r=64, alpha=128, ~138MB)

Usage:
    corrector = BARTphoCorrector()
    corrector.load_model()
    fixed = corrector.correct("xin trào các bạn")  # → "xin chào các bạn"
"""

import logging
from typing import Optional
import torch

logger = logging.getLogger(__name__)

BARTPHO_ADAPTER = "522H0134-NguyenNhatHuy/bartpho-syllable-correction"
BARTPHO_BASE = "vinai/bartpho-syllable"


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

        Args:
            text: Raw ASR output (Vietnamese)
            max_length: Max output token length

        Returns:
            Corrected Vietnamese text
        """
        if not text or not text.strip():
            return ""

        if not self.is_loaded:
            self.load_model()

        try:
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

            corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return corrected.strip()

        except Exception as e:
            logger.error(f"BARTpho correction error: {e}")
            return text  # Return original on error
