"""
NLLB Translation - No Language Left Behind

Optimized for real-time Vietnamese to English translation.

Model options:
- distilled-600M: Fast, low VRAM (recommended for real-time)
- 1.3B: Balanced quality/speed
- 3.3B: Best quality, use with 8-bit quantization

Reference: https://huggingface.co/facebook/nllb-200-distilled-600M
"""

import logging
from typing import Optional
import torch

from src.config import (
    NLLB_MODEL, NLLB_SRC_LANG, NLLB_TGT_LANG,
    NLLB_DEVICE, NLLB_MAX_LENGTH, NLLB_NUM_BEAMS, NLLB_CACHE_DIR,
    NLLB_USE_8BIT
)

logger = logging.getLogger(__name__)


class NLLBTranslator:
    """NLLB Machine Translation"""
    
    def __init__(
        self,
        model_name: str = NLLB_MODEL,
        src_lang: str = NLLB_SRC_LANG,
        tgt_lang: str = NLLB_TGT_LANG,
        device: str = NLLB_DEVICE,
        cache_dir: Optional[str] = NLLB_CACHE_DIR,
    ):
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = device
        self.cache_dir = cache_dir
        
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    def load_model(self):
        """
        Load NLLB model and tokenizer
        
        Supports 8-bit quantization for large models (3.3B) to reduce VRAM.
        Requires: pip install bitsandbytes
        """
        if self.is_loaded:
            return
        
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from src.utils import suppress_stdout
        
        logger.info(f"Loading NLLB: {self.model_name} (8-bit: {NLLB_USE_8BIT})")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            src_lang=self.src_lang
        )
        
        # Suppress noisy print() and logging from accelerate/transformers during loading
        with suppress_stdout():
            # Model loading with optional 8-bit quantization
            if NLLB_USE_8BIT:
                try:
                    from transformers import BitsAndBytesConfig
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                    )
                    
                    # 8-bit needs device_map="auto" for dispatch
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name,
                        cache_dir=self.cache_dir,
                        quantization_config=quantization_config,
                        device_map="auto",
                    )
                    logger.info("NLLB loaded with 8-bit quantization")
                except ImportError:
                    logger.warning("bitsandbytes not installed, falling back to float16")
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name,
                        cache_dir=self.cache_dir,
                        torch_dtype=torch.float16,
                    ).to(self.device)
            else:
                # distilled-600M fits easily on single GPU - no need for device_map="auto"
                # This eliminates the "following layers were not sharded" warning
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16,
                ).to(self.device)
        
        self.model.eval()
        self.is_loaded = True
        
        logger.info("NLLB ready")
    
    def translate(self, text: str, max_length: int = NLLB_MAX_LENGTH) -> str:
        """
        Translate Vietnamese text to English
        
        Args:
            text: Vietnamese text
            max_length: Max output length
            
        Returns:
            English translation
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
                max_length=512
            )
            
            # Move to model device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            tgt_lang_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=max_length,
                    num_beams=NLLB_NUM_BEAMS,
                    early_stopping=True,
                    do_sample=False
                )
            
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated.strip()
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return ""
