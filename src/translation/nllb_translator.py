"""
NLLB Translation - No Language Left Behind

Uses facebook/nllb-200-3.3B for Vietnamese to English translation.

Reference: https://huggingface.co/facebook/nllb-200-3.3B
"""

import logging
from typing import Optional
import torch

from src.config import (
    NLLB_MODEL, NLLB_SRC_LANG, NLLB_TGT_LANG,
    NLLB_DEVICE, NLLB_MAX_LENGTH, NLLB_NUM_BEAMS, NLLB_CACHE_DIR
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
        """Load NLLB model and tokenizer"""
        if self.is_loaded:
            return
        
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        logger.info(f"Loading NLLB: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            src_lang=self.src_lang
        )
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        self.model.eval()
        self.is_loaded = True
        
        logger.info("NLLB loaded")
    
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
