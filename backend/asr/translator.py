import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time

class NLLBTranslator:
    """Vietnamese to English translator using NLLB (Optimized for Streaming)"""
    
    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
        # NLLB language codes
        self.src_lang = "vie_Latn"  # Vietnamese
        self.tgt_lang = "eng_Latn"  # English
    
    def load(self):
        """Load NLLB model and tokenizer"""
        if self._loaded:
            return
        
        print(f"[Translator] Loading {self.model_name}...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 👇 Tối ưu 1: Dùng float16 nếu chạy trên GPU để tăng tốc
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                src_lang=self.src_lang,
                cache_dir="/cache/huggingface" if torch.cuda.is_available() else None
            )
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                dtype=dtype, # Load nhẹ hơn
                low_cpu_mem_usage=True,
                cache_dir="/cache/huggingface" if torch.cuda.is_available() else None
            )
            
            if device == "cuda":
                self.model = self.model.to(device)
                print(f"[Translator] Using GPU: {torch.cuda.get_device_name(0)}")
            
            self.model.eval()
            self._loaded = True
            print(f"[Translator] Ready!")
            
        except Exception as e:
            print(f"[Translator] Error loading model: {e}")
            raise e
    
    def translate(self, text: str, max_length: int = 128) -> str:
        """
        Translate Vietnamese text to English
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if not text or not text.strip():
            return ""
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
                max_length=max_length,
                # Tối ưu 2: Giảm num_beams để Streaming nhanh hơn
                # num_beams=1 (Greedy Search) -> Nhanh nhất
                # num_beams=2 -> Cân bằng
                num_beams=1, 
                do_sample=False, # Đảm bảo kết quả ổn định, không ngẫu nhiên
                early_stopping=True
            )
        
        # Decode
        translation = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
        
        return translation.strip()
