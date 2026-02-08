"""
PyTorch optimizations & compatibility patches

1. TF32: Re-enable TensorFloat-32 for faster matmul on Ampere+ GPUs
2. torch.load: Force weights_only=False for pyannote compatibility
3. Warning filters: Suppress known noisy warnings from dependencies
"""

import warnings
import logging

_patched = False


def apply_torch_load_patch():
    """Apply all PyTorch patches and optimizations"""
    global _patched
    
    if _patched:
        return
    
    import torch
    
    # =========================================================================
    # 1. Enable TF32 for ~3x faster matmul on Ampere+ GPUs (A100, RTX 30xx+)
    #    PyTorch may disable this by default with a ReproducibilityWarning
    # =========================================================================
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("[Patch] TF32 enabled for CUDA matmul and cuDNN")
    
    # =========================================================================
    # 2. Patch torch.load for pyannote compatibility (PyTorch 2.6+)
    # =========================================================================
    _original_load = torch.load
    
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    
    torch.load = _patched_load
    
    # =========================================================================
    # 3. Filter known noisy warnings (warnings.warn)
    # =========================================================================
    # pyannote version mismatch warnings
    warnings.filterwarnings("ignore", message=".*Model was trained with pyannote.*")
    warnings.filterwarnings("ignore", message=".*Model was trained with torch.*")
    # wav2vec2 / Whisper weight initialization warnings
    warnings.filterwarnings("ignore", message=".*Some weights of.*were not initialized.*")
    warnings.filterwarnings("ignore", message=".*Some weights of.*not used.*")
    # Deprecated torch_dtype parameter
    warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
    # TF32 reproducibility warning
    warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
    # General FutureWarnings from transformers
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    # torchaudio deprecation warnings
    warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*torchaudio._backend.*")
    # ctranslate2 / faster-whisper
    warnings.filterwarnings("ignore", message=".*ctranslate2.*")
    
    # =========================================================================
    # 4. Suppress noisy logging from dependencies
    #    Many libraries use logging.warning() instead of warnings.warn()
    # =========================================================================
    for logger_name in [
        "transformers.modeling_utils",
        "transformers.utils.hub",
        "pyannote.audio",
        "pyannote.core",
        "pytorch_lightning",
        "lightning_fabric",
        "speechbrain",
        "torchaudio",
        "whisperx",
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    _patched = True
    print("[Patch] Applied torch.load patch + warning filters")
