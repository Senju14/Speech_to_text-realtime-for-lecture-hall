"""
PyTorch optimizations & compatibility patches

1. TF32: Re-enable TensorFloat-32 for faster matmul on Ampere+ GPUs
2. torch.load: Force weights_only=False for pyannote compatibility
3. Warning filters: Suppress known noisy warnings from dependencies
4. suppress_stdout: Context manager to capture print()-based warnings during model loading
"""

import warnings
import logging
import sys
import io
import os
from contextlib import contextmanager

_patched = False


@contextmanager
def suppress_stdout():
    """
    Suppress stdout AND noisy logging during model loading.
    
    Many libraries (pyannote, pytorch_lightning, accelerate) use print()
    or logging.warning() to emit noise. This captures BOTH:
    - print() calls (via sys.stdout redirect)
    - logging.warning() from accelerate, transformers, pyannote
      (via temporarily raising log levels to CRITICAL)
    
    Usage:
        with suppress_stdout():
            model = load_some_noisy_model()
    """
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    # Temporarily raise logging to CRITICAL for noisy libraries
    _loggers_to_mute = [
        'accelerate', 'accelerate.utils', 'accelerate.utils.modeling',
        'accelerate.big_modeling', 'transformers', 'transformers.modeling_utils',
        'pyannote', 'pyannote.audio', 'pyannote.audio.core.model',
        'pytorch_lightning', 'lightning', 'lightning_fabric',
    ]
    _saved_levels = {}
    for name in _loggers_to_mute:
        _log = logging.getLogger(name)
        _saved_levels[name] = _log.level
        _log.setLevel(logging.CRITICAL)
    
    try:
        yield
    finally:
        sys.stdout = old_stdout
        for name, level in _saved_levels.items():
            logging.getLogger(name).setLevel(level)


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
    # accelerate dispatch/sharding messages
    warnings.filterwarnings("ignore", message=".*layers were not sharded.*")
    warnings.filterwarnings("ignore", message=".*not sharded.*")
    # pytorch_lightning upgrade messages
    warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")
    warnings.filterwarnings("ignore", message=".*lightning.*upgraded.*")
    
    # =========================================================================
    # 4. Suppress noisy logging from dependencies
    #    Many libraries use logging.warning() instead of warnings.warn()
    # =========================================================================
    for logger_name in [
        "transformers.modeling_utils",
        "transformers.utils.hub",
        "transformers.generation.utils",
        "pyannote.audio",
        "pyannote.audio.core.model",
        "pyannote.core",
        "pytorch_lightning",
        "pytorch_lightning.utilities.migration",
        "lightning_fabric",
        "lightning",
        "speechbrain",
        "torchaudio",
        "whisperx",
        "accelerate",
        "accelerate.utils",
        "accelerate.big_modeling",
        "ctranslate2",
        "faster_whisper",
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # Also suppress HF transfer/download progress spam
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    
    _patched = True
