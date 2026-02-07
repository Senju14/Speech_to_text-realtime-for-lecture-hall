"""
PyTorch 2.6+ compatibility patch

PyTorch 2.6 changed torch.load default to weights_only=True which breaks
pyannote model loading (used by WhisperX for VAD).
This patch forces weights_only=False globally.
"""

_patched = False


def apply_torch_load_patch():
    """Apply monkey-patch to torch.load for pyannote compatibility"""
    global _patched
    
    if _patched:
        return
    
    import torch
    
    _original_load = torch.load
    
    def _patched_load(*args, **kwargs):
        # Force weights_only=False for pyannote compatibility
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    
    torch.load = _patched_load
    _patched = True
    
    print("[Patch] Applied torch.load weights_only=False patch")
