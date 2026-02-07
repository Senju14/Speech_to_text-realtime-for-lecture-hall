"""
Audio processing utilities
"""

import base64
import numpy as np


def decode_audio_chunk(audio_b64: str) -> np.ndarray:
    """
    Decode base64 audio chunk from frontend (legacy path)
    
    Frontend sends Int16 PCM at 16kHz.
    Returns float32 audio normalized to [-1, 1].
    
    Args:
        audio_b64: Base64 encoded Int16 audio bytes
        
    Returns:
        Float32 numpy array
    """
    if not audio_b64:
        return np.array([], dtype=np.float32)
    
    audio_bytes = base64.b64decode(audio_b64)
    return decode_audio_bytes(audio_bytes)


def decode_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """
    Decode raw binary audio bytes from frontend (optimized path)
    
    Frontend sends Int16 PCM at 16kHz as raw ArrayBuffer.
    Returns float32 audio normalized to [-1, 1].
    
    This path is ~33% more efficient than Base64 encoding.
    
    Args:
        audio_bytes: Raw Int16 audio bytes
        
    Returns:
        Float32 numpy array
    """
    if not audio_bytes:
        return np.array([], dtype=np.float32)
    
    # Ensure even number of bytes for Int16
    if len(audio_bytes) % 2 != 0:
        audio_bytes = audio_bytes[:-1]
    
    if len(audio_bytes) == 0:
        return np.array([], dtype=np.float32)
    
    # Convert Int16 to Float32 normalized
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    
    return audio_float32
