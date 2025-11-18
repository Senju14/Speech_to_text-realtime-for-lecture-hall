import numpy as np
from scipy.signal import butter, lfilter

def highpass_filter(data, cutoff=80, fs=16000, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)

def normalize_audio(audio, target_dbfs=-20.0):
    # audio expected float32 -1..1
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    rms = (audio**2).mean() ** 0.5
    if rms < 1e-6:
        return audio
    current_db = 20.0 * np.log10(rms + 1e-12)
    change_db = target_dbfs - current_db
    factor = 10.0 ** (change_db / 20.0)
    out = audio * factor
    out = np.clip(out, -1.0, 1.0)
    return out
