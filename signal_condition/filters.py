import numpy as np
from scipy.signal import butter, lfilter
import config

def highpass_filter(x, cutoff=80):
    nyq = 0.5 * config.SAMPLE_RATE
    b, a = butter(3, cutoff / nyq, btype="high")
    return lfilter(b, a, x)

def normalize_audio(x, target_db=-20.0):
    x = x.astype(np.float32)
    rms = np.sqrt(np.mean(x ** 2))
    if rms < 1e-6:
        return x
    gain = 10 ** ((target_db - 20 * np.log10(rms)) / 20)
    return np.clip(x * gain, -1.0, 1.0)
