from utils.audio_utils import highpass_filter, normalize_audio
import noisereduce as nr

def process_realtime_chunk(x):
    x = highpass_filter(x)
    x = normalize_audio(x)
    return x

def process_final_sentence(x, sr=16000):
    x = process_realtime_chunk(x)
    x = nr.reduce_noise(y=x, sr=sr)
    return x
