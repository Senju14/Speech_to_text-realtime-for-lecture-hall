import torch

# === SERVER ===
HOST = "0.0.0.0"
PORT = 8000

# === MODAL CLOUD ===
USE_MODAL = True

MODAL_URL = "wss://ricky13170--asr-whisper-streaming-web-app.modal.run"
WS_URL = f"{MODAL_URL}/ws" if USE_MODAL else f"ws://127.0.0.1:{PORT}/ws"


# === AUDIO ===
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
MAX_BUFFER_SEC = 10
MAX_BUFFER_SAMPLES = MAX_BUFFER_SEC * SAMPLE_RATE

# === MODEL ===
MODEL_ID = "vinai/PhoWhisper-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === VAD ===
VAD_THRESHOLD = 0.6
SILENCE_LIMIT = 0.4
PARTIAL_INTERVAL = 0.25
PARTIAL_MIN_VOTES = 3

# === UI ===
WINDOW_NAME = "ASR Live Caption"
FONT_PATH = "arial.ttf"
USE_CAPTION_OVERLAY = False

# === SLIDING WINDOW PARAMETERS ===
MAX_DECODE_SEC = 3.0        # Whisper ổn nhất
OVERLAP_SEC = 0.5
MIN_DECODE_SEC = 0.8

VAD_ON = 0.02
VAD_OFF = 0.01

ON_DEBOUNCE_FRAMES = 3
OFF_DEBOUNCE_FRAMES = 5

MAX_SILENCE = 0.6   # giữ nguyên


# === CHROME AUDIO BRIDGE ===
CHROME_BRIDGE_PORT = 8765
CHROME_CONTROL_PORT = 8766

# URLs
CHROME_BRIDGE_URL = f"ws://localhost:{CHROME_BRIDGE_PORT}"
CONTROL_URL = f"http://localhost:{CHROME_CONTROL_PORT}"

# === SOURCE SELECTION ===
AUDIO_SOURCE = "microphone"  # "microphone" | "chrome" | "file"
