# backend/config.py

# Audio Settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 500 # 0.5s chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# VAD Settings (Adaptive Energy Based)
# Lưu ý: Code VADManager mới tự động thích ứng, nhưng ta vẫn giữ threshold cơ sở ở đây
VAD_ENABLED = True
VAD_THRESHOLD = 0.015  # Ngưỡng RMS (Thấp vì là năng lượng) - Cũ là 0.5 (Sai)
VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 500

# Audio Preprocessing
NOISE_REDUCE_ENABLED = False   # 👇 QUAN TRỌNG: Tắt đi để giảm độ trễ (Frontend đã lo rồi)
NOISE_REDUCE_PROP_DECREASE = 0.5
HIGHPASS_ENABLED = True        # Lọc tiếng ù (80Hz)
HIGHPASS_CUTOFF_HZ = 80
NORMALIZE_ENABLED = True       # Chống vỡ tiếng (Soft Clipping)

# Local Agreement (Chống giật)
LOCAL_AGREEMENT_N = 2      # Cần 2 lần giống nhau mới chốt
BUFFER_TRIMMING_SEC = 15   # Giới hạn buffer tổng
MIN_CHUNK_SIZE_SEC = 1.0

# Whisper E2E Model (Transcription + Translation)
WHISPER_MODEL = "openai/whisper-large-v3"  # E2E: supports both transcribe and translate
WHISPER_DEVICE = "cuda"
WHISPER_LANGUAGE = "vi"

# Server
WS_HOST = "0.0.0.0"
WS_PORT = 8000

# Modal Cloud Config
MODAL_APP_NAME = "asr-thesis"
MODAL_GPU = "A10G"      # GPU mạnh, VRAM 24GB
MODAL_MEMORY = 16384    # 16GB RAM
MODAL_TIMEOUT = 600     # 10 phút timeout cho request
MODAL_CONTAINER_IDLE_TIMEOUT = 120 # 2 phút không dùng thì tắt container cho đỡ tốn tiền

# Logging
LOG_LEVEL = "INFO"
LOG_TRANSCRIPTIONS = True
LOG_TIMING = True


SILENCE_LIMIT = 0.6
MAX_SEGMENT_SEC = 6.0
OVERLAP_SEC = 0.4
MIN_DECODE_SEC = 1.2