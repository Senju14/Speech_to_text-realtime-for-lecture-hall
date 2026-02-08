"""
ASR Thesis - Vietnamese Speech Recognition with Translation

Modal App using WhisperX for ASR and NLLB for translation.

Usage:
    modal deploy main.py     # Deploy to Modal cloud
    modal serve main.py      # Local development with hot-reload
"""

from modal import App, Image, asgi_app, Volume, enter, Secret

# =============================================================================
# Modal Infrastructure Configuration
# NOTE: These must be defined here (not in src/config) because Modal needs
# them at deploy time before the container starts. Runtime config is in
# src/config/settings.py
# =============================================================================

MODAL_APP_NAME = "asr-thesis"
MODAL_GPU = "A100" 
MODAL_MEMORY = 24576
MODAL_TIMEOUT = 600
MODAL_CONTAINER_IDLE_TIMEOUT = 120

# Also defined in src/config/settings.py for runtime use
WHISPER_MODEL = "large-v3"

# =============================================================================
# Modal App
# =============================================================================

app = App(MODAL_APP_NAME)

# Docker image with all dependencies
image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "sentencepiece>=0.1.99",
    )
    # WhisperX (includes faster-whisper, pyannote)
    .pip_install("whisperx")
    .pip_install(
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "websockets>=12.0",
        "aiofiles>=23.0.0",
    )
    .pip_install(
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "soundfile>=0.12.0",
    )
    .pip_install("safetensors>=0.4.0", "protobuf>=3.20.0")
    .pip_install("groq>=0.4.0")
    # Copy source code
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .add_local_dir("frontend", remote_path="/root/frontend", copy=True)
)


def download_models():
    """Pre-download models during image build for faster cold starts"""
    import sys
    import os
    sys.path.append("/root")
    os.environ["HF_HOME"] = "/cache/huggingface"
    
    # Apply PyTorch patch for pyannote compatibility
    from src.utils import apply_torch_load_patch
    apply_torch_load_patch()
    
    import whisperx
    
    # WhisperX + alignment
    print("[Pre-download] WhisperX large-v3...")
    model = whisperx.load_model("large-v3", device="cuda", compute_type="float16")
    del model
    
    print("[Pre-download] Alignment model (vi)...")
    align_model, _ = whisperx.load_align_model(language_code="vi", device="cuda")
    del align_model
    
    # NLLB Translation
    from src.translation import NLLBTranslator
    print("[Pre-download] NLLB-200 3.3B...")
    translator = NLLBTranslator(cache_dir="/cache/nllb")
    translator.load_model()
    del translator
    
    # Silero VAD
    from src.vad import SileroVAD
    print("[Pre-download] Silero VAD...")
    vad = SileroVAD()
    vad.load_model()
    del vad
    
    print("[Pre-download] Complete!")


# Build image with pre-downloaded models
image_with_models = image.run_function(
    download_models,
    gpu="A100",
    volumes={"/cache": Volume.from_name("asr-model-cache", create_if_missing=True)}
)

cache = Volume.from_name("asr-model-cache", create_if_missing=True)


# =============================================================================
# ASR Container Class
# =============================================================================

@app.cls(
    gpu=MODAL_GPU,
    memory=MODAL_MEMORY,
    timeout=MODAL_TIMEOUT,
    scaledown_window=MODAL_CONTAINER_IDLE_TIMEOUT,
    image=image_with_models,
    volumes={"/cache": cache},
    secrets=[Secret.from_name("groq-api-key")],
)
class ASR:
    """Main ASR container class"""
    
    @enter()
    def setup(self):
        """Initialize models when container starts"""
        import sys
        import os
        import time
        import asyncio
        
        sys.path.append("/root")
        os.environ["HF_HOME"] = "/cache/huggingface"
        
        # Apply PyTorch patch
        from src.utils import apply_torch_load_patch
        apply_torch_load_patch()
        
        print("[Container] Initializing...")
        start = time.time()
        
        # Initialize ASR service
        from src.session import ASRService
        self.service = ASRService()
        asyncio.run(self.service.init())
        
        print(f"[Container] Ready in {time.time() - start:.1f}s | WhisperX: {WHISPER_MODEL} | GPU: {MODAL_GPU}")

    @asgi_app()
    def app(self):
        """Create FastAPI application"""
        from fastapi import FastAPI, WebSocket
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.staticfiles import StaticFiles
        from starlette.middleware.base import BaseHTTPMiddleware
        import time
        
        from src.api import create_api_routes, handle_websocket

        web_app = FastAPI(title="ASR Thesis", version="2.0")
        
        # No-cache middleware for development
        class NoCacheMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                response = await call_next(request)
                if request.url.path.endswith(('.js', '.css', '.html')):
                    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                    response.headers["Pragma"] = "no-cache"
                    response.headers["Expires"] = "0"
                return response
        
        web_app.add_middleware(NoCacheMiddleware)
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        start_time = time.time()
        conn_count = [0]

        # Add API routes
        create_api_routes(web_app, start_time, gpu=MODAL_GPU, service=self.service)

        # WebSocket endpoint
        @web_app.websocket("/ws/transcribe")
        async def websocket_endpoint(websocket: WebSocket):
            conn_count[0] += 1
            await handle_websocket(websocket, self.service, conn_count[0])

        # Static files (frontend)
        web_app.mount("/", StaticFiles(directory="/root/frontend", html=True), name="frontend")
        
        return web_app