"""
ASR Session Handler

Manages WebSocket sessions for real-time speech recognition.
Each client connection gets its own ASRSession instance.

Optimizations:
- Binary audio input: Receives raw Int16 bytes (33% smaller than Base64)
- Async streaming: Sends ASR result immediately, translation follows async
- Context priming: Groq expands topic keywords -> WhisperX initial_prompt
- Auto summary: Groq summarizes lecture on session end
- Multi-language: Per-session source/target language with dynamic switching
"""

import json
import asyncio
import numpy as np
import time
import re
import logging
from typing import Optional

from src.config import (
    WHISPER_MODEL, WHISPER_LANGUAGE, ASR_DEVICE,
    NLLB_MODEL, NLLB_SRC_LANG, NLLB_TGT_LANG, NLLB_DEVICE, NLLB_CACHE_DIR,
    MIN_SILENCE_DURATION, MAX_BUFFER_DURATION, MIN_SEGMENT_DURATION,
    SAMPLE_RATE, AUTO_SUMMARY_MIN_DURATION,
    iso_to_nllb,
)
from src.asr import WhisperXASR
from src.vad import SileroVAD
from src.translation import NLLBTranslator
from src.postprocess import BARTphoCorrector
from src.llm import GroqService
from src.utils import decode_audio_chunk, decode_audio_bytes
from src.session.filters import HallucinationFilter

logger = logging.getLogger(__name__)


class ASRService:
    """
    ASR Service - manages shared model instances
    
    Created once per container, shared across all sessions.
    """
    
    def __init__(self):
        self.asr: Optional[WhisperXASR] = None
        self.corrector: Optional[BARTphoCorrector] = None
        self.translator: Optional[NLLBTranslator] = None
        self.groq: Optional[GroqService] = None
        self.is_initialized = False
    
    async def init(self):
        """Initialize all models"""
        loop = asyncio.get_event_loop()
        
        # Load WhisperX
        t0 = time.time()
        logger.info("[Model] Loading WhisperX...")
        self.asr = WhisperXASR(
            model_size=WHISPER_MODEL,
            language=WHISPER_LANGUAGE,
            device=ASR_DEVICE,
        )
        await loop.run_in_executor(None, self.asr.load_model)
        logger.info(f"[Model] WhisperX loaded in {time.time() - t0:.1f}s")
        
        # Load BARTpho Corrector
        from src.config import BARTPHO_ADAPTER, BARTPHO_DEVICE, ENABLE_BARTPHO
        if ENABLE_BARTPHO:
            t0 = time.time()
            logger.info("[Model] Loading BARTpho Corrector...")
            self.corrector = BARTphoCorrector(
                adapter_id=BARTPHO_ADAPTER,
                device=BARTPHO_DEVICE,
                cache_dir=NLLB_CACHE_DIR,
            )
            await loop.run_in_executor(None, self.corrector.load_model)
            logger.info(f"[Model] BARTpho loaded in {time.time() - t0:.1f}s")
        else:
            logger.info("[Model] BARTpho disabled (ENABLE_BARTPHO=False)")
        
        # Load NLLB
        t0 = time.time()
        logger.info("[Model] Loading NLLB Translator...")
        self.translator = NLLBTranslator(
            model_name=NLLB_MODEL,
            src_lang=NLLB_SRC_LANG,
            tgt_lang=NLLB_TGT_LANG,
            device=NLLB_DEVICE,
            cache_dir=NLLB_CACHE_DIR,
        )
        await loop.run_in_executor(None, self.translator.load_model)
        logger.info(f"[Model] NLLB loaded in {time.time() - t0:.1f}s")
        
        # Init Groq LLM (non-blocking, no model to load)
        logger.info("[Model] Initializing Groq LLM...")
        self.groq = GroqService()
        self.groq.init()
        
        self.is_initialized = True
        logger.info("[Model] All models ready")
    
    def create_session(self) -> "ASRSession":
        """Create a new session for a WebSocket connection"""
        return ASRSession(self)


class ASRSession:
    """
    ASR Session - handles one WebSocket connection
    
    Manages audio buffering, VAD, transcription, and translation
    for a single client.
    """
    
    def __init__(self, service: ASRService):
        self.service = service
        self.out_queue = asyncio.Queue()
        
        # State
        self.is_recording = False
        self.lock = asyncio.Lock()
        self.segment_id = 0
        
        # Audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_speech_time = 0.0
        self.session_start_time = 0.0
        
        # Config
        self.src_lang = "vi"
        self.tgt_lang = "en"
        self.do_translate = True
        
        # Context priming (Groq)
        self.topic = ""
        self.initial_prompt = ""  # Keywords for WhisperX
        self.all_transcripts = []  # Collect transcripts for summary
        
        # Components
        self.vad = SileroVAD()
        self.filter = HallucinationFilter()
    
    async def _send_log(self, message: str, level: str = "info"):
        """Send structured log message to client (for toast notifications)"""
        await self.out_queue.put(json.dumps({
            "type": "log",
            "level": level,
            "message": message,
        }))

    @staticmethod
    def _sanitize_topic(topic: str) -> str:
        """Sanitize user-provided topic input"""
        if not topic:
            return ""
        # Strip control characters, limit length
        topic = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', topic)
        return topic.strip()[:200]

    async def handle_incoming(self, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "start":
                await self._handle_start(data)
            elif msg_type == "audio":
                await self._handle_audio(data)
            elif msg_type == "stop":
                await self._handle_stop()
            elif msg_type == "summarize":
                await self._handle_summarize()
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_start(self, data: dict):
        """Start recording session"""
        # Accept both camelCase (frontend) and snake_case keys
        self.src_lang = data.get("srcLang") or data.get("src_lang") or "vi"
        self.tgt_lang = data.get("tgtLang") or data.get("tgt_lang") or "en"
        self.do_translate = data.get("translate", True)
        self.topic = self._sanitize_topic(data.get("topic", ""))
        
        self.is_recording = True
        self.segment_id = 0
        self.audio_buffer = np.array([], dtype=np.float32)
        self.session_start_time = time.time()
        self.last_speech_time = time.time()
        self.all_transcripts = []
        self.initial_prompt = ""
        self._detected_lang = None  # For auto-detect mode
        
        self.vad.load_model()
        self.vad.reset_state()
        
        # === Context Priming ===
        # 1. Default vocab primer for Vietnamese (helps Whisper with code-switching)
        base_prompt = ""
        if self.src_lang in ("vi", "auto"):
            base_prompt = (
                "AI, ML, deep learning, machine learning, NLP, ChatGPT, GPT, "
                "Gemini, OpenAI, Google, transformer, neural network, LLM, "
                "computer vision, robotics, Python, TensorFlow, PyTorch, "
                "dataset, token, model, fine-tuning, pre-training, embedding, "
                "attention, inference, GPU, API, framework, server, deploy"
            )
        
        # 2. Expand topic into keywords via Groq (if topic provided)
        if self.topic and self.service.groq and self.service.groq.is_available:
            try:
                await self._send_log("Generating keywords from topic...", "info")
                keywords = await self.service.groq.expand_keywords(
                    self.topic, language=self.src_lang
                )
                if keywords:
                    self.initial_prompt = (base_prompt + ", " + keywords) if base_prompt else keywords
                    logger.info(f"[Context] Primed with: {self.initial_prompt[:80]}...")
                else:
                    self.initial_prompt = base_prompt
            except Exception as e:
                logger.error(f"[Context] Keyword expansion failed: {e}")
                self.initial_prompt = base_prompt
                await self._send_log("Keyword generation failed, continuing without", "warning")
        else:
            self.initial_prompt = base_prompt
        
        logger.info(f"[Session] Started ({self.src_lang} → {self.tgt_lang})"
              f"{' | Topic: ' + self.topic[:40] if self.topic else ''}")
        
        await self.out_queue.put(json.dumps({
            "type": "status",
            "status": "started",
            "topic": self.topic,
            "primed": bool(self.initial_prompt),
        }))
    
    async def _handle_audio(self, data: dict):
        """Process incoming audio chunk (legacy Base64 path)"""
        if not self.is_recording:
            return
        
        async with self.lock:
            # Decode audio from base64
            audio_b64 = data.get("audio", "")
            audio_chunk = decode_audio_chunk(audio_b64)
            await self._process_audio_chunk(audio_chunk)
    
    async def handle_binary_audio(self, audio_bytes: bytes):
        """Process incoming binary audio chunk (optimized path)"""
        if not self.is_recording:
            return
        
        async with self.lock:
            # Decode raw Int16 bytes to Float32
            audio_chunk = decode_audio_bytes(audio_bytes)
            await self._process_audio_chunk(audio_chunk)
    
    async def _process_audio_chunk(self, audio_chunk: np.ndarray):
        """Common audio processing logic"""
        if len(audio_chunk) == 0:
            return
        
        # Append to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        # Check VAD
        has_speech = self.vad.is_speech(audio_chunk)
        current_time = time.time()
        
        if has_speech:
            self.last_speech_time = current_time
        
        # Calculate durations
        buffer_duration = len(self.audio_buffer) / SAMPLE_RATE
        silence_duration = current_time - self.last_speech_time
        
        # Send buffering indicator every ~2s so user sees activity
        if has_speech and buffer_duration > 1.0 and int(buffer_duration) % 2 == 0:
            # Send a lightweight "listening" indicator
            elapsed = buffer_duration
            if not hasattr(self, '_last_buffer_notify') or current_time - self._last_buffer_notify > 1.5:
                self._last_buffer_notify = current_time
                await self.out_queue.put(json.dumps({
                    "type": "transcript",
                    "segment_id": self.segment_id + 1,
                    "source": "...",
                    "target": "",
                    "is_final": False,
                }))
        
        # Check finalize conditions
        should_finalize = False
        reason = ""
        
        if buffer_duration >= MAX_BUFFER_DURATION:
            should_finalize = True
            reason = f"max {buffer_duration:.1f}s"
        elif silence_duration >= MIN_SILENCE_DURATION and buffer_duration >= MIN_SEGMENT_DURATION:
            should_finalize = True
            reason = f"silence {silence_duration:.1f}s"
        
        if should_finalize:
            await self._finalize_segment(reason)
    
    async def _finalize_segment(self, reason: str):
        """
        Finalize and transcribe current audio buffer
        
        Async Streaming Strategy:
        1. Run ASR (WhisperX) with initial_prompt vocab priming
        2. Post-process with BARTpho (Vietnamese syllable correction)
        3. Code-switch normalize (phonetic Vietnamese → English terms)
        4. Send intermediate result immediately (source text only)
        5. Fire translation async (non-blocking)
        6. Send final result when translation completes
        
        This reduces perceived latency - user sees source text immediately.
        """
        min_samples = int(MIN_SEGMENT_DURATION * SAMPLE_RATE)
        
        if len(self.audio_buffer) < min_samples:
            self.audio_buffer = np.array([], dtype=np.float32)
            return
        
        # Copy and clear buffer
        audio = self.audio_buffer.copy()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_speech_time = time.time()
        
        logger.info(f"[VAD] Finalize ({reason})")
        
        loop = asyncio.get_event_loop()
        
        # Transcribe with WhisperX (with context priming if available)
        # Pass session language to WhisperX (None for auto-detect)
        whisper_lang = self.src_lang if self.src_lang != "auto" else None
        start_time = time.time()
        result = await loop.run_in_executor(
            None,
            lambda: self.service.asr.transcribe_segment(
                audio,
                initial_prompt=self.initial_prompt,
                language=whisper_lang,
            )
        )
        asr_time = time.time() - start_time
        
        text = result.get("text", "").strip()
        words = result.get("words", [])
        detected_lang = result.get("language", self.src_lang)
        
        # Update src_lang if auto-detected (for downstream translation)
        if self.src_lang == "auto" and detected_lang:
            self._detected_lang = detected_lang
        
        if not text:
            return
        
        # Filter hallucinations
        if self.filter.is_hallucination(text):
            logger.debug(f"[Filter] Skipped: {text[:50]}...")
            return
        
        # Post-process: BARTpho syllable correction (Vietnamese only)
        effective_src = detected_lang if self.src_lang == "auto" else self.src_lang
        if (effective_src == "vi"
            and self.service.corrector
            and self.service.corrector.is_loaded):
            pp_start = time.time()
            corrected = await loop.run_in_executor(
                None, self.service.corrector.correct, text
            )
            pp_time = time.time() - pp_start
            if corrected and corrected != text:
                logger.info(f"[PP] \"{text[:40]}\" → \"{corrected[:40]}\" ({pp_time:.2f}s)")
                text = corrected
            else:
                pp_time = 0.0
        else:
            pp_time = 0.0
        
        self.segment_id += 1
        current_seg_id = self.segment_id
        
        # Collect transcript for auto-summary
        self.all_transcripts.append(text)
        
        logger.info(f"[ASR] #{current_seg_id}: {text[:60]}...")
        if words:
            logger.debug(f"[Words] {len(words)} words aligned")
        
        # === ASYNC STREAMING: Send ASR result immediately ===
        if self.do_translate and self.service.translator:
            # Send intermediate result (source only, is_final=False)
            await self.out_queue.put(json.dumps({
                "type": "transcript",
                "segment_id": current_seg_id,
                "source": text,
                "target": "",  # Translation pending
                "is_final": False,
                "words": words,
                "timing": {
                    "asr_ms": int(asr_time * 1000),
                    "pp_ms": int(pp_time * 1000),
                    "mt_ms": 0,
                }
            }))
            
            # Fire translation async (non-blocking)
            asyncio.create_task(
                self._translate_and_send(current_seg_id, text, words, asr_time, pp_time)
            )
        else:
            # No translation - send final result directly
            await self.out_queue.put(json.dumps({
                "type": "transcript",
                "segment_id": current_seg_id,
                "source": text,
                "target": "",
                "is_final": True,
                "words": words,
                "timing": {
                    "asr_ms": int(asr_time * 1000),
                    "pp_ms": int(pp_time * 1000),
                    "mt_ms": 0,
                }
            }))
    
    async def _translate_and_send(self, segment_id: int, text: str, words: list, asr_time: float, pp_time: float):
        """
        Async translation task - runs after ASR result is already sent.
        Sends final update with translation when complete.
        """
        loop = asyncio.get_event_loop()
        
        try:
            # Resolve NLLB language codes for this session
            effective_src = getattr(self, '_detected_lang', self.src_lang)
            if self.src_lang == "auto" and effective_src:
                nllb_src = iso_to_nllb(effective_src)
            else:
                nllb_src = iso_to_nllb(self.src_lang)
            nllb_tgt = iso_to_nllb(self.tgt_lang)
            
            start_time = time.time()
            translation = await loop.run_in_executor(
                None,
                lambda: self.service.translator.translate(
                    text, src_lang=nllb_src, tgt_lang=nllb_tgt
                )
            )
            mt_time = time.time() - start_time
            
            logger.info(f"[MT] #{segment_id} ({mt_time:.1f}s): {translation[:50]}...")
            
            # Send final update with translation
            await self.out_queue.put(json.dumps({
                "type": "transcript",
                "segment_id": segment_id,
                "source": text,
                "target": translation,
                "is_final": True,
                "words": words,
                "timing": {
                    "asr_ms": int(asr_time * 1000),
                    "pp_ms": int(pp_time * 1000),
                    "mt_ms": int(mt_time * 1000),
                }
            }))
        except Exception as e:
            logger.error(f"Translation error for segment {segment_id}: {e}")
            # Send final without translation on error
            await self.out_queue.put(json.dumps({
                "type": "transcript",
                "segment_id": segment_id,
                "source": text,
                "target": "",
                "is_final": True,
                "words": words,
                "timing": {
                    "asr_ms": int(asr_time * 1000),
                    "pp_ms": int(pp_time * 1000),
                    "mt_ms": 0,
                }
            }))
    
    async def _handle_stop(self):
        """Stop recording session"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Process remaining buffer
        if len(self.audio_buffer) > int(MIN_SEGMENT_DURATION * SAMPLE_RATE):
            await self._finalize_segment("stop")
        
        session_duration = time.time() - self.session_start_time
        
        logger.info(f"[Session] Stopped | Segments: {self.segment_id} | Duration: {session_duration:.0f}s")
        
        await self.out_queue.put(json.dumps({
            "type": "status",
            "status": "stopped",
            "segments": self.segment_id,
        }))
        
        # === Auto Summary via Groq (if session > 2 minutes) ===
        if (session_duration >= AUTO_SUMMARY_MIN_DURATION 
            and self.all_transcripts 
            and self.service.groq 
            and self.service.groq.is_available):
            await self._send_log("Generating lecture summary...", "info")
            asyncio.create_task(self._generate_summary())
    
    async def _handle_summarize(self):
        """Handle manual summarize request from client"""
        if not self.all_transcripts:
            await self._send_log("No transcripts to summarize", "warning")
            return
        
        if not self.service.groq or not self.service.groq.is_available:
            await self._send_log("Summary service not available", "error")
            return
        
        await self._send_log("Generating summary...", "info")
        await self._generate_summary()
    
    async def cleanup(self):
        """Cleanup session resources"""
        self.is_recording = False
        self.audio_buffer = np.array([], dtype=np.float32)
        self.all_transcripts = []
    
    async def _generate_summary(self):
        """Generate lecture summary via Groq (async, non-blocking)"""
        try:
            full_transcript = "\n".join(self.all_transcripts)
            summary = await self.service.groq.summarize_lecture(
                full_transcript, topic=self.topic
            )
            
            if summary:
                await self.out_queue.put(json.dumps({
                    "type": "summary",
                    "summary": summary,
                    "topic": self.topic,
                    "segments_count": self.segment_id,
                }))
                logger.info(f"[Summary] Generated ({len(summary)} chars)")
        except Exception as e:
            logger.error(f"[Summary] Generation failed: {e}")
