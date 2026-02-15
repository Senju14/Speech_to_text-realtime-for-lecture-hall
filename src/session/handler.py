"""
ASR Session Handler (Merged v7)

Merges best of both implementations:
- Senju14/develop: Binary WS, multi-language, async streaming, word timestamps,
  timing breakdown, asyncio.Lock, NLLB language mapping
- Ricky13170/main: SpeechSegmentBuffer, LocalAgreement, multi-layer hallucination
  filter, intermediate decode, pending partial finalize, transcript-history prompt

Pipeline per session:
  Audio → VAD → SpeechSegmentBuffer → WhisperX → HallucinationFilter
        → LocalAgreement → BARTpho → AsyncStreaming → NLLB Translation
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
    SAMPLE_RATE, AUTO_SUMMARY_MIN_DURATION,
    MAX_SEGMENT_SEC, OVERLAP_SEC, SILENCE_LIMIT, MIN_DECODE_SEC,
    AGREEMENT_N, HALLUCINATION_HISTORY_SIZE,
    iso_to_nllb,
)
from src.asr import WhisperXASR
from src.vad import SileroVAD
from src.translation import NLLBTranslator
from src.postprocess import BARTphoCorrector
from src.llm import GroqService
from src.utils import decode_audio_chunk, decode_audio_bytes
from src.session.filters import HallucinationFilter
from src.session.speech_segment_buffer import SpeechSegmentBuffer
from src.session.local_agreement import LocalAgreement

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
    Per-WebSocket session with segment-based processing.

    Uses SpeechSegmentBuffer for overlap-based segmentation and
    LocalAgreement for partial text stabilization (reduces flicker).
    """

    PARTIAL_FINALIZE_TIMEOUT = 3.0  # seconds before auto-finalizing partials

    def __init__(self, service: ASRService):
        self.service = service
        self.out_queue = asyncio.Queue()

        # State
        self.is_recording = False
        self.lock = asyncio.Lock()
        self.segment_id = 0
        self.session_start_time = 0.0

        # Ricky: Segment-based processing state
        self.last_stable = ""
        self.pending_partial_text = None
        self.pending_partial_time = None
        self.transcript_history = []
        self._last_intermediate_decode = 0.0

        # Language config (per-session, set by client)
        self.src_lang = "vi"
        self.tgt_lang = "en"
        self.do_translate = True

        # Context priming
        self.topic = ""
        self.initial_prompt = ""
        self.all_transcripts = []
        self._detected_lang = None

        # Components (Ricky's segment-based architecture)
        self.vad = SileroVAD()
        self.segmenter = SpeechSegmentBuffer(
            sample_rate=SAMPLE_RATE,
            max_sec=MAX_SEGMENT_SEC,
            overlap_sec=OVERLAP_SEC,
            silence_limit=SILENCE_LIMIT,
        )
        self.agreement = LocalAgreement(n=AGREEMENT_N)
        self.hallucination_filter = HallucinationFilter(
            history_size=HALLUCINATION_HISTORY_SIZE,
        )

    # ─── Helpers ──────────────────────────────────────────────────────

    async def _send_log(self, message: str, level: str = "info"):
        """Send structured log to client (toast notification)"""
        await self.out_queue.put(json.dumps({
            "type": "log", "level": level, "message": message,
        }))

    @staticmethod
    def _sanitize_topic(topic: str) -> str:
        if not topic:
            return ""
        topic = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', topic)
        return topic.strip()[:200]

    def _get_timestamp(self) -> str:
        """Session timestamp in MM:SS format"""
        if not self.session_start_time:
            return "00:00"
        elapsed = time.time() - self.session_start_time
        return f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

    def _build_prompt(self) -> str:
        """Build Whisper initial_prompt from keywords + recent history."""
        parts = []
        if self.initial_prompt:
            parts.append(self.initial_prompt)
        if self.transcript_history:
            recent = " ".join(self.transcript_history[-2:])
            if len(recent) > 150:
                recent = recent[-150:]
            parts.append(recent)
        return ", ".join(parts) if parts else ""

    # ─── Message Routing ──────────────────────────────────────────────

    async def handle_incoming(self, message: str):
        """Handle incoming WebSocket text message"""
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
            elif msg_type == "ping":
                await self.out_queue.put(json.dumps({"type": "pong"}))
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    # ─── Start ────────────────────────────────────────────────────────

    async def _handle_start(self, data: dict):
        """Start recording session with multi-language support"""
        # Language config from client
        self.src_lang = data.get("srcLang") or data.get("src_lang") or "vi"
        self.tgt_lang = data.get("tgtLang") or data.get("tgt_lang") or "en"
        self.do_translate = data.get("translate", True)
        self.topic = self._sanitize_topic(data.get("topic", ""))

        # Reset state
        self.is_recording = True
        self.segment_id = 0
        self.session_start_time = time.time()
        self.all_transcripts = []
        self.initial_prompt = ""
        self._detected_lang = None
        self.last_stable = ""
        self.pending_partial_text = None
        self.pending_partial_time = None
        self.transcript_history = []
        self._last_intermediate_decode = 0.0

        # Reset all components
        self.vad.load_model()
        self.vad.reset_state()
        self.segmenter.reset()
        self.agreement.reset()
        self.hallucination_filter.reset()

        # === Context Priming ===
        base_prompt = ""
        if self.src_lang in ("vi", "auto"):
            base_prompt = (
                "AI, ML, deep learning, machine learning, NLP, ChatGPT, GPT, "
                "Gemini, OpenAI, Google, transformer, neural network, LLM, "
                "computer vision, robotics, Python, TensorFlow, PyTorch, "
                "dataset, token, model, fine-tuning, pre-training, embedding, "
                "attention, inference, GPU, API, framework, server, deploy"
            )

        if self.topic and self.service.groq and self.service.groq.is_available:
            try:
                await self._send_log("Generating keywords from topic...", "info")
                keywords = await self.service.groq.expand_keywords(
                    self.topic, language=self.src_lang
                )
                if keywords:
                    self.initial_prompt = (
                        (base_prompt + ", " + keywords) if base_prompt else keywords
                    )
                    logger.info(f"[Context] Primed with: {self.initial_prompt[:80]}...")
                else:
                    self.initial_prompt = base_prompt
            except Exception as e:
                logger.error(f"[Context] Keyword expansion failed: {e}")
                self.initial_prompt = base_prompt
                await self._send_log(
                    "Keyword generation failed, continuing without", "warning"
                )
        else:
            self.initial_prompt = base_prompt

        logger.info(
            f"[Session] Started ({self.src_lang} → {self.tgt_lang})"
            f"{' | Topic: ' + self.topic[:40] if self.topic else ''}"
        )

        await self.out_queue.put(json.dumps({
            "type": "status",
            "status": "started",
            "topic": self.topic,
            "primed": bool(self.initial_prompt),
        }))

    # ─── Audio Processing (Segment-based) ─────────────────────────────

    async def _handle_audio(self, data: dict):
        """Legacy Base64 audio path"""
        if not self.is_recording:
            return
        async with self.lock:
            audio_b64 = data.get("audio", "")
            audio_chunk = decode_audio_chunk(audio_b64)
            await self._process_audio_chunk(audio_chunk)

    async def handle_binary_audio(self, audio_bytes: bytes):
        """Optimized binary audio path (Int16 → Float32)"""
        if not self.is_recording:
            return
        async with self.lock:
            audio_chunk = decode_audio_bytes(audio_bytes)
            await self._process_audio_chunk(audio_chunk)

    async def _process_audio_chunk(self, audio_chunk: np.ndarray):
        """
        Segment-based audio processing (Ricky's architecture):
        1. Check pending partial timeout
        2. VAD → detect speech (with RMS silence skip)
        3. SpeechSegmentBuffer → accumulate & detect segment boundaries
        4. If no boundary: intermediate decode for live feedback
        5. On boundary: full decode pipeline
        """
        if len(audio_chunk) == 0:
            return

        # Check pending partial timeout
        if self.pending_partial_text and self.pending_partial_time:
            if time.time() - self.pending_partial_time > self.PARTIAL_FINALIZE_TIMEOUT:
                await self._finalize_pending_partial()

        # RMS-based silence skip (avoids unnecessary VAD inference)
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        if rms < 0.003:
            is_speech = False
        else:
            is_speech = self.vad.is_speech(audio_chunk)

        # Timestamp within session
        now_ts = time.time() - self.session_start_time

        # Feed to segment buffer
        result = self.segmenter.process(audio_chunk, is_speech, now_ts)

        if result is None:
            # No segment boundary — try intermediate decode for live feedback
            buffer_duration = self.segmenter.get_current_duration()

            if (buffer_duration >= 2.0
                    and now_ts - self._last_intermediate_decode > 1.5):
                self._last_intermediate_decode = now_ts
                current_audio = self.segmenter.get_current_audio()

                if len(current_audio) > SAMPLE_RATE:  # > 1 second
                    asyncio.create_task(
                        self._decode_intermediate(current_audio)
                    )
            return

        # Segment boundary detected
        kind, chunk = result
        duration = len(chunk) / SAMPLE_RATE
        logger.info(f"[Segment] {kind.upper()} | {duration:.2f}s")

        await self._decode_segment(chunk, is_final=(kind == "final"))
    
    # ─── Decode Pipeline ─────────────────────────────────────────────

    async def _decode_segment(self, audio: np.ndarray, is_final: bool):
        """
        Full decode pipeline for a completed segment:
        WhisperX → HallucinationFilter → LocalAgreement → BARTpho → Stream
        """
        duration = len(audio) / SAMPLE_RATE

        if not is_final and duration < MIN_DECODE_SEC:
            logger.debug("Skip PARTIAL (too short)")
            return

        loop = asyncio.get_event_loop()
        start_time = time.time()

        prompt = self._build_prompt()
        whisper_lang = self.src_lang if self.src_lang != "auto" else None

        result = await loop.run_in_executor(
            None,
            lambda: self.service.asr.transcribe_segment(
                audio,
                initial_prompt=prompt,
                language=whisper_lang,
            )
        )
        asr_time = time.time() - start_time

        text = result.get("text", "").strip()
        words = result.get("words", [])
        detected_lang = result.get("language", self.src_lang)

        if self.src_lang == "auto" and detected_lang:
            self._detected_lang = detected_lang

        if not text:
            return

        # Compute confidence from WhisperX word-level scores
        if words:
            scores = [w.get("score", 0.0) for w in words if "score" in w]
            confidence = sum(scores) / len(scores) if scores else 0.9
        else:
            confidence = 0.9

        # Multi-layer hallucination filter (Ricky)
        audio_rms = float(np.sqrt(np.mean(audio ** 2)))
        is_hallu, hallu_reason = self.hallucination_filter.is_hallucination(
            text, audio_rms, confidence
        )

        if is_hallu:
            logger.info(f"[Filter] Hallucination: {hallu_reason}")
            return

        logger.info(f"[ASR] {'FINAL' if is_final else 'PARTIAL'}: {text[:60]}...")

        if is_final:
            # Finalize any pending partial first
            if self.pending_partial_text:
                await self._finalize_pending_partial()

            stable_text = text
            self.segment_id += 1
            self.last_stable = stable_text
            self.agreement.reset()

            # BARTpho correction (FINAL segments only, Vietnamese)
            effective_src = detected_lang if self.src_lang == "auto" else self.src_lang
            pp_time = 0.0
            if (effective_src == "vi"
                    and self.service.corrector
                    and self.service.corrector.is_loaded):
                pp_start = time.time()
                corrected = await loop.run_in_executor(
                    None, self.service.corrector.correct, stable_text
                )
                pp_time = time.time() - pp_start
                if corrected and corrected != stable_text:
                    logger.info(
                        f"[PP] \"{stable_text[:40]}\" → \"{corrected[:40]}\" "
                        f"({pp_time:.2f}s)"
                    )
                    stable_text = corrected

            # Collect for summary
            self.all_transcripts.append(stable_text)

            # Update transcript history (for prompt building)
            self.transcript_history.append(stable_text)
            if len(self.transcript_history) > 3:
                self.transcript_history.pop(0)

            # === Async Streaming: Send ASR immediately, translate async ===
            if self.do_translate and self.service.translator:
                # Send intermediate (source only)
                await self.out_queue.put(json.dumps({
                    "type": "transcript",
                    "segment_id": self.segment_id,
                    "source": stable_text,
                    "target": "",
                    "is_final": False,
                    "words": words,
                    "confidence": confidence,
                    "timestamp": self._get_timestamp(),
                    "timing": {
                        "asr_ms": int(asr_time * 1000),
                        "pp_ms": int(pp_time * 1000),
                        "mt_ms": 0,
                    }
                }))
                # Fire translation async
                asyncio.create_task(
                    self._translate_and_send(
                        self.segment_id, stable_text, words,
                        asr_time, pp_time, confidence,
                    )
                )
            else:
                # No translation — send final directly
                await self.out_queue.put(json.dumps({
                    "type": "transcript",
                    "segment_id": self.segment_id,
                    "source": stable_text,
                    "target": "",
                    "is_final": True,
                    "words": words,
                    "confidence": confidence,
                    "timestamp": self._get_timestamp(),
                    "timing": {
                        "asr_ms": int(asr_time * 1000),
                        "pp_ms": int(pp_time * 1000),
                        "mt_ms": 0,
                    }
                }))
        else:
            # PARTIAL: Apply local agreement to reduce flicker
            stable_text, unstable_text = self.agreement.process(text)
            display_text = f"{stable_text} {unstable_text}".strip()

            self.pending_partial_text = display_text
            self.pending_partial_time = time.time()

            await self.out_queue.put(json.dumps({
                "type": "transcript",
                "segment_id": self.segment_id + 1,
                "source": display_text,
                "target": "",
                "is_final": False,
                "words": words,
                "confidence": confidence,
                "timestamp": self._get_timestamp(),
                "timing": {
                    "asr_ms": int(asr_time * 1000),
                    "pp_ms": 0,
                    "mt_ms": 0,
                }
            }))

    async def _decode_intermediate(self, audio: np.ndarray):
        """
        Intermediate partial decode for live feedback while buffering.
        Less strict — no hallucination filter, just agreement smoothing.
        """
        loop = asyncio.get_event_loop()

        try:
            prompt = self._build_prompt()
            whisper_lang = self.src_lang if self.src_lang != "auto" else None

            result = await loop.run_in_executor(
                None,
                lambda: self.service.asr.transcribe_segment(
                    audio,
                    initial_prompt=prompt,
                    language=whisper_lang,
                )
            )

            text = result.get("text", "").strip()
            words = result.get("words", [])
            if words:
                scores = [w.get("score", 0.0) for w in words if "score" in w]
                confidence = sum(scores) / len(scores) if scores else 0.9
            else:
                confidence = 0.9

            if text and len(text) > 10:
                stable_text, unstable_text = self.agreement.process(text)
                display_text = f"{stable_text} {unstable_text}".strip()

                await self.out_queue.put(json.dumps({
                    "type": "transcript",
                    "segment_id": self.segment_id + 1,
                    "source": display_text,
                    "target": "",
                    "is_final": False,
                    "confidence": confidence,
                    "timestamp": self._get_timestamp(),
                }))
        except Exception as e:
            logger.error(f"[Intermediate] Error: {e}")

    async def _finalize_pending_partial(self):
        """Auto-finalize pending partial text after timeout"""
        if not self.pending_partial_text:
            return

        logger.info(f"[Partial] Auto-finalize: {self.pending_partial_text[:50]}...")
        self.segment_id += 1
        self.last_stable = self.pending_partial_text

        await self.out_queue.put(json.dumps({
            "type": "transcript",
            "segment_id": self.segment_id,
            "source": self.pending_partial_text,
            "target": "",
            "is_final": True,
            "confidence": 0.8,
            "timestamp": self._get_timestamp(),
        }))

        # Trigger translation for the finalized partial
        if self.do_translate and self.service.translator:
            asyncio.create_task(
                self._translate_and_send(
                    self.segment_id, self.pending_partial_text, [],
                    0.0, 0.0, 0.8,
                )
            )

        self.pending_partial_text = None
        self.pending_partial_time = None
        self.agreement.reset()

    # ─── Translation ──────────────────────────────────────────────────
    
    async def _translate_and_send(
        self, segment_id: int, text: str, words: list,
        asr_time: float, pp_time: float, confidence: float,
    ):
        """Async translation — fires after ASR result is already sent to client."""
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

            await self.out_queue.put(json.dumps({
                "type": "transcript",
                "segment_id": segment_id,
                "source": text,
                "target": translation,
                "is_final": True,
                "words": words,
                "confidence": confidence,
                "timestamp": self._get_timestamp(),
                "timing": {
                    "asr_ms": int(asr_time * 1000),
                    "pp_ms": int(pp_time * 1000),
                    "mt_ms": int(mt_time * 1000),
                }
            }))
        except Exception as e:
            logger.error(f"Translation error for segment {segment_id}: {e}")
            await self.out_queue.put(json.dumps({
                "type": "transcript",
                "segment_id": segment_id,
                "source": text,
                "target": "",
                "is_final": True,
                "words": words,
                "confidence": confidence,
                "timestamp": self._get_timestamp(),
                "timing": {
                    "asr_ms": int(asr_time * 1000),
                    "pp_ms": int(pp_time * 1000),
                    "mt_ms": 0,
                }
            }))

    # ─── Stop / Summarize / Cleanup ───────────────────────────────────

    async def _handle_stop(self):
        """Stop recording and finalize any remaining audio"""
        if not self.is_recording:
            return

        self.is_recording = False

        # Finalize any pending partial
        if self.pending_partial_text:
            await self._finalize_pending_partial()

        session_duration = time.time() - self.session_start_time

        logger.info(
            f"[Session] Stopped | Segments: {self.segment_id} "
            f"| Duration: {session_duration:.0f}s"
        )

        await self.out_queue.put(json.dumps({
            "type": "status",
            "status": "stopped",
            "segments": self.segment_id,
        }))

        # Auto Summary via Groq (if session > threshold)
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
        self.all_transcripts = []

    async def _generate_summary(self):
        """Generate lecture summary via Groq"""
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
