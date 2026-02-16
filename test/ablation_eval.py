"""
Ablation Study — Component-Level Evaluation for Thesis

Systematically enables/disables each merged component to measure its
individual contribution to ASR quality and latency.

═══════════════════════════════════════════════════════════════════════
Ablation Configurations:
═══════════════════════════════════════════════════════════════════════
  A0: Baseline         — Raw WhisperX, no enhancements
  A1: +Normalizer      — A0 + AdaptiveNormalizer (highpass, soft clip, noise floor)
  A2: +SegmentBuffer   — A1 + SpeechSegmentBuffer (overlap-based vs simple buffer)
  A3: +Agreement       — A2 + LocalAgreement (common-prefix text stabilization)
  A4: +HalluFilter     — A3 + Multi-layer HallucinationFilter
  A5: +BARTpho         — A4 + BARTpho syllable correction (3-layer EN/VI split)
  A6: Full Pipeline    — All components enabled (= production system)

═══════════════════════════════════════════════════════════════════════
Datasets:
═══════════════════════════════════════════════════════════════════════
  - VLSP 2020 (vi)          : Vietnamese ASR benchmark
  - VIVOS (vi)              : Vietnamese read speech (clean)
  - CommonVoice 17.0 (vi)   : Vietnamese crowd-sourced (noisy)
  - Earnings22 (en)         : English financial calls (accented, noisy)
  - LibriSpeech test-clean   : English read speech (clean baseline)

═══════════════════════════════════════════════════════════════════════
Metrics:
═══════════════════════════════════════════════════════════════════════
  Quality:
    - WER  : Word Error Rate              (lower = better)
    - CER  : Character Error Rate         (lower = better)
  Latency:
    - TTFT : Time to First Token (ms)     (lower = better)
    - AL   : Average Lagging (ms)         (lower = better)
    - RTF  : Real-Time Factor             (<1.0 = faster than real-time)
  Streaming:
    - FLK  : Flicker Rate                 (% of partial changes, lower = better)
    - HAL  : Hallucination Rate           (% segments filtered, lower = better)

═══════════════════════════════════════════════════════════════════════
Usage:
═══════════════════════════════════════════════════════════════════════
  # Run single ablation config
  modal run test/ablation_eval.py --config A0 --dataset vlsp2020 --samples 200

  # Run all ablations on one dataset
  modal run test/ablation_eval.py --config ALL --dataset vlsp2020 --samples 200

  # Run full matrix (all configs × all datasets)
  modal run test/ablation_eval.py --config ALL --dataset ALL --samples 200

Results → test/results/ablation/{config}/{dataset}.json
"""

import modal
import os
import re
import json
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional

app = modal.App("ablation-eval")

# ─── Modal Image ──────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.35.0", "accelerate>=0.25.0", "sentencepiece>=0.1.99",
    )
    .pip_install("whisperx")
    .pip_install("peft>=0.8.0", "bitsandbytes>=0.41.0")
    .pip_install(
        "datasets>=2.14.0", "jiwer>=3.0.0", "librosa>=0.10.0",
        "soundfile>=0.12.0", "scipy>=1.11.0", "numpy>=1.24.0",
        "huggingface_hub>=0.20.0",
    )
    .add_local_dir("src", remote_path="/root/src", copy=True)
)

# ─── Dataset Configurations ───────────────────────────────────────────────────

DATASETS = {
    "vlsp2020": {
        "hf_name": "doof-ferb/vlsp2020_vinai_100h",
        "split": "train",
        "audio_col": "audio",
        "text_col": "transcription",
        "language": "vi",
        "description": "VLSP 2020 Vietnamese ASR benchmark (100h)",
    },
    "vivos": {
        "hf_name": "vivos",
        "split": "test",
        "audio_col": "audio",
        "text_col": "sentence",
        "language": "vi",
        "description": "VIVOS Vietnamese read speech (clean)",
    },
    "commonvoice_vi": {
        "hf_name": "mozilla-foundation/common_voice_17_0",
        "config": "vi",
        "split": "test",
        "audio_col": "audio",
        "text_col": "sentence",
        "language": "vi",
        "description": "CommonVoice 17.0 Vietnamese (crowd-sourced, noisy)",
    },
    "earnings22": {
        "hf_name": "distil-whisper/earnings22",
        "config": "chunked",
        "split": "test",
        "audio_col": "audio",
        "text_col": "transcription",
        "language": "en",
        "description": "Earnings22 English financial calls (accented)",
    },
    "librispeech_clean": {
        "hf_name": "librispeech_asr",
        "config": "clean",
        "split": "test",
        "audio_col": "audio",
        "text_col": "text",
        "language": "en",
        "description": "LibriSpeech test-clean (English read speech)",
    },
}

# ─── Ablation Configurations ──────────────────────────────────────────────────

@dataclass
class AblationConfig:
    """Which components are enabled for this ablation run."""
    name: str
    description: str
    use_normalizer: bool = False
    use_segment_buffer: bool = False   # True = SpeechSegmentBuffer, False = simple buffer
    use_local_agreement: bool = False
    use_hallucination_filter: bool = False
    use_bartpho: bool = False

ABLATIONS = {
    "A0": AblationConfig(
        name="A0_baseline",
        description="Baseline: Raw WhisperX, no enhancements",
    ),
    "A1": AblationConfig(
        name="A1_normalizer",
        description="+AdaptiveNormalizer (highpass 80Hz, soft clip, noise floor)",
        use_normalizer=True,
    ),
    "A2": AblationConfig(
        name="A2_segment_buffer",
        description="+SpeechSegmentBuffer (overlap-based segmentation)",
        use_normalizer=True,
        use_segment_buffer=True,
    ),
    "A3": AblationConfig(
        name="A3_agreement",
        description="+LocalAgreement (partial text stabilization)",
        use_normalizer=True,
        use_segment_buffer=True,
        use_local_agreement=True,
    ),
    "A4": AblationConfig(
        name="A4_hallu_filter",
        description="+Multi-layer HallucinationFilter",
        use_normalizer=True,
        use_segment_buffer=True,
        use_local_agreement=True,
        use_hallucination_filter=True,
    ),
    "A5": AblationConfig(
        name="A5_bartpho",
        description="+BARTpho syllable correction (3-layer EN/VI)",
        use_normalizer=True,
        use_segment_buffer=True,
        use_local_agreement=True,
        use_hallucination_filter=True,
        use_bartpho=True,
    ),
    "A6": AblationConfig(
        name="A6_full_pipeline",
        description="Full pipeline (all components enabled)",
        use_normalizer=True,
        use_segment_buffer=True,
        use_local_agreement=True,
        use_hallucination_filter=True,
        use_bartpho=True,
    ),
}

# ─── Streaming Parameters ─────────────────────────────────────────────────────

SAMPLE_RATE = 16000
CHUNK_MS = 100  # Simulate 100ms audio chunks
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)

# Simple buffer parameters (for A0-A1 without SpeechSegmentBuffer)
SIMPLE_MAX_BUFFER = 5.0  # seconds
SIMPLE_SILENCE_LIMIT = 0.5  # seconds
SIMPLE_MIN_SEGMENT = 1.0  # seconds

# SpeechSegmentBuffer parameters (for A2+)
SEG_MAX_SEC = 5.0
SEG_OVERLAP_SEC = 0.8
SEG_SILENCE_LIMIT = 0.3


# ─── Text Normalization ──────────────────────────────────────────────────────

def normalize_text(text: str, language: str = "vi") -> str:
    """Normalize text for fair comparison."""
    if not text:
        return ""
    text = text.lower()
    if language == "vi":
        # Keep Vietnamese diacritics
        text = re.sub(
            r'[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]',
            '', text,
        )
    else:
        text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split()).strip()


# ─── Core Evaluation ──────────────────────────────────────────────────────────

def run_ablation(
    ablation_config: AblationConfig,
    dataset_name: str,
    max_samples: int,
):
    """
    Run a single ablation evaluation.

    Returns dict with all metrics + per-sample details.
    """
    import sys
    sys.path.insert(0, "/root")

    from jiwer import wer as compute_wer, cer as compute_cer

    config = DATASETS[dataset_name]
    language = config["language"]

    print(f"\n{'='*70}")
    print(f"  ABLATION: {ablation_config.name}")
    print(f"  {ablation_config.description}")
    print(f"  Dataset: {dataset_name} ({config['description']})")
    print(f"  Samples: {max_samples}")
    print(f"{'='*70}")

    # ── Load Models ──

    # 1. WhisperX ASR (always loaded)
    from src.asr import WhisperXASR
    print("[Model] Loading WhisperX large-v3...")
    asr = WhisperXASR(
        model_size="large-v3",
        language=language if language != "auto" else None,
        device="cuda",
    )
    asr.load_model()

    # Optionally disable normalizer
    if not ablation_config.use_normalizer:
        asr.normalizer = None
        print("[Ablation] AdaptiveNormalizer DISABLED")
    else:
        print("[Ablation] AdaptiveNormalizer ENABLED")

    # 2. VAD (always needed)
    from src.vad import SileroVAD
    print("[Model] Loading Silero VAD...")
    vad = SileroVAD()
    vad.load_model()

    # 3. SpeechSegmentBuffer (conditional)
    segmenter = None
    if ablation_config.use_segment_buffer:
        from src.session.speech_segment_buffer import SpeechSegmentBuffer
        segmenter = SpeechSegmentBuffer(
            sample_rate=SAMPLE_RATE,
            max_sec=SEG_MAX_SEC,
            overlap_sec=SEG_OVERLAP_SEC,
            silence_limit=SEG_SILENCE_LIMIT,
        )
        print("[Ablation] SpeechSegmentBuffer ENABLED")
    else:
        print("[Ablation] Using simple buffer (no overlap)")

    # 4. LocalAgreement (conditional)
    agreement = None
    if ablation_config.use_local_agreement:
        from src.session.local_agreement import LocalAgreement
        agreement = LocalAgreement(n=2)
        print("[Ablation] LocalAgreement ENABLED")

    # 5. HallucinationFilter (conditional)
    hallu_filter = None
    if ablation_config.use_hallucination_filter:
        from src.session.filters import HallucinationFilter
        hallu_filter = HallucinationFilter(history_size=5)
        print("[Ablation] HallucinationFilter ENABLED")

    # 6. BARTpho (conditional, Vietnamese only)
    corrector = None
    if ablation_config.use_bartpho and language == "vi":
        from src.postprocess import BARTphoCorrector
        from src.config import BARTPHO_ADAPTER, BARTPHO_DEVICE
        print("[Model] Loading BARTpho...")
        corrector = BARTphoCorrector(
            adapter_id=BARTPHO_ADAPTER,
            device=BARTPHO_DEVICE,
            cache_dir="/cache/huggingface",
        )
        corrector.load_model()
        print("[Ablation] BARTpho ENABLED")
    else:
        print(f"[Ablation] BARTpho {'DISABLED' if language == 'vi' else 'N/A (not Vietnamese)'}")

    # ── Load Dataset ──
    print(f"\n[Data] Loading {dataset_name}...")
    samples = load_dataset_samples(config, max_samples)
    print(f"[Data] Loaded {len(samples)} samples")

    if not samples:
        return {"error": "No samples loaded"}

    # ── Run Streaming Simulation ──
    print(f"\n[Eval] Running streaming simulation...")

    references = []
    hypotheses = []
    metrics_per_sample = []

    for idx, sample in enumerate(samples):
        audio = sample["audio"]
        reference = sample["text"]

        result = simulate_streaming_session(
            audio=audio,
            asr=asr,
            vad=vad,
            language=language,
            segmenter=segmenter,
            agreement=agreement,
            hallu_filter=hallu_filter,
            corrector=corrector,
            ablation_config=ablation_config,
        )

        ref_norm = normalize_text(reference, language)
        hyp_norm = normalize_text(result["final_text"], language)

        if ref_norm and hyp_norm:
            references.append(ref_norm)
            hypotheses.append(hyp_norm)
            metrics_per_sample.append({
                "ref": ref_norm[:80],
                "hyp": hyp_norm[:80],
                "ttft": result["ttft"],
                "rtf": result["rtf"],
                "al": result["al"],
                "latency": result["latency"],
                "num_emissions": result["num_emissions"],
                "num_partials": result["num_partials"],
                "num_flickers": result["num_flickers"],
                "num_filtered": result["num_filtered"],
            })

        if (idx + 1) % 25 == 0:
            print(f"  [{idx+1}/{len(samples)}] processed")

    # ── Compute Aggregate Metrics ──
    if references and hypotheses:
        total_wer = compute_wer(references, hypotheses)
        total_cer = compute_cer(references, hypotheses)
    else:
        total_wer = total_cer = 1.0

    ttfts = [m["ttft"] for m in metrics_per_sample]
    rtfs = [m["rtf"] for m in metrics_per_sample]
    als = [m["al"] for m in metrics_per_sample]
    latencies = [m["latency"] for m in metrics_per_sample]
    total_partials = sum(m["num_partials"] for m in metrics_per_sample)
    total_flickers = sum(m["num_flickers"] for m in metrics_per_sample)
    total_filtered = sum(m["num_filtered"] for m in metrics_per_sample)
    total_emissions = sum(m["num_emissions"] for m in metrics_per_sample)

    flicker_rate = total_flickers / max(total_partials, 1)
    hallucination_rate = total_filtered / max(total_emissions + total_filtered, 1)

    result = {
        "ablation": ablation_config.name,
        "ablation_desc": ablation_config.description,
        "ablation_flags": asdict(ablation_config),
        "dataset": dataset_name,
        "dataset_desc": config["description"],
        "language": language,
        "num_samples": len(references),
        # Quality metrics
        "wer": round(total_wer, 4),
        "cer": round(total_cer, 4),
        # Latency metrics
        "avg_ttft_ms": round(np.mean(ttfts) * 1000, 1) if ttfts else 0,
        "avg_al_ms": round(np.mean(als) * 1000, 1) if als else 0,
        "avg_rtf": round(np.mean(rtfs), 4) if rtfs else 0,
        "avg_latency_ms": round(np.mean(latencies) * 1000, 1) if latencies else 0,
        "p50_latency_ms": round(np.percentile(latencies, 50) * 1000, 1) if latencies else 0,
        "p90_latency_ms": round(np.percentile(latencies, 90) * 1000, 1) if latencies else 0,
        "p99_latency_ms": round(np.percentile(latencies, 99) * 1000, 1) if latencies else 0,
        # Streaming quality metrics
        "flicker_rate": round(flicker_rate, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "total_partials": total_partials,
        "total_flickers": total_flickers,
        "total_filtered": total_filtered,
    }

    # Print summary
    print(f"\n{'='*70}")
    print(f"  RESULTS: {ablation_config.name} on {dataset_name}")
    print(f"{'='*70}")
    print(f"  WER:  {result['wer']*100:.2f}%  |  CER: {result['cer']*100:.2f}%")
    print(f"  TTFT: {result['avg_ttft_ms']:.0f}ms  |  AL:  {result['avg_al_ms']:.0f}ms")
    print(f"  RTF:  {result['avg_rtf']:.3f}x  |  P90 Lat: {result['p90_latency_ms']:.0f}ms")
    print(f"  FLK:  {result['flicker_rate']*100:.1f}%  |  HAL: {result['hallucination_rate']*100:.1f}%")
    print(f"{'='*70}")

    return result


# ─── Streaming Simulation ─────────────────────────────────────────────────────

def simulate_streaming_session(
    audio: np.ndarray,
    asr,
    vad,
    language: str,
    segmenter,
    agreement,
    hallu_filter,
    corrector,
    ablation_config: AblationConfig,
):
    """
    Simulate a streaming ASR session for one audio sample.

    Returns metrics dict with: final_text, ttft, al, rtf, latency,
    num_emissions, num_partials, num_flickers, num_filtered.
    """
    audio_duration = len(audio) / SAMPLE_RATE
    num_chunks = max(1, len(audio) // CHUNK_SAMPLES)
    chunk_duration = CHUNK_SAMPLES / SAMPLE_RATE

    session_start = time.time()
    first_emission_time = None
    emissions = []
    all_final_texts = []

    # Streaming state
    num_partials = 0
    num_flickers = 0
    num_filtered = 0
    last_partial_text = ""

    # Reset components
    if segmenter:
        segmenter.reset()
    if agreement:
        agreement.reset()
    if hallu_filter:
        hallu_filter.reset()

    # Simple buffer state (when not using SpeechSegmentBuffer)
    simple_buffer = np.array([], dtype=np.float32)
    last_speech_time = 0.0

    for i in range(num_chunks):
        start_idx = i * CHUNK_SAMPLES
        end_idx = min((i + 1) * CHUNK_SAMPLES, len(audio))
        chunk = audio[start_idx:end_idx]
        sim_time = (i + 1) * chunk_duration

        # VAD
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        is_speech = vad.is_speech(chunk) if rms >= 0.003 else False

        if is_speech:
            last_speech_time = sim_time

        # ── Path A: SpeechSegmentBuffer ──
        if segmenter:
            seg_result = segmenter.process(chunk, is_speech, sim_time)

            if seg_result is None:
                # Intermediate decode for live feedback
                buf_dur = segmenter.get_current_duration()
                if buf_dur >= 2.0:
                    current_audio = segmenter.get_current_audio()
                    if len(current_audio) > SAMPLE_RATE:
                        # Partial decode
                        text = _transcribe(asr, current_audio, language)
                        if text:
                            num_partials += 1
                            if agreement:
                                stable, unstable = agreement.process(text)
                                display = f"{stable} {unstable}".strip()
                            else:
                                display = text
                            if display != last_partial_text:
                                num_flickers += 1
                            last_partial_text = display
                continue

            kind, seg_audio = seg_result
            # Decode final segment
            text = _transcribe(asr, seg_audio, language)
            if not text:
                continue

            # Hallucination filter
            if hallu_filter:
                seg_rms = float(np.sqrt(np.mean(seg_audio ** 2)))
                is_hallu, _ = hallu_filter.is_hallucination(text, seg_rms, 0.9)
                if is_hallu:
                    num_filtered += 1
                    continue

            # BARTpho correction
            if corrector and corrector.is_loaded:
                corrected = corrector.correct(text)
                if corrected:
                    text = corrected

            # LocalAgreement (reset on final)
            if agreement:
                agreement.reset()

            wall_time = time.time() - session_start
            emissions.append((wall_time, sim_time))
            if first_emission_time is None:
                first_emission_time = wall_time
            all_final_texts.append(text)

        # ── Path B: Simple buffer ──
        else:
            simple_buffer = np.concatenate([simple_buffer, chunk])
            buffer_dur = len(simple_buffer) / SAMPLE_RATE
            silence_dur = sim_time - last_speech_time

            should_finalize = (
                buffer_dur >= SIMPLE_MAX_BUFFER
                or (silence_dur >= SIMPLE_SILENCE_LIMIT and buffer_dur >= SIMPLE_MIN_SEGMENT)
            )

            if should_finalize and len(simple_buffer) > int(SIMPLE_MIN_SEGMENT * SAMPLE_RATE):
                text = _transcribe(asr, simple_buffer, language)
                simple_buffer = np.array([], dtype=np.float32)

                if not text:
                    continue

                # Hallucination filter
                if hallu_filter:
                    buf_rms = float(np.sqrt(np.mean(simple_buffer ** 2))) if len(simple_buffer) > 0 else rms
                    is_hallu, _ = hallu_filter.is_hallucination(text, buf_rms, 0.9)
                    if is_hallu:
                        num_filtered += 1
                        continue

                # BARTpho
                if corrector and corrector.is_loaded:
                    corrected = corrector.correct(text)
                    if corrected:
                        text = corrected

                wall_time = time.time() - session_start
                emissions.append((wall_time, sim_time))
                if first_emission_time is None:
                    first_emission_time = wall_time
                all_final_texts.append(text)

    # Process remaining audio in simple buffer
    if not segmenter and len(simple_buffer) > int(SIMPLE_MIN_SEGMENT * SAMPLE_RATE):
        text = _transcribe(asr, simple_buffer, language)
        if text:
            wall_time = time.time() - session_start
            emissions.append((wall_time, audio_duration))
            if first_emission_time is None:
                first_emission_time = wall_time
            all_final_texts.append(text)

    total_time = time.time() - session_start
    final_text = " ".join(all_final_texts)

    # Compute metrics
    ttft = first_emission_time if first_emission_time else audio_duration
    al = np.mean([w - s for w, s in emissions]) if emissions else total_time
    rtf = total_time / audio_duration if audio_duration > 0 else 1.0

    return {
        "final_text": final_text,
        "ttft": ttft,
        "al": al,
        "rtf": rtf,
        "latency": total_time,
        "num_emissions": len(emissions),
        "num_partials": num_partials,
        "num_flickers": num_flickers,
        "num_filtered": num_filtered,
    }


def _transcribe(asr, audio: np.ndarray, language: str) -> str:
    """Transcribe audio segment using WhisperX."""
    if len(audio) < 8000:
        return ""
    result = asr.transcribe_segment(
        audio,
        language=language if language != "auto" else None,
    )
    return result.get("text", "").strip()


# ─── Dataset Loading ──────────────────────────────────────────────────────────

def load_dataset_samples(config: dict, max_samples: int):
    """Load audio samples from HuggingFace dataset (streaming mode)."""
    from datasets import load_dataset
    import librosa

    load_kwargs = {"split": config["split"], "streaming": True}
    if config.get("config"):
        load_kwargs["name"] = config["config"]

    ds = load_dataset(config["hf_name"], **load_kwargs, trust_remote_code=True)

    samples = []
    audio_col = config["audio_col"]
    text_col = config["text_col"]

    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        try:
            text = str(item.get(text_col, "")).strip()
            if not text or len(text) < 3:
                continue

            audio_data = item[audio_col]
            audio, sr = _decode_audio(audio_data)
            if audio is None or len(audio) < 8000:
                continue

            if sr != SAMPLE_RATE and sr > 0:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

            samples.append({"audio": audio, "text": text})
        except Exception as e:
            if i < 3:
                print(f"  Warning: Failed sample {i}: {e}")
            continue

    return samples


def _decode_audio(audio_data):
    """Decode audio from various HuggingFace formats."""
    audio, sr = None, SAMPLE_RATE

    if hasattr(audio_data, 'get_all_samples'):
        samples_obj = audio_data.get_all_samples()
        if hasattr(samples_obj, 'data'):
            data = samples_obj.data
            audio = data.numpy().flatten().astype(np.float32) if hasattr(data, 'numpy') else np.asarray(data, dtype=np.float32).flatten()
            sr = getattr(samples_obj, 'sample_rate', SAMPLE_RATE)
    elif isinstance(audio_data, dict):
        if "array" in audio_data:
            arr = audio_data["array"]
            sr = audio_data.get("sampling_rate", SAMPLE_RATE)
            audio = arr.numpy().flatten().astype(np.float32) if hasattr(arr, 'numpy') else np.asarray(arr, dtype=np.float32).flatten()
        elif "bytes" in audio_data:
            import soundfile as sf
            import io
            audio, sr = sf.read(io.BytesIO(audio_data["bytes"]))
            audio = audio.astype(np.float32).flatten()
    elif hasattr(audio_data, 'numpy'):
        audio = audio_data.numpy().flatten().astype(np.float32)

    return audio, sr


# ─── Modal Functions ──────────────────────────────────────────────────────────

@app.function(
    gpu="A100",
    timeout=7200,
    memory=32768,
    image=image,
    volumes={"/cache": modal.Volume.from_name("asr-model-cache", create_if_missing=True)},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def evaluate_ablation(config_name: str, dataset_name: str, max_samples: int = 200):
    """Run single ablation config on single dataset."""
    import sys, os
    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/cache/huggingface"

    from src.utils import apply_torch_load_patch
    apply_torch_load_patch()

    ablation_config = ABLATIONS[config_name]
    return run_ablation(ablation_config, dataset_name, max_samples)


@app.local_entrypoint()
def main(
    config: str = "ALL",
    dataset: str = "vlsp2020",
    samples: int = 200,
):
    """
    Run ablation evaluation.

    Args:
        config: Ablation config (A0-A6) or ALL
        dataset: Dataset name or ALL
        samples: Number of samples per evaluation
    """
    from pathlib import Path

    configs = list(ABLATIONS.keys()) if config == "ALL" else [config]
    datasets = list(DATASETS.keys()) if dataset == "ALL" else [dataset]

    print(f"\n{'#'*70}")
    print(f"# ABLATION STUDY")
    print(f"# Configs: {configs}")
    print(f"# Datasets: {datasets}")
    print(f"# Samples per eval: {samples}")
    print(f"# Total runs: {len(configs) * len(datasets)}")
    print(f"{'#'*70}")

    all_results = []

    for cfg in configs:
        for ds in datasets:
            print(f"\n>>> Running {cfg} on {ds}...")
            try:
                result = evaluate_ablation.remote(cfg, ds, samples)

                # Save individual result
                results_dir = Path(__file__).parent / "results" / "ablation" / cfg
                results_dir.mkdir(parents=True, exist_ok=True)
                output_file = results_dir / f"{ds}.json"
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"    Saved: {output_file}")

                all_results.append(result)
            except Exception as e:
                print(f"    ERROR: {e}")
                all_results.append({"ablation": cfg, "dataset": ds, "error": str(e)})

    # Save combined results
    combined_file = Path(__file__).parent / "results" / "ablation" / "combined.json"
    combined_file.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n\n{'='*90}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*90}")
    print(f"{'Config':<20} {'Dataset':<18} {'WER%':>7} {'CER%':>7} {'TTFT':>7} {'RTF':>7} {'FLK%':>7} {'HAL%':>7}")
    print(f"{'-'*90}")
    for r in all_results:
        if "error" in r:
            print(f"{r.get('ablation','?'):<20} {r.get('dataset','?'):<18} {'ERROR':>7}")
        else:
            print(
                f"{r['ablation']:<20} {r['dataset']:<18} "
                f"{r['wer']*100:>6.2f}% {r['cer']*100:>6.2f}% "
                f"{r['avg_ttft_ms']:>6.0f}m {r['avg_rtf']:>6.3f}x "
                f"{r['flicker_rate']*100:>5.1f}% {r['hallucination_rate']*100:>5.1f}%"
            )
    print(f"{'='*90}")
