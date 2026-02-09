"""
Benchmark: WhisperX (our system) vs Whisper large-v3 baseline
Measures impact of BARTpho post-processing on WER/CER.

Configs:
  A. WhisperX large-v3                    (our ASR engine)
  B. WhisperX large-v3 + BARTpho         (our full pipeline)
  C. Whisper large-v3 (transformers)      (vanilla baseline)

Dataset: VLSP 2020 Vietnamese (doof-ferb/vlsp2020_vinai_100h)

Usage:
    modal run test/compare_models.py --samples 100
    modal run test/compare_models.py --samples 50 --gpu h100
"""

import modal, re, time

app = modal.App("compare-models")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.6.0", "torchaudio==2.6.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "faster-whisper>=1.0.0",
        "transformers>=4.45.0",
        "accelerate>=0.25.0",
        "datasets>=2.14.0,<3.0.0",
        "jiwer>=3.0.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "numpy>=1.24.0",
        "tabulate>=0.9.0",
        "whisperx",
        "peft>=0.8.0",
        "sentencepiece>=0.1.99",
    )
)

DATASET = "doof-ferb/vlsp2020_vinai_100h"
SAMPLE_RATE = 16000
_VI = r"àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"


def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(rf"[^\w\s{_VI}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_samples(max_n: int) -> list[dict]:
    from datasets import load_dataset
    import numpy as np, librosa

    ds = load_dataset(DATASET, split="train", streaming=True)
    samples = []
    for i, item in enumerate(ds):
        if len(samples) >= max_n:
            break
        try:
            ref = str(item.get("transcription", "")).strip()
            if len(ref) < 3:
                continue
            audio_data = item["audio"]
            arr = np.asarray(audio_data["array"], dtype=np.float32)
            sr = int(audio_data.get("sampling_rate", SAMPLE_RATE))
            if sr != SAMPLE_RATE:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)
            if len(arr) < 4000:
                continue
            samples.append({"audio": arr, "ref": ref, "id": f"s{i:05d}"})
        except Exception as e:
            if i < 3:
                print(f"  skip {i}: {e}")
    return samples


def run_configs(samples: list[dict]) -> dict:
    import torch, whisperx
    from transformers import pipeline as hf_pipeline
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    # Patch torch.load for pyannote compatibility (weights_only=True breaks it on 2.6+)
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load

    results = {}

    # ── Config A: WhisperX large-v3 ──
    print("\n  Loading WhisperX large-v3...")
    wx_model = whisperx.load_model("large-v3", "cuda", compute_type="float16", language="vi")
    print("  Running WhisperX...")
    wx_results = []
    for i, s in enumerate(samples):
        try:
            t0 = time.perf_counter()
            out = wx_model.transcribe(s["audio"], batch_size=16, language="vi")
            dt = time.perf_counter() - t0
            text = " ".join(seg.get("text", "").strip() for seg in out.get("segments", []))
            wx_results.append({"hyp": text.strip(), "time": dt})
        except Exception as e:
            wx_results.append({"hyp": "", "time": 0, "err": str(e)})
        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{len(samples)}]")
    results["WhisperX"] = wx_results

    # ── Config B: WhisperX + BARTpho ──
    print("\n  Loading BARTpho corrector...")
    bartpho_adapter = "522H0134-NguyenNhatHuy/bartpho-syllable-correction"
    bartpho_base = "vinai/bartpho-syllable"
    tok = AutoTokenizer.from_pretrained(bartpho_adapter)
    base_m = AutoModelForSeq2SeqLM.from_pretrained(bartpho_base, torch_dtype=torch.float16)
    corrector = PeftModel.from_pretrained(base_m, bartpho_adapter).to("cuda").eval()
    print("  Running WhisperX + BARTpho...")
    pp_results = []
    for i, s in enumerate(samples):
        wx_r = wx_results[i]
        raw_hyp = wx_r["hyp"]
        try:
            t0 = time.perf_counter()
            if raw_hyp.strip():
                inputs = tok(raw_hyp, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad():
                    out = corrector.generate(**inputs, max_length=256, num_beams=4, early_stopping=True)
                corrected = tok.decode(out[0], skip_special_tokens=True).strip()
            else:
                corrected = ""
            dt_pp = time.perf_counter() - t0
            pp_results.append({"hyp": corrected, "time": wx_r["time"] + dt_pp})
        except Exception as e:
            pp_results.append({"hyp": raw_hyp, "time": wx_r["time"], "err": str(e)})
        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{len(samples)}]")
    results["WhisperX+BARTpho"] = pp_results
    del corrector, base_m, tok
    torch.cuda.empty_cache()
    del wx_model
    torch.cuda.empty_cache()

    # ── Config C: Vanilla Whisper large-v3 (transformers) ──
    print("\n  Loading Whisper large-v3 (transformers)...")
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float16,
        device="cuda",
    )
    print("  Running Whisper baseline...")
    bl_results = []
    for i, s in enumerate(samples):
        try:
            t0 = time.perf_counter()
            out = pipe(
                {"raw": s["audio"], "sampling_rate": SAMPLE_RATE},
                generate_kwargs={"language": "vi", "task": "transcribe"},
            )
            dt = time.perf_counter() - t0
            bl_results.append({"hyp": out["text"].strip(), "time": dt})
        except Exception as e:
            bl_results.append({"hyp": "", "time": 0, "err": str(e)})
        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{len(samples)}]")
    results["Baseline"] = bl_results
    del pipe
    torch.cuda.empty_cache()

    return results


def compute(samples, all_results):
    from jiwer import wer as jwer, cer as jcer
    import numpy as np

    metrics = {}
    for name, results in all_results.items():
        wers, cers, lats = [], [], []
        for s, r in zip(samples, results):
            ref = normalize(s["ref"])
            if not ref:
                continue
            hyp = normalize(r["hyp"])
            wers.append(jwer(ref, hyp) if hyp else 1.0)
            cers.append(jcer(ref, hyp) if hyp else 1.0)
            lats.append(r["time"])
        metrics[name] = {
            "wer": float(np.mean(wers)),
            "cer": float(np.mean(cers)),
            "lat_avg": float(np.mean(lats)),
            "lat_p90": float(np.percentile(lats, 90)),
        }

    details = []
    for i, s in enumerate(samples):
        row = {"id": s["id"], "ref": s["ref"]}
        for name, res in all_results.items():
            key = name.replace("+", "_plus_").replace(" ", "_").lower()
            row[f"hyp_{key}"] = res[i]["hyp"]
            ref_n = normalize(s["ref"])
            hyp_n = normalize(res[i]["hyp"])
            row[f"wer_{key}"] = float(jwer(ref_n, hyp_n)) if ref_n and hyp_n else 1.0
        details.append(row)

    return metrics, details


def run_benchmark(max_samples: int):
    t_total = time.perf_counter()

    configs = ["WhisperX", "WhisperX+BARTpho", "Baseline"]

    print(f"\n{'='*65}")
    print(f"  WhisperX vs Baseline  |  VLSP2020  |  n={max_samples}")
    print(f"    A. WhisperX large-v3")
    print(f"    B. WhisperX large-v3 + BARTpho correction")
    print(f"    C. Whisper large-v3 (transformers baseline)")
    print(f"{'='*65}")

    print("\n[1/3] Loading samples...")
    samples = load_samples(max_samples)
    print(f"  → {len(samples)} loaded")
    if not samples:
        return {"error": "no samples"}

    print("[2/3] Running configs...")
    all_results = run_configs(samples)

    print("\n[3/3] Computing metrics...")
    metrics, details = compute(samples, all_results)

    from tabulate import tabulate

    headers = ["Metric"] + configs
    table = [
        ["WER ↓"]  + [f'{metrics[n]["wer"]:.2%}' for n in configs],
        ["CER ↓"]  + [f'{metrics[n]["cer"]:.2%}' for n in configs],
        ["Lat avg"] + [f'{metrics[n]["lat_avg"]:.3f}s' for n in configs],
        ["Lat P90"] + [f'{metrics[n]["lat_p90"]:.3f}s' for n in configs],
    ]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

    best = min(configs, key=lambda n: metrics[n]["wer"])
    worst = max(configs, key=lambda n: metrics[n]["wer"])
    if best != worst:
        diff = (metrics[worst]["wer"] - metrics[best]["wer"]) / metrics[worst]["wer"] * 100
        print(f"\n  ✓ Best: {best} (WER {diff:.1f}% lower than {worst})")

    # Show BARTpho improvement
    wx_wer = metrics["WhisperX"]["wer"]
    pp_wer = metrics["WhisperX+BARTpho"]["wer"]
    if wx_wer > 0:
        pp_gain = (wx_wer - pp_wer) / wx_wer * 100
        print(f"  → BARTpho reduces WER by {pp_gain:.1f}%" if pp_gain > 0
              else f"  → BARTpho increases WER by {abs(pp_gain):.1f}%")

    elapsed = time.perf_counter() - t_total
    print(f"  Total: {elapsed:.0f}s")

    import csv, json
    from pathlib import Path
    out = Path("/root/results")
    out.mkdir(exist_ok=True)

    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=details[0].keys())
        w.writeheader()
        w.writerows(details)

    summary = {
        "n": len(samples),
        "total_time": round(elapsed, 1),
        "models": {n: {k: round(v, 4) for k, v in m.items()} for n, m in metrics.items()},
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved → /root/results/\n")
    return summary


@app.function(gpu="A100", timeout=3600, image=image, secrets=[modal.Secret.from_name("huggingface-secret")])
def bench_a100(max_samples: int = 100):
    return run_benchmark(max_samples)


@app.function(gpu="H100", timeout=3600, image=image, secrets=[modal.Secret.from_name("huggingface-secret")])
def bench_h100(max_samples: int = 100):
    return run_benchmark(max_samples)


@app.local_entrypoint()
def main(samples: int = 100, gpu: str = "a100"):
    import json
    from pathlib import Path

    print(f"\n  GPU={gpu.upper()}  samples={samples}")

    if gpu.lower() == "h100":
        result = bench_h100.remote(samples)
    else:
        result = bench_a100.remote(samples)

    out = Path(__file__).parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Local → {out / 'comparison_summary.json'}")
