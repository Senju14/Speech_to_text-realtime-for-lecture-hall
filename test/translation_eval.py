"""
Translation Quality Evaluation — NLLB + BARTpho for Thesis

Evaluates the translation pipeline (vi→en, en→vi) using standard
MT metrics: BLEU, chrF++, and optionally COMET.

═══════════════════════════════════════════════════════════════════════
Datasets:
═══════════════════════════════════════════════════════════════════════
  - PhoMT (vi↔en)         : Vietnamese-English parallel corpus (research benchmark)
  - IWSLT 2015 (vi→en)    : TED Talk translations
  - FLORES-200 devtest     : Multi-language MT benchmark (includes vi, en)

═══════════════════════════════════════════════════════════════════════
Metrics:
═══════════════════════════════════════════════════════════════════════
  - BLEU    : Bilingual Evaluation Understudy (n-gram precision, 0-100)
  - chrF++  : Character n-gram F-score (better for morphologically rich langs)
  - TER     : Translation Edit Rate (lower = better)
  - Latency : Translation time per sentence (ms)

═══════════════════════════════════════════════════════════════════════
Ablation Configs:
═══════════════════════════════════════════════════════════════════════
  T0: NLLB distilled-600M (baseline, fast)
  T1: NLLB 3.3B float16 (full precision)
  T2: NLLB 3.3B 8-bit quantized (our production config)
  T3: NLLB 3.3B 8-bit + BARTpho pre-correction (full pipeline)

═══════════════════════════════════════════════════════════════════════
Usage:
═══════════════════════════════════════════════════════════════════════
  modal run test/translation_eval.py --config T2 --dataset flores --samples 500
  modal run test/translation_eval.py --config ALL --dataset ALL --samples 500

Results → test/results/translation/{config}/{dataset}.json
"""

import modal
import json
import time
import re
import numpy as np
from dataclasses import dataclass, asdict

app = modal.App("translation-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.35.0", "accelerate>=0.25.0", "sentencepiece>=0.1.99",
    )
    .pip_install("peft>=0.8.0", "bitsandbytes>=0.41.0")
    .pip_install(
        "datasets>=2.14.0", "sacrebleu>=2.3.0", "numpy>=1.24.0",
        "huggingface_hub>=0.20.0",
    )
    .add_local_dir("src", remote_path="/root/src", copy=True)
)

# ─── Datasets ─────────────────────────────────────────────────────────────────

DATASETS = {
    "phomt": {
        "hf_name": "vinai/PhoMT_detokenization",
        "split": "test",
        "src_col": "vi",
        "tgt_col": "en",
        "src_lang": "vie_Latn",
        "tgt_lang": "eng_Latn",
        "description": "PhoMT Vietnamese-English parallel corpus",
    },
    "flores_vi_en": {
        "hf_name": "facebook/flores",
        "config": "vie_Latn-eng_Latn",
        "split": "devtest",
        "src_col": "sentence_vie_Latn",
        "tgt_col": "sentence_eng_Latn",
        "src_lang": "vie_Latn",
        "tgt_lang": "eng_Latn",
        "description": "FLORES-200 devtest Vietnamese→English",
    },
    "flores_en_vi": {
        "hf_name": "facebook/flores",
        "config": "eng_Latn-vie_Latn",
        "split": "devtest",
        "src_col": "sentence_eng_Latn",
        "tgt_col": "sentence_vie_Latn",
        "src_lang": "eng_Latn",
        "tgt_lang": "vie_Latn",
        "description": "FLORES-200 devtest English→Vietnamese",
    },
}

# ─── Translation Configs ──────────────────────────────────────────────────────

@dataclass
class TranslationConfig:
    name: str
    description: str
    model_name: str
    use_8bit: bool = False
    use_bartpho_pre: bool = False  # BARTpho pre-correction before translation

TRANSLATION_CONFIGS = {
    "T0": TranslationConfig(
        name="T0_nllb_600M",
        description="NLLB distilled-600M (baseline, fast)",
        model_name="facebook/nllb-200-distilled-600M",
    ),
    "T1": TranslationConfig(
        name="T1_nllb_3.3B_fp16",
        description="NLLB 3.3B float16 (full precision)",
        model_name="facebook/nllb-200-3.3B",
    ),
    "T2": TranslationConfig(
        name="T2_nllb_3.3B_8bit",
        description="NLLB 3.3B 8-bit quantized (production)",
        model_name="facebook/nllb-200-3.3B",
        use_8bit=True,
    ),
    "T3": TranslationConfig(
        name="T3_nllb_3.3B_8bit_bartpho",
        description="NLLB 3.3B 8-bit + BARTpho pre-correction",
        model_name="facebook/nllb-200-3.3B",
        use_8bit=True,
        use_bartpho_pre=True,
    ),
}


# ─── Core Evaluation ──────────────────────────────────────────────────────────

def run_translation_eval(
    trans_config: TranslationConfig,
    dataset_name: str,
    max_samples: int,
):
    """Run translation evaluation for one config × one dataset."""
    import sys, os
    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/cache/huggingface"

    import sacrebleu

    config = DATASETS[dataset_name]

    print(f"\n{'='*70}")
    print(f"  TRANSLATION EVAL: {trans_config.name}")
    print(f"  {trans_config.description}")
    print(f"  Dataset: {dataset_name} ({config['description']})")
    print(f"  Direction: {config['src_lang']} → {config['tgt_lang']}")
    print(f"{'='*70}")

    # Load translator
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    print(f"[Model] Loading {trans_config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        trans_config.model_name,
        cache_dir="/cache/nllb",
        src_lang=config["src_lang"],
    )

    if trans_config.use_8bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            trans_config.model_name,
            cache_dir="/cache/nllb",
            quantization_config=quant_config,
            device_map="auto",
        )
        print(f"  Loaded with 8-bit quantization")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            trans_config.model_name,
            cache_dir="/cache/nllb",
            torch_dtype=torch.float16,
        ).to("cuda")
        print(f"  Loaded in float16")

    # Load BARTpho (optional)
    corrector = None
    if trans_config.use_bartpho_pre and config["src_lang"] == "vie_Latn":
        from src.postprocess import BARTphoCorrector
        from src.config import BARTPHO_ADAPTER, BARTPHO_DEVICE
        print("[Model] Loading BARTpho for pre-correction...")
        corrector = BARTphoCorrector(
            adapter_id=BARTPHO_ADAPTER,
            device=BARTPHO_DEVICE,
            cache_dir="/cache/huggingface",
        )
        corrector.load_model()
        print("  BARTpho loaded")

    # Load dataset
    print(f"\n[Data] Loading {dataset_name}...")
    pairs = load_parallel_data(config, max_samples)
    print(f"[Data] Loaded {len(pairs)} sentence pairs")

    if not pairs:
        return {"error": "No data loaded"}

    # Translate
    print(f"\n[Eval] Translating {len(pairs)} sentences...")
    sources = [p["src"] for p in pairs]
    references = [p["tgt"] for p in pairs]
    hypotheses = []
    latencies = []

    tgt_lang_id = tokenizer.convert_tokens_to_ids(config["tgt_lang"])

    for idx, src_text in enumerate(sources):
        # Optional BARTpho pre-correction
        if corrector and corrector.is_loaded:
            src_text = corrector.correct(src_text) or src_text

        t0 = time.time()
        tokenizer.src_lang = config["src_lang"]
        inputs = tokenizer(src_text, return_tensors="pt", max_length=256, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_new_tokens=256,
                num_beams=4,
            )

        translation = tokenizer.decode(generated[0], skip_special_tokens=True)
        latency = time.time() - t0

        hypotheses.append(translation)
        latencies.append(latency)

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(sources)}] avg latency: {np.mean(latencies)*1000:.0f}ms")

    # Compute metrics
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)  # chrF++
    ter = sacrebleu.corpus_ter(hypotheses, [references])

    result = {
        "config": trans_config.name,
        "config_desc": trans_config.description,
        "config_flags": asdict(trans_config),
        "dataset": dataset_name,
        "dataset_desc": config["description"],
        "direction": f"{config['src_lang']} → {config['tgt_lang']}",
        "num_samples": len(hypotheses),
        # MT metrics
        "bleu": round(bleu.score, 2),
        "chrf_pp": round(chrf.score, 2),
        "ter": round(ter.score, 2),
        # Latency
        "avg_latency_ms": round(np.mean(latencies) * 1000, 1),
        "p50_latency_ms": round(np.percentile(latencies, 50) * 1000, 1),
        "p90_latency_ms": round(np.percentile(latencies, 90) * 1000, 1),
        # Detailed BLEU
        "bleu_1": round(bleu.precisions[0], 2),
        "bleu_2": round(bleu.precisions[1], 2),
        "bleu_3": round(bleu.precisions[2], 2),
        "bleu_4": round(bleu.precisions[3], 2),
        "brevity_penalty": round(bleu.bp, 4),
    }

    # Print summary
    print(f"\n{'='*70}")
    print(f"  RESULTS: {trans_config.name} on {dataset_name}")
    print(f"{'='*70}")
    print(f"  BLEU:  {result['bleu']:.2f}  |  chrF++: {result['chrf_pp']:.2f}  |  TER: {result['ter']:.2f}")
    print(f"  Avg latency: {result['avg_latency_ms']:.0f}ms  |  P90: {result['p90_latency_ms']:.0f}ms")
    print(f"  BLEU n-grams: {result['bleu_1']:.1f}/{result['bleu_2']:.1f}/{result['bleu_3']:.1f}/{result['bleu_4']:.1f}")
    print(f"{'='*70}")

    # Print some examples
    print("\n--- Sample translations ---")
    for i in range(min(5, len(sources))):
        print(f"  SRC: {sources[i][:80]}")
        print(f"  REF: {references[i][:80]}")
        print(f"  HYP: {hypotheses[i][:80]}")
        print()

    return result


def load_parallel_data(config: dict, max_samples: int):
    """Load parallel sentence pairs from HuggingFace dataset."""
    from datasets import load_dataset

    load_kwargs = {"split": config["split"], "streaming": True}
    if config.get("config"):
        load_kwargs["name"] = config["config"]

    ds = load_dataset(config["hf_name"], **load_kwargs, trust_remote_code=True)

    pairs = []
    src_col = config["src_col"]
    tgt_col = config["tgt_col"]

    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        try:
            # PhoMT has nested structure: {"en": "...", "vi": "..."}
            if isinstance(item.get("translation"), dict):
                src = str(item["translation"].get(src_col, "")).strip()
                tgt = str(item["translation"].get(tgt_col, "")).strip()
            else:
                src = str(item.get(src_col, "")).strip()
                tgt = str(item.get(tgt_col, "")).strip()

            if src and tgt and len(src) > 3 and len(tgt) > 3:
                pairs.append({"src": src, "tgt": tgt})
        except Exception:
            continue

    return pairs


# ─── Modal Functions ──────────────────────────────────────────────────────────

@app.function(
    gpu="A100",
    timeout=7200,
    memory=40960,
    image=image,
    volumes={"/cache": modal.Volume.from_name("asr-model-cache", create_if_missing=True)},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def evaluate_translation(config_name: str, dataset_name: str, max_samples: int = 500):
    """Run translation evaluation on A100."""
    import sys, os
    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = "/cache/huggingface"

    trans_config = TRANSLATION_CONFIGS[config_name]
    return run_translation_eval(trans_config, dataset_name, max_samples)


@app.local_entrypoint()
def main(
    config: str = "ALL",
    dataset: str = "flores_vi_en",
    samples: int = 500,
):
    """
    Run translation evaluation.

    Args:
        config: T0-T3 or ALL
        dataset: Dataset name or ALL
        samples: Number of sentence pairs
    """
    from pathlib import Path

    configs = list(TRANSLATION_CONFIGS.keys()) if config == "ALL" else [config]
    datasets = list(DATASETS.keys()) if dataset == "ALL" else [dataset]

    print(f"\n{'#'*70}")
    print(f"# TRANSLATION EVALUATION")
    print(f"# Configs: {configs}")
    print(f"# Datasets: {datasets}")
    print(f"# Samples: {samples}")
    print(f"{'#'*70}")

    all_results = []

    for cfg in configs:
        for ds in datasets:
            print(f"\n>>> Running {cfg} on {ds}...")
            try:
                result = evaluate_translation.remote(cfg, ds, samples)

                results_dir = Path(__file__).parent / "results" / "translation" / cfg
                results_dir.mkdir(parents=True, exist_ok=True)
                with open(results_dir / f"{ds}.json", "w") as f:
                    json.dump(result, f, indent=2)
                all_results.append(result)
            except Exception as e:
                print(f"    ERROR: {e}")
                all_results.append({"config": cfg, "dataset": ds, "error": str(e)})

    # Combined
    combined_file = Path(__file__).parent / "results" / "translation" / "combined.json"
    combined_file.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n\n{'='*90}")
    print("TRANSLATION EVALUATION SUMMARY")
    print(f"{'='*90}")
    print(f"{'Config':<25} {'Dataset':<18} {'BLEU':>7} {'chrF++':>7} {'TER':>7} {'Lat(ms)':>8}")
    print(f"{'-'*90}")
    for r in all_results:
        if "error" in r:
            print(f"{r.get('config','?'):<25} {r.get('dataset','?'):<18} {'ERROR':>7}")
        else:
            print(
                f"{r['config']:<25} {r['dataset']:<18} "
                f"{r['bleu']:>6.2f} {r['chrf_pp']:>7.2f} {r['ter']:>6.2f} "
                f"{r['avg_latency_ms']:>7.0f}"
            )
    print(f"{'='*90}")
