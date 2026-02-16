"""
Generate LaTeX / Markdown Tables from Evaluation Results

Reads JSON results from ablation and translation evaluations,
generates publication-ready tables for thesis.

Usage:
    python test/generate_tables.py                    # All tables, Markdown
    python test/generate_tables.py --latex             # All tables, LaTeX
    python test/generate_tables.py --ablation          # Ablation only
    python test/generate_tables.py --translation       # Translation only
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"
TABLES_DIR = Path(__file__).parent / "tables"


def load_json_results(subdir: str):
    """Load combined.json from a results subdirectory."""
    combined = RESULTS_DIR / subdir / "combined.json"
    if combined.exists():
        with open(combined) as f:
            return json.load(f)

    # Fallback: load individual files
    results = []
    base = RESULTS_DIR / subdir
    if base.exists():
        for config_dir in sorted(base.iterdir()):
            if config_dir.is_dir():
                for f in sorted(config_dir.glob("*.json")):
                    with open(f) as fh:
                        results.append(json.load(fh))
    return results


# ─── Ablation Tables ──────────────────────────────────────────────────────────

def generate_ablation_markdown(results: list) -> str:
    lines = []
    lines.append("# Ablation Study Results")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Group by dataset
    datasets = {}
    for r in results:
        if "error" in r:
            continue
        ds = r.get("dataset", "unknown")
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(r)

    for ds_name, ds_results in datasets.items():
        desc = ds_results[0].get("dataset_desc", ds_name) if ds_results else ds_name
        lines.append(f"\n## {desc}")
        lines.append("")
        lines.append("| Config | Description | WER↓ | CER↓ | TTFT↓ | RTF↓ | FLK↓ | HAL↓ |")
        lines.append("|--------|-------------|------|------|-------|------|------|------|")

        for r in sorted(ds_results, key=lambda x: x.get("ablation", "")):
            name = r.get("ablation", "?")
            desc_short = r.get("ablation_desc", "")[:45]
            wer = f"{r['wer']*100:.2f}%"
            cer = f"{r['cer']*100:.2f}%"
            ttft = f"{r['avg_ttft_ms']:.0f}ms"
            rtf = f"{r['avg_rtf']:.3f}x"
            flk = f"{r['flicker_rate']*100:.1f}%"
            hal = f"{r['hallucination_rate']*100:.1f}%"
            lines.append(f"| {name} | {desc_short} | {wer} | {cer} | {ttft} | {rtf} | {flk} | {hal} |")

    # Delta improvement table
    lines.append("\n\n## Component Contribution (Δ from previous)")
    lines.append("")

    for ds_name, ds_results in datasets.items():
        sorted_results = sorted(ds_results, key=lambda x: x.get("ablation", ""))
        if len(sorted_results) < 2:
            continue

        lines.append(f"\n### {ds_name}")
        lines.append("")
        lines.append("| Component Added | ΔWER | ΔCER | ΔTTFT | ΔFLK |")
        lines.append("|----------------|------|------|-------|------|")

        for i in range(1, len(sorted_results)):
            prev = sorted_results[i - 1]
            curr = sorted_results[i]
            name = curr.get("ablation_desc", "").replace("+", "").strip()[:40]
            dwer = (curr["wer"] - prev["wer"]) * 100
            dcer = (curr["cer"] - prev["cer"]) * 100
            dttft = curr["avg_ttft_ms"] - prev["avg_ttft_ms"]
            dflk = (curr["flicker_rate"] - prev["flicker_rate"]) * 100

            sign_wer = "+" if dwer > 0 else ""
            sign_cer = "+" if dcer > 0 else ""
            sign_ttft = "+" if dttft > 0 else ""
            sign_flk = "+" if dflk > 0 else ""

            lines.append(
                f"| {name} | {sign_wer}{dwer:.2f}% | {sign_cer}{dcer:.2f}% | "
                f"{sign_ttft}{dttft:.0f}ms | {sign_flk}{dflk:.1f}% |"
            )

    return "\n".join(lines)


def generate_ablation_latex(results: list) -> str:
    lines = []
    lines.append("% Ablation Study Results")
    lines.append(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    datasets = {}
    for r in results:
        if "error" in r:
            continue
        ds = r.get("dataset", "unknown")
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(r)

    for ds_name, ds_results in datasets.items():
        desc = ds_results[0].get("dataset_desc", ds_name) if ds_results else ds_name
        lines.append(f"% Dataset: {desc}")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Ablation study on {desc}}}")
        lines.append(f"\\label{{tab:ablation_{ds_name}}}")
        lines.append("\\begin{tabular}{llrrrrrr}")
        lines.append("\\toprule")
        lines.append("Config & Description & WER$\\downarrow$ & CER$\\downarrow$ & TTFT$\\downarrow$ & RTF$\\downarrow$ & FLK$\\downarrow$ & HAL$\\downarrow$ \\\\")
        lines.append("\\midrule")

        sorted_results = sorted(ds_results, key=lambda x: x.get("ablation", ""))
        for r in sorted_results:
            name = r.get("ablation", "?").replace("_", "\\_")
            desc_short = r.get("ablation_desc", "")[:35].replace("_", "\\_")
            wer = f"{r['wer']*100:.2f}\\%"
            cer = f"{r['cer']*100:.2f}\\%"
            ttft = f"{r['avg_ttft_ms']:.0f}"
            rtf = f"{r['avg_rtf']:.3f}"
            flk = f"{r['flicker_rate']*100:.1f}\\%"
            hal = f"{r['hallucination_rate']*100:.1f}\\%"

            if r == sorted_results[-1]:
                lines.append(f"\\textbf{{{name}}} & \\textbf{{{desc_short}}} & \\textbf{{{wer}}} & \\textbf{{{cer}}} & \\textbf{{{ttft}}} & \\textbf{{{rtf}}} & \\textbf{{{flk}}} & \\textbf{{{hal}}} \\\\")
            else:
                lines.append(f"{name} & {desc_short} & {wer} & {cer} & {ttft} & {rtf} & {flk} & {hal} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    return "\n".join(lines)


# ─── Translation Tables ──────────────────────────────────────────────────────

def generate_translation_markdown(results: list) -> str:
    lines = []
    lines.append("# Translation Evaluation Results")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("| Config | Dataset | Direction | BLEU↑ | chrF++↑ | TER↓ | Lat(ms)↓ |")
    lines.append("|--------|---------|-----------|-------|---------|------|----------|")

    for r in sorted(results, key=lambda x: (x.get("config", ""), x.get("dataset", ""))):
        if "error" in r:
            continue
        lines.append(
            f"| {r['config']} | {r['dataset']} | {r.get('direction','')} | "
            f"{r['bleu']:.2f} | {r['chrf_pp']:.2f} | {r['ter']:.2f} | "
            f"{r['avg_latency_ms']:.0f} |"
        )

    lines.append("\n\n## BLEU N-gram Breakdown")
    lines.append("")
    lines.append("| Config | Dataset | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | BP |")
    lines.append("|--------|---------|--------|--------|--------|--------|-----|")

    for r in sorted(results, key=lambda x: (x.get("config", ""), x.get("dataset", ""))):
        if "error" in r:
            continue
        lines.append(
            f"| {r['config']} | {r['dataset']} | "
            f"{r.get('bleu_1', 0):.1f} | {r.get('bleu_2', 0):.1f} | "
            f"{r.get('bleu_3', 0):.1f} | {r.get('bleu_4', 0):.1f} | "
            f"{r.get('brevity_penalty', 0):.3f} |"
        )

    return "\n".join(lines)


def generate_translation_latex(results: list) -> str:
    lines = []
    lines.append("% Translation Evaluation Results")
    lines.append(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Translation quality comparison across NLLB configurations}")
    lines.append("\\label{tab:translation}")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("Config & Dataset & BLEU$\\uparrow$ & chrF++$\\uparrow$ & TER$\\downarrow$ & Latency (ms) \\\\")
    lines.append("\\midrule")

    sorted_results = sorted(results, key=lambda x: (x.get("dataset", ""), x.get("config", "")))
    for r in sorted_results:
        if "error" in r:
            continue
        name = r['config'].replace("_", "\\_")
        ds = r['dataset'].replace("_", "\\_")
        lines.append(
            f"{name} & {ds} & {r['bleu']:.2f} & {r['chrf_pp']:.2f} & "
            f"{r['ter']:.2f} & {r['avg_latency_ms']:.0f} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation tables")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX instead of Markdown")
    parser.add_argument("--ablation", action="store_true", help="Ablation tables only")
    parser.add_argument("--translation", action="store_true", help="Translation tables only")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    do_all = not args.ablation and not args.translation

    ext = "tex" if args.latex else "md"

    if do_all or args.ablation:
        ablation_results = load_json_results("ablation")
        if ablation_results:
            if args.latex:
                content = generate_ablation_latex(ablation_results)
            else:
                content = generate_ablation_markdown(ablation_results)
            outfile = TABLES_DIR / f"ablation.{ext}"
            outfile.write_text(content, encoding="utf-8")
            print(f"Saved: {outfile}")
        else:
            print("No ablation results found in test/results/ablation/")

    if do_all or args.translation:
        trans_results = load_json_results("translation")
        if trans_results:
            if args.latex:
                content = generate_translation_latex(trans_results)
            else:
                content = generate_translation_markdown(trans_results)
            outfile = TABLES_DIR / f"translation.{ext}"
            outfile.write_text(content, encoding="utf-8")
            print(f"Saved: {outfile}")
        else:
            print("No translation results found in test/results/translation/")


if __name__ == "__main__":
    main()
    lines.append("- CER: Character Error Rate (lower = better)")
    lines.append("- TTFT: Time to First Token (lower = better)")
    lines.append("- RTF: Real-Time Factor (<1.0 = faster than real-time)")
    
    return "\n".join(lines)


def generate_latex(results: dict):
    """Generate LaTeX table."""
    lines = []
    lines.append("% Streaming ASR Evaluation Results")
    lines.append(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Streaming ASR Evaluation Results}")
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\hline")
    lines.append("Model & Dataset & GPU & WER & CER & TTFT & RTF \\\\")
    lines.append("\\hline")
    
    for model_name, model_label in [("whisper", "Whisper"), ("phowhisper", "PhoWhisper")]:
        for gpu in ["a100", "h100"]:
            for ds, data in sorted(results[model_name][gpu].items()):
                wer = f"{data.get('wer', 0)*100:.2f}\\%"
                cer = f"{data.get('cer', 0)*100:.2f}\\%"
                ttft = f"{data.get('avg_ttft_ms', 0):.0f}ms"
                rtf = f"{data.get('avg_rtf', 0):.3f}x"
                lines.append(f"{model_label} & {ds} & {gpu.upper()} & {wer} & {cer} & {ttft} & {rtf} \\\\")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def main():
    import sys
    
    format = "latex" if "--latex" in sys.argv else "markdown"
    
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    results = load_results()
    
    if format == "latex":
        content = generate_latex(results)
        ext = "tex"
    else:
        content = generate_markdown(results)
        ext = "md"
    
    print(content)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = TABLES_DIR / f"results_{timestamp}.{ext}"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\n\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
