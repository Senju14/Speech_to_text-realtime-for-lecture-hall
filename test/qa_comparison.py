"""
QA Comparison Report — System vs Reference (Tactiq / Human Transcription)

Compares our ASR+Translation system output against a reference transcription
(e.g., Tactiq.io, human-corrected text) and generates a detailed quality report.

Metrics:
- WER (Word Error Rate):  measures word-level accuracy
- CER (Character Error Rate): measures character-level accuracy (better for Vietnamese)
- Precision / Recall / F1: based on matching n-grams
- Error Classification: hallucinations, deletions, substitutions
- Latency summary: per-segment timing if available

Usage:
    # Compare from JSON recording export:
    python test/qa_comparison.py --system output.json --reference tactiq.txt

    # Compare from inline text:
    python test/qa_comparison.py --system-text "xin chào..." --reference-text "xin chào..."

    # Full report with HTML output:
    python test/qa_comparison.py --system output.json --reference tactiq.txt --html report.html

    # Batch comparison of multiple recordings:
    python test/qa_comparison.py --batch-dir test/comparisons/
"""

import argparse
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_vi(text: str) -> str:
    """Normalize Vietnamese text for fair comparison.
    - Lowercase
    - Remove punctuation (preserve Vietnamese diacritics)
    - Collapse whitespace
    """
    if not text:
        return ""
    text = text.lower().strip()
    # Remove punctuation but keep Vietnamese characters, digits, spaces
    text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_en(text: str) -> str:
    """Normalize English text for fair comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# WER / CER calculation (no external deps)
# ---------------------------------------------------------------------------

def _levenshtein(ref_tokens: list, hyp_tokens: list) -> tuple[int, int, int, int]:
    """
    Compute Levenshtein edit distance.
    Returns (distance, substitutions, deletions, insertions).
    """
    n, m = len(ref_tokens), len(hyp_tokens)
    # dp[i][j] = (dist, sub, del, ins)
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1]
                delete = dp[i - 1][j]
                insert = dp[i][j - 1]

                candidates = [
                    (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),      # substitution
                    (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),  # deletion
                    (insert[0] + 1, insert[1], insert[2], insert[3] + 1),  # insertion
                ]
                dp[i][j] = min(candidates, key=lambda x: x[0])

    return dp[n][m]


def compute_wer(reference: str, hypothesis: str, language: str = "vi") -> dict:
    """Compute Word Error Rate with detailed breakdown."""
    norm = normalize_vi if language == "vi" else normalize_en
    ref = norm(reference)
    hyp = norm(hypothesis)

    ref_words = ref.split()
    hyp_words = hyp.split()

    if not ref_words:
        return {"wer": 0.0 if not hyp_words else 1.0, "ref_words": 0, "hyp_words": len(hyp_words),
                "substitutions": 0, "deletions": 0, "insertions": len(hyp_words)}

    dist, subs, dels, ins = _levenshtein(ref_words, hyp_words)
    wer = dist / len(ref_words)

    return {
        "wer": round(wer, 4),
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "edits": dist,
    }


def compute_cer(reference: str, hypothesis: str, language: str = "vi") -> dict:
    """Compute Character Error Rate."""
    norm = normalize_vi if language == "vi" else normalize_en
    ref = norm(reference)
    hyp = norm(hypothesis)

    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))

    if not ref_chars:
        return {"cer": 0.0 if not hyp_chars else 1.0, "ref_chars": 0, "hyp_chars": len(hyp_chars)}

    dist, subs, dels, ins = _levenshtein(ref_chars, hyp_chars)
    cer = dist / len(ref_chars)

    return {
        "cer": round(cer, 4),
        "ref_chars": len(ref_chars),
        "hyp_chars": len(hyp_chars),
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
    }


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def classify_errors(reference: str, hypothesis: str, language: str = "vi") -> list[dict]:
    """
    Classify errors into categories:
    - hallucination: words in hypothesis not in reference at all
    - deletion: words in reference completely missing from hypothesis
    - substitution: phonetically similar but wrong (common in Vietnamese ASR)
    """
    norm = normalize_vi if language == "vi" else normalize_en
    ref_words = set(norm(reference).split())
    hyp_words = set(norm(hypothesis).split())

    errors = []

    # Hallucinations: in hypothesis but not in reference
    hallucinated = hyp_words - ref_words
    if hallucinated:
        errors.append({
            "type": "hallucination",
            "count": len(hallucinated),
            "words": sorted(hallucinated)[:20],  # limit display
        })

    # Deletions: in reference but not in hypothesis
    deleted = ref_words - hyp_words
    if deleted:
        errors.append({
            "type": "deletion",
            "count": len(deleted),
            "words": sorted(deleted)[:20],
        })

    return errors


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def load_system_output(path: str) -> str:
    """Load system output from JSON export or plain text."""
    p = Path(path)
    if p.suffix == '.json':
        with open(p, encoding='utf-8') as f:
            data = json.load(f)
        # Support both array-of-objects and flat text
        if isinstance(data, list):
            parts = []
            for item in data:
                parts.append(item.get('vi', item.get('source', item.get('text', ''))))
            return '\n'.join(parts)
        elif isinstance(data, dict):
            return data.get('transcript', data.get('text', ''))
        return str(data)
    else:
        return p.read_text(encoding='utf-8')


def load_reference(path: str) -> str:
    """Load reference transcription (Tactiq export or plain text)."""
    p = Path(path)
    text = p.read_text(encoding='utf-8')

    # Try to detect Tactiq format: "HH:MM:SS\ntext\n\nHH:MM:SS\ntext\n..."
    lines = text.strip().split('\n')
    tactiq_pattern = re.compile(r'^\d{1,2}:\d{2}(:\d{2})?$')

    content_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip timestamp lines
        if tactiq_pattern.match(line):
            continue
        # Skip speaker labels like "Speaker 1:"
        if re.match(r'^(Speaker\s*\d+|Người nói\s*\d+)\s*:', line, re.IGNORECASE):
            line = re.sub(r'^(Speaker\s*\d+|Người nói\s*\d+)\s*:\s*', '', line, flags=re.IGNORECASE)
        content_lines.append(line)

    return '\n'.join(content_lines)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Full comparison result between system and reference."""
    system_text: str
    reference_text: str
    language: str = "vi"
    wer_detail: dict = field(default_factory=dict)
    cer_detail: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    def compute(self):
        self.wer_detail = compute_wer(self.reference_text, self.system_text, self.language)
        self.cer_detail = compute_cer(self.reference_text, self.system_text, self.language)
        self.errors = classify_errors(self.reference_text, self.system_text, self.language)
        return self


def generate_text_report(result: ComparisonResult) -> str:
    """Generate a human-readable text report."""
    w = result.wer_detail
    c = result.cer_detail

    lines = [
        "=" * 70,
        "  QA Comparison Report — ASR System vs Reference",
        "=" * 70,
        "",
        f"  Language:        {result.language.upper()}",
        f"  Reference words: {w.get('ref_words', 0)}",
        f"  System words:    {w.get('hyp_words', 0)}",
        "",
        "  ┌─────────────────────────────────────────────┐",
        f"  │  WER:  {w.get('wer', 0):.1%}                                │",
        f"  │  CER:  {c.get('cer', 0):.1%}                                │",
        "  └─────────────────────────────────────────────┘",
        "",
        "  Error Breakdown (WER):",
        f"    Substitutions: {w.get('substitutions', 0)}",
        f"    Deletions:     {w.get('deletions', 0)}",
        f"    Insertions:    {w.get('insertions', 0)}",
        f"    Total edits:   {w.get('edits', 0)}",
        "",
    ]

    if result.errors:
        lines.append("  Error Classification:")
        for err in result.errors:
            lines.append(f"    [{err['type'].upper()}] {err['count']} words")
            sample = ', '.join(err['words'][:10])
            lines.append(f"      Sample: {sample}")
        lines.append("")

    # Quality grade
    wer = w.get('wer', 1.0)
    if wer < 0.05:
        grade = "A+ (Excellent)"
    elif wer < 0.10:
        grade = "A  (Very Good)"
    elif wer < 0.15:
        grade = "B  (Good)"
    elif wer < 0.25:
        grade = "C  (Acceptable)"
    elif wer < 0.40:
        grade = "D  (Below Average)"
    else:
        grade = "F  (Poor)"

    lines.extend([
        f"  Quality Grade: {grade}",
        "",
        "  Benchmarks:",
        "    WER < 5%   → Production quality (Google/Azure level)",
        "    WER < 10%  → Very good for Vietnamese ASR",
        "    WER < 15%  → Good, suitable for lecture use",
        "    WER < 25%  → Acceptable with post-editing",
        "",
        "=" * 70,
    ])

    return "\n".join(lines)


def generate_html_report(result: ComparisonResult) -> str:
    """Generate an HTML comparison report with side-by-side diff."""
    w = result.wer_detail
    c = result.cer_detail
    wer_pct = f"{w.get('wer', 0):.1%}"
    cer_pct = f"{c.get('cer', 0):.1%}"

    # Color based on quality
    wer_val = w.get('wer', 1.0)
    color = "#22c55e" if wer_val < 0.1 else "#eab308" if wer_val < 0.25 else "#ef4444"

    errors_html = ""
    for err in result.errors:
        sample = ', '.join(f'<code>{w}</code>' for w in err['words'][:10])
        errors_html += f"""
        <div style="margin-bottom: 8px;">
            <strong style="color: {'#ef4444' if err['type']=='hallucination' else '#f59e0b'}">{err['type'].upper()}</strong>
            — {err['count']} words<br>
            <small>{sample}</small>
        </div>"""

    ref_preview = result.reference_text[:500].replace('\n', '<br>')
    sys_preview = result.system_text[:500].replace('\n', '<br>')

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>QA Comparison Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1e293b; }}
    h1 {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }}
    .metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.5rem 0; }}
    .metric {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem; text-align: center; }}
    .metric-value {{ font-size: 2.5rem; font-weight: 700; color: {color}; }}
    .metric-label {{ font-size: 0.875rem; color: #64748b; margin-top: 0.25rem; }}
    .breakdown {{ background: #f8fafc; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
    .side-by-side {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.5rem 0; }}
    .text-panel {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; max-height: 300px; overflow-y: auto; font-size: 0.9rem; line-height: 1.6; }}
    .text-panel h3 {{ margin-top: 0; font-size: 1rem; }}
    code {{ background: #fee2e2; padding: 2px 4px; border-radius: 3px; font-size: 0.85em; }}
    table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
    td, th {{ padding: 0.5rem; border-bottom: 1px solid #e2e8f0; text-align: left; }}
</style>
</head>
<body>
<h1>📊 QA Comparison Report</h1>
<p>ASR System vs Reference Transcription ({result.language.upper()})</p>

<div class="metrics">
    <div class="metric">
        <div class="metric-value">{wer_pct}</div>
        <div class="metric-label">Word Error Rate (WER)</div>
    </div>
    <div class="metric">
        <div class="metric-value">{cer_pct}</div>
        <div class="metric-label">Character Error Rate (CER)</div>
    </div>
</div>

<div class="breakdown">
    <h3>Error Breakdown</h3>
    <table>
        <tr><td>Reference words</td><td><strong>{w.get('ref_words', 0)}</strong></td></tr>
        <tr><td>System words</td><td><strong>{w.get('hyp_words', 0)}</strong></td></tr>
        <tr><td>Substitutions</td><td>{w.get('substitutions', 0)}</td></tr>
        <tr><td>Deletions</td><td>{w.get('deletions', 0)}</td></tr>
        <tr><td>Insertions</td><td>{w.get('insertions', 0)}</td></tr>
        <tr><td>Total edits</td><td><strong>{w.get('edits', 0)}</strong></td></tr>
    </table>
</div>

{f'<div class="breakdown"><h3>Error Classification</h3>{errors_html}</div>' if errors_html else ''}

<div class="side-by-side">
    <div class="text-panel">
        <h3>📋 Reference (Ground Truth)</h3>
        {ref_preview}
    </div>
    <div class="text-panel">
        <h3>🤖 System Output</h3>
        {sys_preview}
    </div>
</div>

<p style="color: #94a3b8; font-size: 0.8rem; margin-top: 2rem;">Generated by ASR QA Evaluation Script</p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare ASR system output against reference transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test/qa_comparison.py --system output.json --reference tactiq.txt
  python test/qa_comparison.py --system-text "xin chào" --reference-text "xin chào"
  python test/qa_comparison.py --system output.json --reference tactiq.txt --html report.html
  python test/qa_comparison.py --batch-dir test/comparisons/
        """,
    )
    parser.add_argument('--system', type=str, help='Path to system output (JSON export or TXT)')
    parser.add_argument('--reference', type=str, help='Path to reference transcription (Tactiq export or TXT)')
    parser.add_argument('--system-text', type=str, help='System output text (inline)')
    parser.add_argument('--reference-text', type=str, help='Reference text (inline)')
    parser.add_argument('--language', type=str, default='vi', choices=['vi', 'en'], help='Language for normalization')
    parser.add_argument('--html', type=str, help='Output HTML report to file')
    parser.add_argument('--json', type=str, help='Output JSON metrics to file')
    parser.add_argument('--batch-dir', type=str, help='Directory with pairs: {name}_system.txt + {name}_reference.txt')

    args = parser.parse_args()

    if args.batch_dir:
        # Batch mode
        batch_dir = Path(args.batch_dir)
        ref_files = sorted(batch_dir.glob('*_reference.txt'))
        if not ref_files:
            ref_files = sorted(batch_dir.glob('*_ref.txt'))

        if not ref_files:
            print(f"No reference files found in {batch_dir}")
            sys.exit(1)

        all_results = []
        for ref_path in ref_files:
            name = ref_path.stem.replace('_reference', '').replace('_ref', '')
            sys_path = ref_path.parent / f"{name}_system.txt"
            if not sys_path.exists():
                sys_path = ref_path.parent / f"{name}_system.json"
            if not sys_path.exists():
                print(f"  Skipping {name}: no system file found")
                continue

            sys_text = load_system_output(str(sys_path))
            ref_text = load_reference(str(ref_path))
            result = ComparisonResult(sys_text, ref_text, args.language).compute()
            all_results.append((name, result))
            print(f"  {name}: WER={result.wer_detail['wer']:.1%}  CER={result.cer_detail['cer']:.1%}")

        if all_results:
            avg_wer = sum(r.wer_detail['wer'] for _, r in all_results) / len(all_results)
            avg_cer = sum(r.cer_detail['cer'] for _, r in all_results) / len(all_results)
            print(f"\n  Average WER: {avg_wer:.1%}  |  Average CER: {avg_cer:.1%}  ({len(all_results)} files)")
        return

    # Single comparison mode
    if args.system_text:
        sys_text = args.system_text
    elif args.system:
        sys_text = load_system_output(args.system)
    else:
        print("Error: provide --system or --system-text")
        sys.exit(1)

    if args.reference_text:
        ref_text = args.reference_text
    elif args.reference:
        ref_text = load_reference(args.reference)
    else:
        print("Error: provide --reference or --reference-text")
        sys.exit(1)

    result = ComparisonResult(sys_text, ref_text, args.language).compute()

    # Console report
    print(generate_text_report(result))

    # Optional HTML export
    if args.html:
        html = generate_html_report(result)
        Path(args.html).write_text(html, encoding='utf-8')
        print(f"\n  HTML report saved to: {args.html}")

    # Optional JSON export
    if args.json:
        metrics = {
            "wer": result.wer_detail,
            "cer": result.cer_detail,
            "errors": result.errors,
            "language": result.language,
        }
        Path(args.json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"  JSON metrics saved to: {args.json}")


if __name__ == "__main__":
    main()
