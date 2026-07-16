"""Error analysis and report artifacts for NSTT.

Uses jiwer character alignment for substitution detection.
Adapted from jiwer documentation: https://github.com/jitsi/jiwer
(Cite in academic work per start.md §7.)
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import jiwer

from src.evaluation import (
    UtterancePrediction,
    collect_manifest_predictions,
    load_checkpoint,
    normalize_pair_for_scoring,
)

# Devanagari Unicode block
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")

ERROR_CATEGORIES = [
    "phonetic_confusion",
    "oov_rare_vocabulary",
    "noise_degradation",
    "code_switching",
    "dialect_accent",
    "other",
]


@dataclass
class CategorizedSample:
    utterance_id: str
    reference: str
    hypothesis: str
    categories: list[str]
    speaker_id: str = ""
    duration_s: float = 0.0


def _has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text))


def categorize_errors(
    reference: str,
    hypothesis: str,
    *,
    duration_s: float = 0.0,
) -> list[str]:
    """Heuristic error tagging per start.md §3.3 checklist."""
    ref, hyp = normalize_pair_for_scoring(reference, hypothesis)
    categories: list[str] = []

    if LATIN_RE.search(ref):
        categories.append("code_switching")

    if not hyp.strip():
        categories.append("noise_degradation")
    elif len(ref) > 0 and jiwer.wer([ref], [hyp]) >= 0.95:
        categories.append("noise_degradation")

    ref_words = set(ref.split())
    hyp_words = set(hyp.split())
    missing_words = ref_words - hyp_words
    if missing_words and any(len(w) >= 4 for w in missing_words):
        categories.append("oov_rare_vocabulary")

    if _has_devanagari(ref) and ref != hyp:
        alignment = jiwer.process_characters(ref, hyp)
        if alignment.substitutions > 0:
            categories.append("phonetic_confusion")
        if alignment.deletions > len(ref) * 0.3:
            categories.append("dialect_accent")

    if not categories:
        if ref != hyp:
            categories.append("other")
        else:
            categories.append("other")

    return sorted(set(categories))


def categorize_predictions(
    predictions: list[UtterancePrediction],
) -> list[CategorizedSample]:
    return [
        CategorizedSample(
            utterance_id=p.utterance_id,
            reference=p.reference,
            hypothesis=p.hypothesis,
            categories=categorize_errors(
                p.reference, p.hypothesis, duration_s=p.duration_s
            ),
            speaker_id=p.speaker_id,
            duration_s=p.duration_s,
        )
        for p in predictions
    ]


def summarize_categories(samples: list[CategorizedSample]) -> Counter:
    counts: Counter = Counter()
    for sample in samples:
        for cat in sample.categories:
            counts[cat] += 1
    return counts


def export_error_samples(
    samples: list[CategorizedSample],
    reports_dir: Path,
) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "error_samples.csv"
    md_path = reports_dir / "error_samples.md"

    rows = [
        {
            "utterance_id": s.utterance_id,
            "speaker_id": s.speaker_id,
            "duration_s": s.duration_s,
            "reference": s.reference,
            "hypothesis": s.hypothesis,
            "categories": ";".join(s.categories),
        }
        for s in samples
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Error Analysis — Qualitative Samples",
        "",
        "> **Note:** Samples from the T-003 smoke checkpoint (3 training steps). "
        "Categories illustrate the analysis pipeline, not a trained model's error profile.",
        "",
        "| ID | Speaker | Duration | Categories | Reference | Hypothesis |",
        "|----|---------|----------|------------|-----------|------------|",
    ]
    for s in samples:
        cats = ", ".join(s.categories)
        lines.append(
            f"| {s.utterance_id} | {s.speaker_id} | {s.duration_s:.1f}s | "
            f"{cats} | {s.reference} | {s.hypothesis} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def export_error_categories(
    counts: Counter,
    reports_dir: Path,
    *,
    total_samples: int,
) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / "error_categories.md"
    csv_path = reports_dir / "error_categories.csv"

    rows = [
        {"category": cat, "count": counts.get(cat, 0)}
        for cat in ERROR_CATEGORIES
        if counts.get(cat, 0) > 0 or cat in counts
    ]
    if not rows:
        rows = [{"category": "other", "count": 0}]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "count"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Error Category Breakdown",
        "",
        f"Total utterances analyzed: **{total_samples}**",
        "",
        "> Smoke-checkpoint caveat: expect degenerate/high-error categories until a full Colab training run.",
        "",
        "| Category | Count | % of utterances |",
        "|----------|-------|-----------------|",
    ]
    for row in rows:
        pct = (row["count"] / total_samples * 100) if total_samples else 0
        label = row["category"].replace("_", " ").title()
        lines.append(f"| {label} | {row['count']} | {pct:.1f}% |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        png_path = reports_dir / "error_categories.png"
        labels = [r["category"].replace("_", "\n") for r in rows]
        values = [r["count"] for r in rows]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, values, color="#4C72B0")
        plt.title("Error Category Counts (NSTT)")
        plt.ylabel("Utterances tagged")
        plt.tight_layout()
        plt.savefig(png_path, dpi=120)
        plt.close()
    except ImportError:
        png_path = None

    return csv_path, md_path


def run_error_analysis(
    project_root: Path,
    checkpoint_dir: Path,
    *,
    device: str = "cpu",
    max_samples: int | None = None,
) -> dict:
    model, processor = load_checkpoint(checkpoint_dir, device=device)
    manifest_path = project_root / "data" / "manifests" / "test.jsonl"

    predictions = collect_manifest_predictions(
        model, processor, manifest_path, project_root, device, max_samples=max_samples
    )
    samples = categorize_predictions(predictions)
    counts = summarize_categories(samples)
    reports_dir = project_root / "reports"

    sample_paths = export_error_samples(samples, reports_dir)
    category_paths = export_error_categories(
        counts, reports_dir, total_samples=len(samples)
    )

    return {
        "num_samples": len(samples),
        "category_counts": dict(counts),
        "sample_paths": sample_paths,
        "category_paths": category_paths,
    }
