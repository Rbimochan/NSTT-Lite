"""Minimal, model-agnostic WER/CER computation (jiwer-based)."""
from __future__ import annotations

import jiwer


def compute_wer_cer(references: list[str], hypotheses: list[str]) -> tuple[float, float]:
    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)
    return wer, cer
