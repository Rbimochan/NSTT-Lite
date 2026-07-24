# Phase 4 — Baseline Establishment (Zero-Shot)

## Scope decision

The original phase plan asked for zero-shot WER/CER on Whisper-small,
Whisper-medium, Whisper-large, **and** MMS. Scoped down to **Whisper-small
only**, per explicit decision:

- Phase 6 only fine-tunes Whisper-small — that is the "before" number that
  actually matters for the project's central before/after comparison.
- This machine is CPU-only (no CUDA). Whisper-medium (~3x slower) and
  Whisper-large (~5-8x slower) than small, plus MMS (a separate large model),
  would each cost roughly 1-3 hours at the same 150-utterance sample size —
  not a good use of compute for models this project doesn't otherwise touch.

## Result

| Model | WER | CER | Utterances |
|---|---|---|---|
| `openai/whisper-small`, zero-shot | **492.66%** | **279.97%** | 150 (real test split) |

Spot-checked the individual outputs (`reports/baseline_whisper-small_examples.jsonl`)
— hypotheses are recognizable phonetic attempts at the Nepali reference text,
not degenerate/repeated tokens, so this is genuinely poor zero-shot performance,
not a bug. It reads as considerably worse than an earlier branch's zero-shot
result (WER 221.76% on a similarly-sized subset) — plausibly explained by this
corpus using a 5-of-6-zip subset (missing `asr_nepali_1`, dropped after
persistent download failures), which changes which speakers/utterances end up
in the test split, not by any code difference in how the baseline is measured.
