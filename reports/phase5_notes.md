# Phase 5 — Algorithm Selection & Justification

**Rubric target:** Algorithm Selection (5 marks).

## Decision

Fine-tune **`openai/whisper-small`** only (encoder–decoder Transformer,
~244M parameters). Phase 4's zero-shot Whisper-small numbers are the sole
ASR "before" baseline for the Phase 6 before/after comparison.

## Alternatives considered

| Option | Verdict |
|---|---|
| Train an ASR model from scratch | Rejected. A 10–20 hour speaker-disjoint subset is far too small to train a competitive speech Transformer; large labeled speech and long wall-clock training would be required. Transfer learning is the only feasible path under this project's compute budget. |
| Whisper-medium / Whisper-large | Rejected for fine-tuning. Higher capacity, but Phase 4 already scoped zero-shot baselines to small because this machine is CPU-only and Phase 6 only fine-tunes small on Colab T4. Larger Whisper would dominate wall-clock without changing the research question (does fine-tuning close the Nepali zero-shot gap?). |
| Meta MMS | Rejected as the primary model. Strong multilingual ASR, but a separate stack (fairseq / different decoding conventions) and not what this project fine-tunes — so it would not form a clean before/after pair with Phase 6. |
| wav2vec 2.0 / HuBERT | Rejected as the primary model. Proven for low-resource ASR, but typically needs a CTC (+ optional LM) decoding setup and less convenient off-the-shelf multilingual language conditioning than Whisper's seq2seq + language token path. |

## Why Whisper-small fits this project

1. **Transfer learning for low-resource Nepali.** Whisper is pretrained on
   large-scale multilingual speech. Fine-tuning adapts the encoder/decoder to
   Devanagari Nepali with limited hours — the standard approach when labeled
   data is scarce relative to model capacity.
2. **Data size vs capacity.** The working subset is tens of hours, not
   hundreds. Small capacity reduces overfitting risk relative to medium/large
   under the same step budget and early-stopping regime.
3. **Compute budget.** Fits Colab free-tier T4 with FP16, batch size 2, and
   gradient accumulation 4 (`src/training.py`). Local CPU-only smoke tests
   remain practical; medium/large would not.
4. **Two-task architecture.** The shared Whisper encoder supports Task 1
   (ASR fine-tuning) and Task 2 (mean-pooled encoder embeddings → gender
   classifier) without a second front-end acoustic model.
5. **Reproducibility.** Hugging Face `Seq2SeqTrainer` path is already wired
   in `src/training.py`, with pinned deps in `requirements.txt` — matches the
   coursework requirement for screenshots and a reproducible stack.
6. **Phase 4 evidence.** Zero-shot Whisper-small on the real test split:
   **WER 492.66%**, **CER 279.97%** (150 utterances). The pretrained model
   fails hard on this Nepali split; that gap is exactly what Phase 6
   fine-tuning is meant to close.

7. **Prior XLS-R evidence from this repo's history.** An earlier audit in this
   repository evaluated `gagan3012/wav2vec2-xlsr-nepali` (an XLS-R/CTC model
   fine-tuned on the single-speaker OpenSLR-43 corpus): 4.91% WER in-domain,
   collapsing to ~65% WER on a multi-speaker OpenSLR-54 test set. That is
   direct, first-hand evidence that (a) narrow-corpus CTC fine-tunes
   generalize poorly across speakers, and (b) speaker-diverse training data —
   which this project's 160-speaker speaker-disjoint subset provides — matters
   at least as much as architecture choice.
8. **Deployment path.** Phase 8 requires a production-style benchmark, and
   CTranslate2/Faster-Whisper is a mature conversion path for Whisper
   specifically; the rejected alternatives have weaker or no equivalent
   tooling.

## Scope link to Phase 4 / Phase 6

Phase 4 deliberately measured only Whisper-small (not medium/large/MMS) so
the baseline matches the model Phase 6 actually fine-tunes. Algorithm
selection and baseline measurement stay consistent: one model family, one
before number, one after number.
