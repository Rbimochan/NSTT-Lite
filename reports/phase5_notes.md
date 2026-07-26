# Phase 5 — Algorithm Selection & Justification

**Rubric target:** Algorithm Selection (5 marks).

## Decision

Fine-tune **`gagan3012/wav2vec2-xlsr-nepali`** itself — the audited checkpoint
(XLS-R / wav2vec2, CTC objective) — on the speaker-disjoint OpenSLR-54 subset,
keeping its architecture, tokenizer, and vocabulary unchanged.

## Why fine-tune the audited checkpoint rather than any other model?

1. **It isolates the variable under study.** The research question is whether
   the model's generalization collapse (4.91% WER in-domain → ~65% WER
   out-of-domain, Phase 4) is caused by *training-data narrowness* rather than
   architecture. Fine-tuning the same checkpoint on speaker-diverse data and
   re-measuring both test sets answers that directly. Swapping architecture
   (e.g. to Whisper) would confound the comparison — any improvement could be
   attributed to the backbone change instead of the data fix.
2. **Direct before/after comparability.** Phase 7's four-cell table
   ({original, fine-tuned} × {in-domain, out-of-domain}) is only meaningful if
   both checkpoints share architecture, vocab, and decoding — which they do,
   by construction.
3. **XLS-R is a proven low-resource ASR backbone.** XLS-R was pretrained on
   436k hours across 128 languages specifically for cross-lingual transfer;
   wav2vec2-style self-supervised pretraining + CTC fine-tuning is the
   standard, well-evidenced recipe for languages with limited labeled data.
4. **The checkpoint already knows Nepali.** Its CTC head and Devanagari vocab
   are trained; our job is adaptation to speaker diversity, not learning the
   language from scratch. A low learning rate (3e-5) with a frozen
   convolutional feature encoder is the standard configuration for this kind
   of continued fine-tuning, and Phase 7 explicitly checks we haven't
   catastrophically forgotten the original domain.

## Alternatives considered

| Option | Verdict |
|---|---|
| Train from scratch | Rejected. ~15 hours of labeled audio is two orders of magnitude short of what end-to-end ASR needs from random init; transfer learning is the only feasible path under a Colab budget. |
| Whisper (any size) | Rejected. A different architecture/decoder family would break the before/after comparability that the audit design depends on (see #1). This project was explicitly re-scoped away from Whisper. |
| Meta MMS | Rejected. Same confound as Whisper, plus a separate adapter/stack; and its Nepali path is also CTC, so it would not even test a different hypothesis. |
| Fresh XLS-R from `facebook/wav2vec2-xls-r-300m` | Considered. Would test "is speaker-diverse data alone enough starting from generic multilingual pretraining?" — a fine question, but it needs a new Nepali vocab/CTC head and substantially more training to converge, and it does not audit the published checkpoint, which is the point. Noted as future work. |

## Training configuration (locked in `src/xlsr_training.py`)

| Setting | Value | Why |
|---|---|---|
| Base checkpoint | `gagan3012/wav2vec2-xlsr-nepali` | The model under audit |
| Objective | CTC (unchanged) | Same head, same vocab — comparability |
| Feature encoder | Frozen | Standard practice; conv encoder was pretrained on far more audio than we have |
| LR | 3e-5, warmup 500 | Low LR for continued fine-tuning of an already-adapted checkpoint |
| Batch / grad-accum | 2 / 4 (effective 8) | Fits Colab T4 with FP16 |
| Epochs | 5 ceiling | Coursework budget |
| Early stopping | patience 2 on validation WER | Stop on the metric we report; best checkpoint kept |
| Precision | FP16 on CUDA, FP32 on CPU | T4 speed; CPU smoke stability |
| Tracking | MLflow (`mlruns/`, experiment `phase6-finetune`) | Project decision: MLflow, not TensorBoard |
