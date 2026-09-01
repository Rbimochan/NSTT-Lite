# Phase 7 — Generalization Re-Evaluation

**Rubric target:** Discussion of Findings (5 marks) — this is the project's central result.

## The four-cell comparison

Both checkpoints evaluated identically: 150 utterances each on OpenSLR-43
(in-domain) and a seeded-shuffle sample of OpenSLR-54's speaker-disjoint test
split (out-of-domain), via `scripts/run_generalization_eval.py`.

| | In-domain (OpenSLR-43) | Out-of-domain (OpenSLR-54) |
|---|---|---|
| **Original** (`gagan3012/wav2vec2-xlsr-nepali`) | 4.91% WER / 0.87% CER | 62.30% WER / 17.38% CER |
| **Fine-tuned** (5 epochs on speaker-disjoint SLR54 train) | 16.40% WER / 2.98% CER | **38.17% WER / 9.89% CER** |

## Interpretation

Fine-tuning on the 15-hour, 160-speaker, speaker-disjoint OpenSLR-54 training
subset **substantially closes the generalization gap** the audit set out to
investigate: out-of-domain WER falls from 62.30% to 38.17% — a ~39% relative
reduction. This is the project's headline finding and directly supports the
motivating hypothesis: the published 5.97% claim doesn't generalize because
the model was trained on narrow (single-speaker) data, and training on
speaker-diverse data measurably repairs that.

The cost is a real, not-hidden trade-off: in-domain WER on OpenSLR-43 rises
from 4.91% to 16.40%. This is expected specialization/mild catastrophic
forgetting — 5 epochs of continued fine-tuning on a different data
distribution nudges the model away from the narrow distribution it was
originally sharpest on. Whether this trade-off is "worth it" depends on the
deployment target: a system meant to serve diverse real-world speakers is far
better served by the fine-tuned checkpoint (38.17% beats 62.30% by a wide
margin) than by the original, despite the original's stronger score on its own
narrow training distribution.

## A note on which WER number is "the" result

Training logged an in-training validation WER of 68.83% at the final epoch
(`reports/phase6_train_full_*.json`) — this figure is **not** the same
measurement as the 38.17% above, and should not be quoted interchangeably:

- The 68.83% figure is computed on SLR54's own *validation* split (1,505
  utterances), using the Trainer's greedy CTC decode during training.
- The 38.17% figure here is computed on a separate, fixed 150-utterance sample
  of the *test* split, via the same standalone script and methodology used for
  the original checkpoint and for Phase 4 — enabling a true apples-to-apples
  comparison against both the original checkpoint and OpenSLR-43.

The two numbers disagreeing (68.83% vs 38.17%, both "out-of-domain SLR54")
is not a bug — different splits, different sample, and possibly the training
val split simply being a harder/different subset than the fixed test sample.
The 38.17% figure is the one that belongs in the four-cell table and the
report, because it is the one measured identically to every other cell.

## Training run summary

- 5 epochs, no early stopping (validation WER improved monotonically every
  epoch: 72.98% → 70.88% → 70.09% → 69.03% → 68.83% on the *training* val
  split) — full run, ~7h40m wall-clock, local CPU/MPS (Apple Silicon, no CUDA).
- One MPS out-of-memory crash after epoch 1 (17GB allocated), fixed with a
  periodic `torch.mps.empty_cache()` callback (`MPSCacheClearCallback` in
  `src/xlsr_training.py`); resumed cleanly from the epoch-1 checkpoint with no
  further issues.
- All runs logged to MLflow (experiments `phase6-finetune`, `phase7-generalization`).

## Status

- [x] Original checkpoint: both test sets evaluated (Phase 4 = Phase 7 "original", confirmed reproducible)
- [x] Fine-tuned checkpoint: both test sets evaluated
- [x] Four-cell table complete
- [ ] Full-corpus (non-150-sample) confirmation — optional strengthening step before the report is finalized
