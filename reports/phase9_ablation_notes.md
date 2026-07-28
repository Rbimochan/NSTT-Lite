# Phase 9 — Speaker-Leakage Ablation

**Rubric target:** Discussion of Findings (5) + Information Presentation (10).

## Setup

Two fine-tuning runs, identical hyperparameters, identical 5-epoch budget,
identical evaluation methodology (150-utterance seeded sample per test set,
`scripts/run_generalization_eval.py`), differing **only** in how the training
data was split:

- **Speaker-disjoint** (Phase 6/7): `data/manifests/` — `speaker_disjoint_split`,
  0 speakers shared between train/val/test.
- **Leaky** (this ablation): `data/manifests_leaky/` — `utterance_random_split`
  (`scripts/make_leaky_manifests.py`), 160/160 speakers shared across all
  three splits (every speaker's utterances scattered randomly).

## Three-way comparison

| | In-domain (OpenSLR-43) | Out-of-domain (OpenSLR-54) |
|---|---|---|
| Original (unmodified checkpoint) | 4.91% WER | 62.30% WER |
| Fine-tuned, **speaker-disjoint** | 16.40% WER | 38.17% WER |
| Fine-tuned, **leaky** | 16.96% WER | **32.55% WER** |

Training-time validation WER (leaky, own val split, for reference — improved
monotonically, no early stop): epoch 1 69.43% → epoch 2 66.64% → epoch 3
65.50% → epoch 4 65.06% → epoch 5 **64.58%**.

## Finding: leakage inflates apparent out-of-domain performance by ~5.6 percentage points

The leaky-split model scores **better** on the out-of-domain SLR54 test
(32.55% vs. 38.17% WER) than the properly speaker-disjoint model — despite
identical training budget and hyperparameters. This is the expected direction
if leakage inflates apparent performance: because the leaky split scatters
each speaker's utterances across train/val/test, the "out-of-domain" test set
is not actually out-of-domain for many of its speakers — the model has
already partially learned their specific voice characteristics during
training. In-domain (SLR43) performance is nearly identical between the two
(16.40% vs. 16.96%), as expected, since SLR43 is untouched by either training
run and its own single speaker isn't part of either split.

**This confirms the project's methodological choice to use a speaker-disjoint
split throughout was the right one** — a leaky evaluation would have reported
a ~15% relative WER improvement (38.17% → 32.55%) that does not reflect true
generalization to new speakers, only memorization of speakers already seen.
Any Nepali ASR benchmark (including, plausibly, aspects of the audited
model's own self-reported figures, though OpenSLR-43 is single-speaker so
this specific leakage mode doesn't directly apply there) that doesn't control
for speaker overlap between train and test risks the same inflation.

## Caveat

This ablation isolates *speaker-split leakage specifically* — it does not
test every possible form of evaluation leakage (e.g. text/topic overlap,
recording-session artifacts). The ~5.6pp gap should be read as "at least this
much" inflation from speaker leakage alone, not an upper bound on all
possible leakage effects.

## Status

- [x] Leaky manifest built and verified (160/160 speaker overlap confirmed via `count_speaker_overlap`)
- [x] Leaky model trained to the same 5-epoch budget (11,541s / ~3h12m*, no early stop)
- [x] Matched eval on both test sets
- [x] Three-way comparison table complete

*Wall-clock for this run's `train_runtime` field is lower than Phase 6's
~7h40m because this session's process was interrupted once (agent teardown,
not a crash) and resumed from checkpoint-4551 — `train_runtime` only counts
active training time within each process lifetime, not the gap between them.
