![](appendix_screenshots/softwarica_coventry_logo.png){width="6.0in"}

<br><br><br>

# Project Proposal

### NSTT-Lite: Auditing and Repairing a Published Nepali ASR Model

<br><br>

**Module:** ST7088CEM — Artificial Neural Networks

**Student:** Bimochan Raj Kunwar

**Coventry ID:** 17108924

**Programme:** MSc7-S2

**Email:** 250594@softwarica.edu.np

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

## Problem

`gagan3012/wav2vec2-xlsr-nepali` is a published Hugging Face XLS-R (wav2vec2)
model for Nepali speech recognition that self-reports **5.97% WER** — a
remarkably strong number for a low-resource language. However, that figure was
measured on the model's own training corpus, OpenSLR-43, which is effectively
single-speaker (one female voice). My preliminary experiments confirm the claim
technically holds in-domain (I measured **4.91% WER** on OpenSLR-43) but the
same model collapses to roughly **65% WER** on multi-speaker Nepali speech
(OpenSLR-54) — a >13x degradation. The published benchmark therefore does not
describe real-world performance. This project audits that claim rigorously and
then repairs the model.

## Tasks

1. **Benchmark audit.** Reproduce the self-reported figure in-domain
   (OpenSLR-43) and measure the same checkpoint zero-shot on a speaker-disjoint
   multi-speaker test split (OpenSLR-54), quantifying the generalization gap.
2. **Fine-tuning.** Fine-tune the same XLS-R model (CTC objective) on a
   ~15-hour, 160-speaker, speaker-disjoint OpenSLR-54 training subset to close
   that gap.
3. **Generalization re-evaluation.** Evaluate original vs. fine-tuned
   checkpoints on both test sets — including a catastrophic-forgetting check
   on the original corpus — plus error analysis and a speaker-leakage ablation.

## Datasets

- **OpenSLR-54** — crowdsourced multi-speaker Nepali ASR corpus (Kjartansson
  et al., SLTU 2018), 157,905 utterances, CC BY-SA 4.0.
  Link: https://www.openslr.org/54/
  (Working subset already prepared: 15,171 utterances / 160 speakers /
  15.03 hours, speaker-disjoint 80/10/10 split, zero speaker overlap.)
- **OpenSLR-43** — the model's own training corpus (single-speaker female
  Nepali TTS-style data), used only for in-domain reproduction of the claim.
  Link: https://www.openslr.org/43/

The two corpora are kept strictly separate in all results.

## Method and infrastructure

Fine-tuning uses the Hugging Face `transformers` CTC training stack on a Colab
T4 GPU (FP16, batch 2, gradient accumulation, early stopping on validation
WER). All experiments — audit runs, training, and re-evaluation — are tracked
with **MLflow** (parameters, WER/CER metrics, artifacts), giving a reproducible
evidence trail; environment versions are pinned and device screenshots
captured throughout.

## Work plan (10 phases)

| Phase | Deliverable |
|---|---|
| 1 | This proposal |
| 2 | Environment + MLflow reproducibility setup |
| 3 | Data preparation (complete: speaker-disjoint 15hr subset) |
| 4 | Baseline audit: in-domain vs. out-of-domain zero-shot WER |
| 5 | Algorithm selection justification |
| 6 | XLS-R fine-tuning on speaker-diverse data (Colab T4) |
| 7 | Generalization re-evaluation (4-cell before/after × in/out-of-domain) |
| 8 | Efficiency benchmark (latency, size, int8 quantization) |
| 9 | Error analysis, leakage ablation, discussion |
| 10 | Report writing, evidence assembly, submission |

## Achievability

The audit half is already demonstrated end-to-end at small scale (the 4.91% /
~65% preliminary numbers above), and the data pipeline (download,
preprocessing, speaker-disjoint splitting) is built and validated. The main
remaining cost is the Phase 6 GPU fine-tuning run, which fits a free Colab T4
budget with checkpointed, resumable training. Every phase produces a concrete,
checkable artifact (a metric, a checkpoint, an MLflow run), so progress is
verifiable throughout rather than only at submission.
