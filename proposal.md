# Project Proposal — NSTT-Lite: Nepali Speech-to-Text with Whisper-Small

**Module:** ST7088CEM — Artificial Neural Networks
**Student:** [Name] — [Student ID]

## Problem

Automatic speech recognition (ASR) for Nepali is under-served by off-the-shelf
multilingual models: Nepali is a low-resource language in the pretraining mix
of general-purpose ASR systems, so zero-shot transcription quality is
substantially worse than for high-resource languages. This project fine-tunes
OpenAI's Whisper-small on a Nepali speech corpus and measures the improvement
over zero-shot performance, while also tackling a second, distinct task:
classifying speaker gender from the model's own encoder representations.

## Tasks (satisfying "more than one task")

1. **ASR fine-tuning.** Fine-tune Whisper-small on a Nepali speech subset;
   report WER/CER before (zero-shot) and after fine-tuning.
2. **Speaker gender classification.** Train a lightweight classifier head on
   mean-pooled Whisper encoder embeddings to predict speaker gender, evaluated
   against a trivial majority-class baseline.

## Dataset

**OpenSLR-54** — a crowdsourced Nepali ASR corpus (Kjartansson et al., SLTU
2018), 157,905 utterances, CC BY-SA 4.0.
Link: https://www.openslr.org/54/

Chosen because: it is the largest open Nepali ASR corpus available, includes
multiple speakers (enabling a genuine speaker-disjoint train/test split,
unlike single-speaker TTS-style corpora), and ships only audio + transcript +
speaker ID — no demographic metadata, which is itself a modeling constraint
this project addresses explicitly (Phase 3) rather than ignores.

## Work plan (10 phases — see `10_phase_plan.md` for full detail)

| Phase | Deliverable |
|---|---|
| 1 | This proposal |
| 2 | Environment setup, reproducibility scaffolding (manifests, speaker-disjoint split) |
| 3 | Data preparation: preprocessing, acoustic gender pseudo-labels |
| 4 | Zero-shot baseline (Whisper-small, and comparison models) |
| 5 | Algorithm selection justification |
| 6 | ASR fine-tuning (full-scale, GPU) |
| 7 | Gender classification (second task) |
| 8 | Deployment benchmark (CTranslate2/Faster-Whisper) |
| 9 | Experimental analysis, ablations, error analysis |
| 10 | Report writing, evidence assembly, submission |

## Achievability

This plan reuses a working pipeline (data loading, preprocessing, splitting,
training, evaluation) already prototyped and validated at reduced scale in
earlier iterations of this project, so the risk is compute time (a full 5-epoch
Colab T4 run), not unproven methodology. Each phase has a concrete, checkable
output (a report file, a checkpoint, a metric), so progress is verifiable at
every step rather than only at submission.
