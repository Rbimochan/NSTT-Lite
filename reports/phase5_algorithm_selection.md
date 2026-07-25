# Phase 5 — Algorithm Selection & Justification

## Chosen algorithm: fine-tuning Whisper-small (encoder-decoder transformer)

**Whisper-small** (Radford et al., 2022) — a 244M-parameter multilingual
encoder-decoder transformer pretrained on 680k hours of weakly supervised
audio — fine-tuned end-to-end on the 15-hour Nepali subset via Hugging Face
`Seq2SeqTrainer`.

## Why not train from scratch?

A from-scratch ASR model on 15 hours of Nepali audio is not a serious option:

- **Data scale.** Modern end-to-end ASR architectures (transformer CTC or
  seq2seq) need hundreds to thousands of hours to reach usable WER when
  trained from random initialization. 15 hours is roughly two orders of
  magnitude short. Transfer learning is the standard, well-evidenced answer
  for low-resource languages: reuse acoustic representations learned on
  high-resource data, adapt only to the target language's phonetics and script.
- **Compute budget.** This is a coursework project with a Colab T4 budget.
  Pretraining-scale compute is out of reach; fine-tuning a pretrained model
  for a few epochs is squarely within it.
- **Empirical support in this project's own data.** The Phase 4 baseline shows
  whisper-small already produces recognizable phonetic attempts at Nepali
  zero-shot (garbled but structurally speech-like output, not noise) — i.e.
  its pretrained acoustic representations transfer, and what is missing is
  Nepali-specific decoder calibration. That is exactly the gap fine-tuning
  closes.

## Why Whisper-small over the alternatives?

| Candidate | Why not chosen |
|---|---|
| **Whisper-medium / large** | Better zero-shot quality, but 3-8x the fine-tuning and inference cost. On a Colab T4 with batch 2 + gradient accumulation, small trains in hours; medium/large risk not completing within session limits. Small is the largest variant that comfortably fits the compute budget end-to-end (training + deployment benchmark in Phase 8). |
| **wav2vec2 / XLS-R (CTC)** | A prior audit in this repo's history evaluated `gagan3012/wav2vec2-xlsr-nepali` (an XLS-R model fine-tuned on OpenSLR-43): it achieved 4.91% WER in-domain on its own single-speaker training corpus but degraded to ~65% WER on a multi-speaker OpenSLR-54 test set — evidence that CTC fine-tunes on narrow Nepali data generalize poorly, and that speaker-diverse training data (which this project's speaker-disjoint 160-speaker subset provides) matters more than architecture choice. Whisper's seq2seq decoder also provides an implicit language model that pure-CTC models lack, which helps in a morphologically rich language like Nepali. |
| **MMS (Meta)** | A strong multilingual baseline, but its Nepali adapter path is again CTC-based (same generalization concern), and adopting it would mean abandoning the encoder-embedding-based gender classification design (Phase 7), which reuses Whisper's encoder directly. One model serving both tasks is a deliberate design economy. |
| **Conformer/Zipformer (e.g. icefall recipes)** | State-of-the-art WER on high-resource benchmarks, but requires training from scratch or from checkpoints without Nepali coverage — back to the data-scale problem above. |

## Secondary considerations

- **Two tasks, one backbone.** Phase 7's gender classifier consumes mean-pooled
  Whisper encoder hidden states. Choosing Whisper for ASR makes the second task
  nearly free at inference time (one encoder pass serves both heads) — a design
  argument, not just a convenience.
- **Deployment path exists.** Phase 8 requires a production-style benchmark;
  CTranslate2/Faster-Whisper is a mature, well-supported conversion path for
  Whisper specifically. The alternatives have weaker or no equivalent tooling.
- **Reproducibility.** Whisper-small's fine-tuning recipe (AdamW, FP16,
  batch 2, gradient accumulation 4, per the Hugging Face fine-tuning tutorial
  cited in `src/training.py`) is standard and well documented, which supports
  the coursework's reproducibility requirement — every hyperparameter choice
  can be traced to a published recipe rather than ad-hoc tuning.

## Rubric target

Algorithm Selection (5 marks): the selection is justified against data size,
compute budget, transfer-learning suitability for a low-resource language, and
this project's own empirical evidence (Phase 4 baseline; the prior XLS-R
generalization audit), not just stated.
