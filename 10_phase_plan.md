# STW7088CEM — Artificial Neural Networks Coursework
## 10-Phase Project Plan (NSTT-Lite: Auditing & Fixing a Published Nepali ASR Model)

**Pivot note (2026-07-24, supersedes the earlier Whisper-centric version of this
file):** this project is NOT a Whisper project. Its subject is
`gagan3012/wav2vec2-xlsr-nepali` — a published Hugging Face XLS-R (wav2vec2)
Nepali ASR model that self-reports **5.97% WER**. Preliminary auditing in this
repository found that claim technically holds in-domain (**4.91% WER** measured
on its own single-speaker OpenSLR-43 training corpus) but **collapses to ~65%
WER** on multi-speaker OpenSLR-54 data. The project: audit that benchmark
rigorously, then **fine-tune the same XLS-R model on speaker-diverse data** to
close the generalization gap. Experiment tracking is **MLflow** (not
TensorBoard) throughout.

This maps onto the assignment brief: proposal (1 page), 4500-word report,
reproducibility with screenshots, and the 100-mark rubric (Technical Quality
45, Difficulty 15, Originality 10, Reproducibility 10, Style 10, Proposal 10).

---

### Phase 1 — Project Proposal (1 A4 page, submitted separately + in appendix)
- Title, problem description (the 5.97% claim and why self-reported single-speaker
  benchmarks mislead), dataset names + direct links (OpenSLR-43, OpenSLR-54), work plan.
- Tasks stated explicitly: (1) benchmark audit/reproduction, (2) XLS-R fine-tuning
  on speaker-diverse data, (3) generalization re-evaluation.
- **Rubric target:** The Project Proposal (10 marks).

### Phase 2 — Environment & Reproducibility Setup
- Pinned Python/library versions (`requirements.txt`), document GPU/CPU used
  (screenshot device info).
- **MLflow** tracking wired: every training/eval run logs params, metrics, and
  artifacts to a local `mlruns/` store; `mlflow ui` screenshots for evidence.
- Manifest-based speaker-disjoint train/val/test splits (already built, model-agnostic).
- **Rubric target:** Reproducibility (10 marks).

### Phase 3 — Data Preparation  ✅ (done, model-agnostic — reused as-is)
- OpenSLR-54: 15,171 utterances / 160 speakers / 15.03 hours, 16kHz mono,
  NFC-normalized Devanagari, speaker-disjoint 80/10/10 (overlap = 0).
- Gender pseudo-labels (F0-pitch, 165Hz) retained as dataset metadata for
  per-group error analysis in Phase 9 — documented as a heuristic, not ground truth.
- OpenSLR-43 (the model's own corpus) accessed via `gauravparajuli/slr43` on HF
  for in-domain reproduction — kept strictly separate from OpenSLR-54 everywhere.
- **Rubric target:** Data Preparation (5 marks).

### Phase 4 — Baseline Establishment (The Audit)
- Reproduce the 5.97% claim: zero-shot `gagan3012/wav2vec2-xlsr-nepali` on
  OpenSLR-43 (in-domain).
- Same model, zero-shot, on this project's speaker-disjoint OpenSLR-54 test split
  (out-of-domain) — quantify the generalization collapse. Both runs logged to MLflow.
- **Rubric target:** Depth of Information + Experimental section; this audit is
  also the project's originality hook.

### Phase 5 — Algorithm Selection & Justification
- Why keep the XLS-R backbone and fine-tune it (rather than switch architecture):
  isolates the variable that matters (training-data diversity, not architecture);
  wav2vec2/XLS-R is proven for low-resource ASR; direct comparability with the
  audited checkpoint.
- Why not Whisper/from-scratch/MMS — data scale, compute budget, and the fact
  that the research question is about THIS published model's claim.
- **Rubric target:** Algorithm Selection (5 marks).

### Phase 6 — Task: XLS-R Fine-Tuning on Speaker-Diverse Data
- Fine-tune `gagan3012/wav2vec2-xlsr-nepali` (CTC head) on the 15-hour
  speaker-disjoint OpenSLR-54 train split; document modifications (freezing
  strategy, LR schedule, early stopping on val WER) and why.
- Full run on Colab T4; smoke-test locally first; MLflow logging + resumable
  checkpoints throughout.
- **Rubric target:** Algorithm Modifications (5 marks) + Difficulty (15 marks).

### Phase 7 — Generalization Re-Evaluation
- Fine-tuned model vs. original checkpoint, on BOTH test sets:
  (a) OpenSLR-54 speaker-disjoint test (did we close the gap?),
  (b) OpenSLR-43 (did we destroy in-domain performance? catastrophic forgetting check).
- Four-cell result table: {original, fine-tuned} × {in-domain, out-of-domain}.
- **Rubric target:** Discussion of Findings (5 marks) — this is the paper's core claim.

### Phase 8 — Efficiency / Deployment Benchmark
- Inference benchmark of the fine-tuned model: CPU latency, model size,
  dynamic int8 quantization (torch) vs. FP32 accuracy/latency trade-off.
- Frame as originality: a published-model audit that ends in a deployable,
  measured artifact rather than only a table.
- **Rubric target:** Originality (10 marks).

### Phase 9 — Experimental Analysis & Discussion
- Error analysis on real outputs: error categories, worst examples, per-speaker
  WER spread, WER by gender pseudo-label group (explicitly labeled as
  "pitch-threshold group", not verified gender).
- Ablation: speaker-disjoint vs. utterance-random ("leaky") split at matched
  training budget — quantify how much leakage inflates apparent WER.
- Discuss findings against the research question (do self-reported low-resource
  ASR benchmarks survive speaker diversity?), not just report numbers.
- **Rubric target:** Discussion of Findings (5) + Information Presentation (10).

### Phase 10 — Report Writing, Screenshots & Submission Packaging
- Full report (intro, background, method, experiments, discussion, conclusion,
  references, appendices); code in appendix; screenshots of every run showing
  device/software (including `mlflow ui`); proposal verbatim in appendix.
- 4500-word limit (excluding appendices); correct referencing; NAME_studentID
  file naming; submit via Campus 4.0.
- **Rubric target:** Style and Format (10 marks) + Reproducibility final check.

---

## Notes on rubric alignment
- **Difficulty (15)** is earned in Phase 6: fine-tuning a real published
  low-resource ASR model to fix a demonstrated generalization failure is a
  legitimately hard, current problem.
- **Originality (10)** is the audit itself — independently stress-testing a
  published benchmark claim and then repairing the model — plus the Phase 8
  efficiency benchmark.
- **Compare only against models/numbers you ran yourself** (the audited
  checkpoint before/after). The 5.97% self-reported figure is quoted as the
  claim under audit, with your own measured numbers alongside it.
- Never merge OpenSLR-43 and OpenSLR-54 results into one number anywhere.
