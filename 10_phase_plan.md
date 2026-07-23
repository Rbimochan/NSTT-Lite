# STW7088CEM — Artificial Neural Networks Coursework
## 10-Phase Project Plan (NSTT-Lite: Nepali Speech-to-Text)

This plan maps your existing NSTT-Lite work onto the assignment brief's requirements:
project proposal (1 page), 4500-word report, multiple tasks (ASR + gender
classification satisfies "more than one task"), reproducibility with screenshots,
and the 100-mark rubric (Technical Quality 45, Difficulty 15, Originality 10,
Reproducibility 10, Style 10, Proposal 10).

---

### Phase 1 — Project Proposal (1 A4 page, submitted separately + in appendix)
- Title, problem description, dataset name + direct link, work plan.
- Two tasks stated explicitly: (1) ASR fine-tuning, (2) speaker gender classification.
- Justify dataset choice (OpenSLR54) and why it's suitable for a master's-level ANN project.
- **Rubric target:** The Project Proposal (10 marks) — needs to read as an "achievable goal, clear steps."

### Phase 2 — Environment & Reproducibility Setup
- Fix Python/library versions, document GPU/CPU used (screenshot device info).
- Set up manifest-based train/val/test splits (speaker-disjoint) so results are reproducible.
- **Rubric target:** Reproducibility (10 marks) — this phase produces the "evidence" (screenshots, environment) required later.

### Phase 3 — Data Preparation
- Load OpenSLR54, clean transcripts, normalize Devanagari text, derive acoustic
  pseudo-labels for gender (F0/pitch threshold) — document this as a labeled
  limitation, not ground truth.
- Report class balance, hours of audio, speaker counts.
- **Rubric target:** Data Preparation (5 marks).

### Phase 4 — Baseline Establishment (Zero-Shot)
- Run zero-shot WER/CER for Whisper-small/medium/large and MMS on your test split, unmodified.
- This is your "before" — needed to demonstrate improvement later.
- **Rubric target:** contributes to Depth of Information + Experimental section.

### Phase 5 — Algorithm Selection & Justification
- Explain why Whisper-small (encoder-decoder transformer) was chosen over
  training from scratch or other architectures (MMS, wav2vec2) — data size,
  compute budget, transfer learning suitability for low-resource languages.
- **Rubric target:** Algorithm Selection (5 marks).

### Phase 6 — Task 1: ASR Fine-Tuning
- Fine-tune Whisper-small on Nepali subset; document any modifications to
  training loop, learning rate schedule, or decoding strategy, and explain why.
- Full training run (Colab T4, not just the reduced local run) for final reported numbers.
- **Rubric target:** Algorithm Modifications (5 marks) + Difficulty (15 marks — this is your main complexity driver).

### Phase 7 — Task 2: Gender Classification (Second Task)
- Train a classifier on encoder embeddings (or separate feature extractor) against
  the acoustic pseudo-labels from Phase 3.
- Report the result honestly, including the negative finding (below majority-class
  baseline) with a discussion of likely causes (undertrained encoder, noisy labels).
- **Rubric target:** satisfies "more than one task" requirement + Discussion of Findings (5 marks) — examiners value honest negative results with real analysis.

### Phase 8 — Deployment / Production Benchmark
- Convert fine-tuned model to CTranslate2/Faster-Whisper; benchmark latency and
  model size vs. the original checkpoint.
- Frame this as "originality" — going beyond a standard fine-tuning exercise into
  a deployable, efficient artifact.
- **Rubric target:** Originality (10 marks).

### Phase 9 — Experimental Analysis & Discussion
- Compile all results: baseline vs. fine-tuned WER/CER, ablations (e.g.
  speaker-disjoint vs. leaky split, matched step counts), error analysis
  (worst-performing examples, error categories).
- Discuss findings against the research question, not just report numbers.
- **Rubric target:** Discussion of Findings (5 marks) + Information Presentation (10 marks — use tables/graphs here).

### Phase 10 — Report Writing, Screenshots & Submission Packaging
- Assemble full report (intro, background/related work, method, experiments,
  discussion, conclusion, references, appendices).
- Insert code in appendix (in full, using a proper Word code-formatting method),
  plus screenshots of every experiment run showing device/software.
- Include the original project proposal verbatim in the appendix.
- Proofread for word count (4500 strict limit, excluding appendices) and correct referencing.
- **Rubric target:** Style and Format (10 marks) + final check against Reproducibility (10 marks).

---

## Notes on rubric alignment
- **Difficulty (15 marks)** is best earned in Phase 6 — fine-tuning a modern
  transformer ASR model on a genuinely low-resource language is a legitimately
  hard, current problem, not a toy dataset.
- **Originality (10 marks)** is earned by combining two tasks (ASR + gender) and
  the deployment/efficiency angle (Phase 8), not by claiming to beat commercial SOTA.
- **Do not reference beating closed commercial models (e.g. Scribe)** as your
  benchmark in the report — compare only against other open models you actually
  ran yourself, since the brief requires full reproducibility and no external links.
