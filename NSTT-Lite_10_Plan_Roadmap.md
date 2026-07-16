# NSTT-Lite — 10-Plan Execution Roadmap

Ten sequential plans, Plan 1 through Plan 10, taking the project from a blank Colab notebook to a submission-ready coursework package. Each plan lists the objective, concrete actions, and the output that feeds the next plan. Marking-criteria tags show which of the 100 marks each plan protects.

## Plan 1 — Environment and Data Acquisition
**Objective:** Get a working Colab environment and the raw dataset in hand.
- Set up Colab (T4 GPU), install Hugging Face Transformers, Datasets, torchaudio, TensorBoard, CTranslate2, Streamlit.
- Download OpenSLR SLR54 (https://openslr.org/54/); verify file counts against the corpus documentation.
- Take first screenshots: Colab runtime type (GPU), library versions, dataset download completing.
**Output:** Raw corpus stored in Drive/Colab, environment confirmed reproducible.
**Protects:** Reproducibility (10), Data Preparation (5).

## Plan 2 — Preprocessing and Dataset Split
**Objective:** Turn raw SLR54 into clean, split, model-ready data.
- Resample to 16kHz mono, drop invalid/corrupt clips, normalize Nepali transcripts (Unicode NFC), clean punctuation/casing.
- Extract or derive speaker gender labels from metadata for the classification task.
- Build the speaker-disjoint 10–20 hour subset; split 80/10/10 train/val/test, confirming no speaker overlap across splits.
**Output:** A finalized, versioned dataset manifest (CSV/JSON) mapping audio → transcript → speaker → gender → split.
**Protects:** Data Preparation (5), Reproducibility (10).

## Plan 3 — Baseline Evaluation (Zero-Shot)
**Objective:** Establish the "before fine-tuning" numbers for both tasks.
- Run pretrained Whisper-Small zero-shot on the Nepali test set; compute WER/CER.
- Build a trivial baseline for gender classification (majority-class or simple acoustic-feature classifier) to give the second task a comparison point too.
- Screenshot every run with visible device info.
**Output:** Baseline WER/CER numbers and baseline gender-classification accuracy, both logged.
**Protects:** Discussion of Findings (5), Algorithm Selection (5), Difficulty (part of 15).

## Plan 4 — ASR Fine-Tuning Setup
**Objective:** Get the Whisper-Small fine-tuning loop running correctly before trusting it with real training time.
- Write the Seq2SeqTrainer script: AdamW, FP16, batch size 2, gradient accumulation 4, max 5 epochs, early stopping (patience 2 on validation WER).
- Wire up TensorBoard logging and checkpoint saving to Drive.
- Do a short smoke-test run (a few dozen steps) to confirm the loop works end-to-end without crashing.
**Output:** A verified, checkpointable training script.
**Protects:** Technical Quality (Data Prep, Algorithm Selection), Reproducibility.

## Plan 5 — ASR Fine-Tuning Execution
**Objective:** Actually train the model to completion.
- Run full fine-tuning to the early-stopping criterion or 5-epoch ceiling, resuming from checkpoints across Colab sessions as needed.
- Monitor TensorBoard curves (train/val loss, WER over epochs); screenshot the dashboard.
- Select the best checkpoint by validation WER.
**Output:** A fine-tuned Whisper-Small checkpoint plus TensorBoard logs.
**Protects:** Technical Quality (15+10), Difficulty (15).

## Plan 6 — Speaker Gender Classification (Second Task)
**Objective:** Build and train the second, distinct task required by the brief.
- Extract fixed-length embeddings from the Whisper encoder (mean-pooled hidden states) for each utterance.
- Train a lightweight classifier head (logistic regression or small MLP) on these embeddings to predict speaker gender.
- Evaluate with accuracy, precision/recall, F1, and a confusion matrix; compare against the Plan 3 baseline.
**Output:** A trained gender classifier and its evaluation metrics, directly comparable to the baseline.
**Protects:** Difficulty (15) — satisfies "more than one task" — and Discussion of Findings (5).

## Plan 7 — Evaluation and Error Analysis
**Objective:** Turn raw metrics into the critical analysis the marking criteria reward.
- Compute final WER/CER (baseline vs fine-tuned) for ASR; finalize classification metrics for gender prediction.
- Categorize ASR errors: substitutions, insertions, deletions, phonetic confusion, OOV words, code-switching, pronunciation variation. Build example tables.
- Analyze classification errors: which speakers/genders/utterance types get misclassified and why.
**Output:** Error-analysis tables, example transcriptions, and a written interpretation tying results back to the research question.
**Protects:** Discussion of Findings (5), Depth of Information (15).

## Plan 8 — Optimization and Deployment
**Objective:** Show the practical, real-world side of the pipeline.
- Convert the fine-tuned Whisper model to CTranslate2 format; benchmark inference latency via Faster-Whisper (CPU and GPU if available).
- Build the Streamlit app: audio upload, transcription, gender prediction, WER/CER summary, inference time display.
- Screenshot the running app performing both tasks on sample inputs.
**Output:** A working local demo and deployment benchmark numbers.
**Protects:** Difficulty (15), Originality (10) — the multi-task deployment is the differentiator here.

## Plan 9 — Evidence and Appendix Assembly
**Objective:** Satisfy Reproducibility (10 marks) completely before writing begins.
- Collect and organize every screenshot (environment, each experiment, TensorBoard, Streamlit app), each showing device/software.
- Assemble the complete code (preprocessing, training, evaluation, classification, conversion, Streamlit) into an appendix-ready format, clearly marking any code adapted from external sources.
- Cross-check against the brief: no external links anywhere except the dataset link; the 1-page proposal reproduced exactly in the appendix.
**Output:** A complete, self-contained appendix package ready to paste into the report.
**Protects:** Reproducibility (10), Style and Format (10).

## Plan 10 — Report Writing and Final Submission
**Objective:** Produce and submit the compliant 4500-word report.
- Write the report in brief order: title, introduction, background/related work, problem/method, experimental section, discussion, conclusion, references, appendices.
- Compress all prior findings into the strict word limit — full technical detail lives in the appendix, not the main body.
- Insert the exact 1-page proposal into the appendix, verbatim.
- Proofread for language standards, correct all citations, check figure/table captions credit any non-original material.
- Export to the required file format and naming (NAME_studentID), submit via Campus 4.0.
**Output:** Final submitted coursework.
**Protects:** Information Presentation (10), Style and Format (10), Project Proposal marks (10), and ties together all Technical Quality marks (45).

---

### Sequencing note
Plans 4–6 are the only ones with meaningful GPU wall-clock time (multi-hour training runs); everything else is engineering, analysis, and writing that can happen alongside or right after. If Colab session limits bite, Plan 5 is the one to buffer extra time around.
