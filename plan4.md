# Nepali ASR (Whisper Fine-tuning) — Plan 4

Supersedes plan3.md. Steps 1-14 of plan3.md are DONE, confirmed on disk (see PR #2).
Everything in plan3 ran at **reduced local scale** (CPU/MPS, 300 fine-tuning steps,
100-step ablation) to stay tractable on a laptop. Plan 4 closes that gap and takes
the project from "pipeline proven end-to-end" to "submission-ready coursework."

Known carry-forward caveats (must stay in the report until resolved or explicitly
accepted as permanent limitations):
- `gender` is an F0-threshold (165Hz) pseudo-label, not ground truth — SLR54 has no
  gender metadata at all.
- The Step 9 ablation compared unequal step counts (300 vs 100) — the WER gap
  conflates speaker leakage with training budget, not a controlled comparison.
- The gender classifier (44.78%) underperforms the majority-class baseline (75.16%).
- `src/error_analysis.py` / `src/evaluation.py` still emit stale "T-003 smoke
  checkpoint" caption text in their markdown exports — cosmetic, unfixed.

---

STEP 1: Full-scale training on Colab T4
Re-run fine-tuning per the original Plan 4/5 spec (AdamW, FP16, batch 2,
grad-accum 4, max 5 epochs / 2000 steps, early stopping patience 2 on val WER) on
a Colab T4 GPU instead of local CPU/MPS — the 300-step local run was a proof of
loop correctness, not a real training budget.
Checkpoint: training reaches early-stopping or the 5-epoch ceiling; best checkpoint
selected by validation WER; TensorBoard curves screenshotted with GPU visible.

STEP 2: Re-evaluate against Step 1's checkpoint
Re-run the Step 5 baseline comparison and Step 9 ablation using the new full-scale
checkpoint, on the full 1,675-utterance test split (not the 150-utterance subset
used for local tractability).
Checkpoint: new WER/CER numbers exist for baseline vs. full-scale fine-tuned;
ablation re-run at matched step counts on both sides (fixing plan3's confound).

STEP 3: Re-run gender classification on the full-scale encoder
Re-extract mean-pooled embeddings from the Step 1 checkpoint (not the 300-step one)
and retrain/re-evaluate the classifier. If it still underperforms the majority-class
baseline, keep that as a reported negative result — do not tune until it "looks good."
Checkpoint: classifier metrics recomputed on the full-scale checkpoint; comparison
to majority-class baseline stated either way.

STEP 4: Refresh error analysis and own-voice test
Re-run error categorization and the own-voice test against the Step 1 checkpoint.
Checkpoint: reports/error_samples.*, reports/error_categories.*, and
reports/own_voice_test.json all reflect the full-scale checkpoint, not the 300-step one.

STEP 5: Re-run deployment against the full-scale checkpoint
Re-convert to CTranslate2, re-benchmark latency, re-point the Streamlit app at the
new checkpoint.
Checkpoint: reports/ctranslate2_benchmark.json and scripts/app.py reflect the
full-scale checkpoint.

STEP 6: Screenshot and evidence capture
Capture every screenshot the original roadmap requires and that Colab-only steps
made impossible to get locally: Colab runtime type (GPU), library versions,
dataset download, TensorBoard dashboard (Step 1), Streamlit app running on a real
sample (Step 5).
Checkpoint: a screenshots/ folder (or equivalent) has one image per required step,
each showing visible device/software info.

STEP 7: Expand the report to submission length
reports/report_draft.md is currently ~1,689 words against a 4,500-word limit —
expand Related Work, Methodology, and Discussion with the Step 1-6 numbers, without
padding. Full technical detail (hyperparameter tables, full error tables, code
listings) belongs in the appendix, not the main body.
Checkpoint: report body is close to but under 4,500 words; every section
(Introduction, Related work, Dataset, Methodology, Experimental setup, Results,
Discussion, Conclusion) has real content, not placeholders.

STEP 8: Assemble the appendix
Collect: all screenshots from Step 6, complete code listings (clearly marking any
code adapted from external sources, e.g. the HF Whisper fine-tuning tutorial cited
in src/training.py), and the exact 1-page project proposal reproduced verbatim.
Checkpoint: appendix package satisfies the reproducibility checklist in start.md —
no external links anywhere except the SLR54 dataset link.

STEP 9: Final proofread and submission
Proofread for language standards, verify all citations, check figure/table
captions credit non-original material, export to the required filename format
(NAME_studentID), submit via Campus 4.0.
Checkpoint: final file exported and submitted; this is the human-owned step per
project-state.json (T-010) — Claude's role is compiling/organizing, not submitting
on the user's behalf.

---

## Checker Prompt

```
Go through this project folder and check each step in plan4.md against
what actually exists on disk. For each step, answer only one of:
DONE / PARTIAL (explain what's missing) / NOT DONE.
Do not assume a step is done because a script for it exists — a step is
only DONE if it has actually been run and its checkpoint condition is
verifiably true on disk right now.
For Step 1, confirm the checkpoint was actually trained on a GPU (check for
CUDA references in logs/config, not just that a checkpoint file exists) --
a checkpoint produced on local CPU/MPS does not satisfy this step.
For Step 2, confirm the ablation in this step used matched step counts on
both sides, unlike plan3's confounded comparison.
For Step 7, do a real word count against reports/report_draft.md and state
the number, don't estimate.
Flag any step whose checkpoint condition is false even if a later step's
code already exists. Output as a table:
Step | Status | Evidence | What's needed to reach DONE.
```
