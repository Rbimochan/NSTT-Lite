# Plan — XLS-R Nepali Baseline Audit

## Trajectory change

This project no longer targets Whisper-Small fine-tuning (that work — Plans
1–4 of the old NSTT-Lite roadmap — lives untouched in
[PR #1](https://github.com/Rbimochan/NSTT-Lite/pull/1) and
[PR #2](https://github.com/Rbimochan/NSTT-Lite/pull/2), not here). This branch
is single-purpose: benchmark `gagan3012/wav2vec2-xlsr-nepali` (a Hugging Face
XLS-R model) against `gauravparajuli/slr43` (OpenSLR-43), the same corpus it
was originally trained/self-evaluated on.

Direction going forward: keep the benchmark **lighter** (no training loop, no
deployment track, no second model) but make the number itself **more
accurate** — the current result is a 150-utterance slice, not the full corpus.

## Current state — DONE

- `scripts/run_xlsr_baseline.py` runs end to end: loads the model, loads
  OpenSLR-43, transcribes via greedy CTC decode, computes WER/CER.
- Result on 150 utterances: **WER 4.91%, CER 0.87%**, vs. the model's
  self-reported **5.97%** — `reports/xlsr_baseline_results.json`.
- `notebooks/xlsr_baseline_colab.ipynb` mirrors the script for Colab.
- `data/manifests/xlsr_openslr43_test_manifest.csv` tracks which utterances
  (by HF dataset index) were evaluated, without storing raw audio in git.

## Next steps

**STEP 1: Run the full corpus, not a slice**
`gauravparajuli/slr43` has 2,064 utterances total; the current result covers
150. Set `N_SAMPLES = None` in `scripts/run_xlsr_baseline.py` (or the notebook)
and re-run.
Checkpoint: `reports/xlsr_baseline_results.json` shows `num_utterances: 2064`,
and the WER/CER are recomputed on the full corpus, not a subsample.

**STEP 2: Record the result and compare**
Note whether the full-corpus WER differs meaningfully from the 150-sample
figure (4.91%) and from the self-reported 5.97%. A meaningful difference
either direction is worth reporting as-is — don't re-run repeatedly looking
for a "better" number.
Checkpoint: `reports/xlsr_baseline_results.json`'s `note` field states clearly
that this is now the full-corpus result, not a sample.

**STEP 3: Close out**
No further scope beyond this — no fine-tuning, no second dataset, no
deployment benchmark, per the explicit single-purpose decision already made.
Checkpoint: PR #3 (`plan4-continuation` → `main`) reflects the full-corpus
result and is ready for merge.
