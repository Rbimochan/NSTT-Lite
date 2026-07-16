# NSTT-Lite — Start Checklist

Run the **General pre-flight** once per Colab session, before any plan. Then for
whichever Plan you're running, do its **Before** check, do the work, then its
**After** check before marking the task done in `.duo/project-state.json`.

Canonical scope/acceptance-criteria source: `NSTT-Lite_10_Plan_Roadmap.md`.
Duo task tracking: `.duo/project-state.json` (T-001 … T-010, 1:1 with Plan 1 … 10).

---

## General pre-flight (every Colab session, before anything else)

- [ ] `Runtime → Change runtime type` → correct accelerator for this step (T4 GPU
      for training/eval; **None/CPU is fine and preferred for pure data-prep** to
      avoid burning GPU quota you'll need later)
- [ ] Mount Drive: `from google.colab import drive; drive.mount("/content/drive", force_remount=True)`
- [ ] Confirm repo present at `/content/drive/MyDrive/NSTT-Lite` (clone/pull, don't
      re-upload manually — `git clone`/`git pull` keeps it in sync with GitHub)
- [ ] `pip install -q -r requirements.txt` — expect harmless yellow warnings about
      `gradio`/`google-colab`/`fsspec`/`diffusers`; **only stop on a red `ERROR:
      ResolutionImpossible`**, not on dependency-conflict warnings
- [ ] `Runtime → Restart session` after any install, then re-run the GPU-check +
      working-directory cells only (not the install cell again)
- [ ] Import smoke-test passes: `transformers`, `datasets`, `torchaudio`, `librosa`,
      `evaluate`, `jiwer`, `accelerate`, `tensorboard`, `ctranslate2`,
      `faster_whisper`, `sklearn` all import with no red traceback

**Known landmines already solved — don't undo these:**
- Don't pin `torchaudio` in `requirements.txt` — Colab ships a matched
  torch/torchvision/torchaudio trio; pinning breaks it (`operator torchvision::nms
  does not exist`).
- `transformers==4.49.0` + `huggingface_hub==0.27.1` is a verified-working exact
  pair — don't loosen to a range; Colab's own default huggingface_hub has
  intermittently dropped the `is_offline_mode` export transformers needs.
- Any loop over many individual files (audio processing, downloads) run against a
  **Drive-mounted path is slow** (~2 it/s) — process to local `/content/...` disk
  first (~50+ it/s), then bulk-copy to Drive.
- Any long-running loop must be **resumable**: flush progress to a Drive-persisted
  file every N items, and check that file first on re-run. Colab disconnects and
  free-tier GPU quota limits are routine here, not exceptional — assume you will be
  interrupted, not that you won't.

---

## Plan 1 — Environment and Data Acquisition
**Before:** general pre-flight above. Check whether the original NSTT project's
Drive copy of SLR54 (`MyDrive/NSTT/data/raw/slr54/`) already exists — reuse rather
than re-downloading ~8GB.
**After:** all 16 zips + `utt_spk_text.tsv` present and not corrupted (each zip
≥50MB); GPU type, library versions, and completed download all screenshotted.

## Plan 2 — Preprocessing and Dataset Split
**Before:** Plan 1's corpus present and verified.
**After:** manifest has `audio_path`, `transcript`, `speaker_id`, **`gender`**,
`split` columns for every row; `count_speaker_overlap(splits) == 0`; total subset
duration is 10-20 hours (not the full ~165hr corpus); resample confirmed 16kHz mono
on a spot-check of a few files.

## Plan 3 — Baseline Evaluation (Zero-Shot)
**Before:** Plan 2's manifests exist.
**After:** zero-shot WER/CER logged and not suspiciously 0% or 100%; trivial
gender-classification baseline accuracy logged; both runs screenshotted with
device info visible.

## Plan 4 — ASR Fine-Tuning Setup
**Before:** Plan 3's baselines recorded (so you have something to compare against).
**After:** a few-dozen-step smoke test completes with no crash; TensorBoard log dir
populated; a checkpoint file exists on Drive from the smoke run.

## Plan 5 — ASR Fine-Tuning Execution
**Before:** Plan 4's smoke test passed. **GPU quota check** — this is the
multi-hour step; confirm you have enough quota/time before starting, since a
mid-run disconnect should only cost one checkpoint interval, not the whole run (see
resumability note above).
**After:** training reached early-stopping or the 5-epoch ceiling; best checkpoint
identified by validation WER; TensorBoard loss/WER curves screenshotted.

## Plan 6 — Speaker Gender Classification
**Before:** Plan 5's best checkpoint identified and loadable.
**After:** classifier trained on mean-pooled encoder embeddings; accuracy/precision/
recall/F1/confusion matrix reported and compared explicitly against Plan 3's
baseline number.

## Plan 7 — Evaluation and Error Analysis
**Before:** Plans 5 and 6 both have final numbers.
**After:** final WER/CER (baseline vs fine-tuned) tabulated; ASR error-category
table produced (substitutions/insertions/deletions/phonetic/OOV/code-switching);
gender-classification error breakdown by speaker/utterance type; written
interpretation drafted.

## Plan 8 — Optimization and Deployment
**Before:** Plan 7's final checkpoint and metrics finalized (don't benchmark/deploy
an interim checkpoint).
**After:** CTranslate2 conversion succeeds; Faster-Whisper latency benchmarked
(CPU, and GPU if available); Streamlit app shows transcription + gender prediction
+ WER/CER + inference time; screenshotted running on sample inputs.

## Plan 9 — Evidence and Appendix Assembly
**Before:** Plans 1-8 all screenshotted along the way — this step assembles, it
doesn't generate new screenshots retroactively for steps you skipped.
**After:** every screenshot present and organized; full code assembled
appendix-ready with external sources marked; no external links outside the dataset
link anywhere; 1-page proposal reproduced verbatim in the appendix draft.

## Plan 10 — Report Writing and Final Submission
**Before:** Plan 9's evidence package complete — do not start writing prose against
incomplete evidence.
**After:** report is ≤4500 words, proposal reproduced verbatim, citations checked,
correct file naming (`NAME_studentID`), submitted via Campus 4.0.

---

### Sequencing note (from the roadmap)
Plans 4-6 carry the meaningful GPU wall-clock time; everything else is engineering,
analysis, and writing that can happen alongside or right after. If Colab session
limits bite, Plan 5 is the one to buffer extra time around.
