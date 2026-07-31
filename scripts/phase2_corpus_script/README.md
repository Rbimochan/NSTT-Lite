# Phase 2 — corpus script (pre-recording)

This is the scripted-text stage of Phase 2 corpus collection, not the finished
corpus. It cannot include audio — that requires a human recording session (or
a decision to use TTS, which the ASR README explicitly avoids relying on for
the base checkpoint).

## What's here
- `term_vocabulary.csv` — 86 English medical terms across 7 categories
  (disease, symptom, vital, lab_value, imaging, medication, procedure, admin),
  tagged with whether they appeared in the original 50-line pilot set.
- `dialogue_script.csv` — 55 new Nepali-English code-switched dialogue lines
  (25 female-speaker, 30 male-speaker), each embedding 1–2 terms from the
  vocabulary in varied sentence position (start/mid/end), covering terms not
  in the pilot set (hypertension, stroke, cancer, dialysis, biopsy, etc).
  `audio_path` is blank and `recording_status` is `pending` for every row —
  these are lines to be read aloud, not existing recordings.

## What's still needed before this becomes a real manifest
1. Record each line (native Nepali speakers, doctor/patient role per
   `speaker_role`) — this is a manual step, not something scriptable here.
2. Fill in `audio_path` and flip `recording_status` to `done` per line.
3. Run the recorded set through `src/manifests.py` / `src/splits.py` the same
   way `run_slr54_pipeline` does, to produce speaker-disjoint train/val/test
   manifests alongside the existing pilot data.

Combined with the 50-line pilot set already in
`reports_medical_pilot_baseline.csv`, this brings the scripted corpus to 105
lines — toward the 150–250 target set in `plans/plan2.html`.

Lives under `scripts/` rather than `data/` because `data/` is gitignored in
this repo (bulk audio/manifests only); these are small tracked text files.
