# NSTT-Lite — Nepali Speech-to-Text (Rescoped)

A leaner rebuild of the original [NSTT](https://github.com/Rbimochan/NSTT) project,
restructured around a stricter 100-mark academic rubric. Adds a second task
(speaker gender classification) and production-style deployment
(CTranslate2/Faster-Whisper) alongside the original Whisper-Small Nepali ASR
fine-tuning work.

## Start here
- **`NSTT-Lite_10_Plan_Roadmap.md`** — the canonical spec: 10 sequential plans from
  environment setup to final submission, each tagged with which marks it protects.
- **`start.md`** — a before/after checklist for every plan, including every Colab
  dependency/Drive-I/O/resumability landmine already solved on the original NSTT
  project.
- **`.duo/`** — the Claude & Cursor Duo Framework workspace tracking task state
  (`project-state.json`), architecture decisions, and implementation history.

## Setup
```bash
git clone https://github.com/<your-username>/NSTT-Lite.git
cd NSTT-Lite
pip install -r requirements.txt
```

Or in Google Colab, mount Drive and clone into it:
```python
from google.colab import drive
drive.mount("/content/drive")
!git clone https://github.com/<your-username>/NSTT-Lite.git /content/drive/MyDrive/NSTT-Lite
```

## Status
Currently at Plan 1 of 10 (environment + data acquisition) — fine-tuning and
evaluation haven't run yet, so no WER/accuracy numbers exist yet. See
`.duo/project-state.json` for live task status (T-001 … T-010, mapped 1:1 to
Plan 1 … Plan 10); results will be added here once Plan 3 (baseline evaluation)
and later fine-tuning runs complete.
