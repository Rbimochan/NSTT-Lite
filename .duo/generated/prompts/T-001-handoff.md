# Duo Handoff — Claude → Cursor (state v1)
Task: T-001 — Plan 1: Environment and Data Acquisition (type: feature)

## Acceptance criteria
- Colab T4 GPU environment set up; transformers, datasets, torchaudio, tensorboard,
  ctranslate2, streamlit installed (reuse NSTT-Lite/requirements.txt, already pinned
  to known-good versions from the original NSTT project)
- OpenSLR SLR54 corpus available (reuse original NSTT Colab/Drive download rather
  than re-downloading ~8GB from scratch, if that Drive copy is accessible; otherwise
  re-run src/slr54.py download functions, which already skip existing files)
- File counts verified against SLR54 documentation (157,905 utterances expected per
  utt_spk_text.tsv)
- Screenshots captured: Colab runtime type (GPU), library versions, dataset download
  completing

## Implementation guidance
- **Reuse, don't rebuild.** `src/slr54.py` (download_tsv, download_zips,
  extract_zips, parse_utt_spk_text_tsv, index_audio_files, attach_audio_paths)
  already implements everything this task needs and was tested extensively on the
  original NSTT project (including fixing a Drive-I/O bottleneck and adding
  resumable/batched processing — see the original NSTT `.duo/handoffs/implementation-log.md`
  and `notebooks/01_data_prep.ipynb` for the working pattern).
- **Do not re-download the full ~8GB corpus if avoidable.** Check whether the
  original NSTT project's Google Drive location (`MyDrive/NSTT/data/raw/slr54/`)
  already has the extracted corpus, and either symlink/reuse it or point
  `raw_dir` at that same Drive path for NSTT-Lite, rather than downloading a second
  independent copy.
- Create `notebooks/00_setup.ipynb` for NSTT-Lite following the same structure as
  the original NSTT project's `00_setup.ipynb`: mount Drive, clone/pull this repo,
  `pip install -r requirements.txt`, GPU check, import smoke-test — but update the
  import list to also include `tensorboard`, `ctranslate2`, `faster_whisper`,
  `sklearn`.
- Known dependency landmine already solved: `transformers==4.49.0` requires
  `huggingface_hub>=0.26.0`; Colab's default huggingface_hub versions have at
  various points dropped the `is_offline_mode` export transformers imports at
  load time. `requirements.txt` already pins `huggingface_hub==0.27.1` to avoid
  this — do not loosen this pin without re-verifying compatibility.
- Known landmine: `torchaudio` must NOT be pinned in requirements.txt — Colab ships
  a matched torch/torchvision/torchaudio trio, and pinning torchaudio independently
  breaks that pairing (`operator torchvision::nms does not exist`).

## Attached
- `.duo/project-context.md`
- `.duo/project-state.json` (v1)
- `../NSTT-Lite_10_Plan_Roadmap.md` (canonical spec for all 10 plans)

## Human
Paste this into Cursor and activate dearcursor.
