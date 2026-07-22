# XLS-R Nepali Baseline Audit

A single-purpose project: evaluate `gagan3012/wav2vec2-xlsr-nepali` (a Hugging
Face XLS-R/wav2vec2 model) on **OpenSLR-43** (`gauravparajuli/slr43`) — the same
corpus it was originally trained and self-evaluated on — to sanity-check its
published **5.97% WER** claim.

This is not a fine-tuning project. No training happens here — this repo only
runs inference with the existing checkpoint and measures WER/CER.

## Finding

| Metric | Self-reported | Measured (this audit, 150 utterances) |
|---|---|---|
| WER | 5.97% | **4.91%** |
| CER | — | **0.87%** |

The measured number is close to (in fact slightly better than) the published
figure, on a fixed-seed slice of the same corpus. This is not an independent
held-out test set — OpenSLR-43 ships only a single `train` split upstream, so
there is no separate test partition to evaluate against.

OpenSLR-43 has no speaker or gender metadata (it's a single-speaker, female
corpus) — no demographic breakdown is possible or attempted.

## Repo map

```
src/
  xlsr_baseline.py    model/dataset loading, CTC greedy decode, manifest writer
  wer_metrics.py       minimal WER/CER computation (jiwer)
scripts/
  run_xlsr_baseline.py runnable entry point
notebooks/
  xlsr_baseline_colab.ipynb   same flow, for running on Colab (e.g. to try the
                              full 2,064-utterance corpus rather than a 150 slice)
data/manifests/
  xlsr_openslr43_test_manifest.csv   references the HF dataset by index — raw
                                     audio is never stored here
reports/
  xlsr_baseline_results.json   the result, written fresh each run
```

## Setup

```bash
git clone https://github.com/Rbimochan/NSTT-Lite.git
cd NSTT-Lite
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_xlsr_baseline.py
```

Downloads (first run only): the `gagan3012/wav2vec2-xlsr-nepali` model
(~1.2GB) and the `gauravparajuli/slr43` dataset (~935MB), both from Hugging
Face. Evaluates 150 utterances by default; edit `N_SAMPLES` in
`scripts/run_xlsr_baseline.py` (or `notebooks/xlsr_baseline_colab.ipynb`) to
change that, or set it to `None` to run the full corpus.

Output: `reports/xlsr_baseline_results.json` (WER, CER, utterance count,
comparison against the self-reported 5.97% figure) and
`data/manifests/xlsr_openslr43_test_manifest.csv` (the evaluated slice's
transcripts, referencing the HF dataset by index).
