# NSTT-Lite: Nepali Speech-to-Text with Whisper-Small Fine-Tuning and Speaker Gender Classification

*Draft — Step 14, compiled from real experimental results in `reports/`. Word count target for final submission: 4500 words; this draft prioritizes completeness of real numbers over brevity and should be trimmed before submission.*

## 1. Introduction

Automatic speech recognition (ASR) for Nepali, a low-resource language with limited annotated speech data, remains substantially behind high-resource languages in off-the-shelf model quality. This project fine-tunes OpenAI's Whisper-small on a subset of the OpenSLR54 Nepali ASR corpus and evaluates the improvement over zero-shot performance. As a second, distinct task, it also builds a speaker-gender classifier on top of the fine-tuned model's encoder representations, and benchmarks a production-oriented deployment path via CTranslate2/Faster-Whisper.

## 2. Related Work

Whisper (Radford et al., 2022) is a multilingual, multitask sequence-to-sequence ASR model trained on 680k hours of weakly supervised audio; Nepali is a lower-resource language in its pretraining mix, so zero-shot performance is expected to be weak. Fine-tuning recipes for low-resource Whisper adaptation follow the standard Hugging Face `Seq2SeqTrainer` approach (Hugging Face, "Fine-Tune Whisper" blog, cited in `src/training.py`). CTranslate2/Faster-Whisper (Klein et al.) provide an efficient inference runtime for deployed Whisper models.

## 3. Dataset

**Source:** OpenSLR54, a crowdsourced Nepali ASR corpus (Kjartansson et al., SLTU 2018), CC BY-SA 4.0.

**Corpus used:** 55,980 of the corpus's 157,905 utterances were downloaded (96 of 256 two-character speaker-hash subdirectories), from which a **15.03-hour, speaker-disjoint subset of 15,459 utterances across 146 speakers** was processed: audio resampled to 16kHz mono, transcripts NFC-normalized (Devanagari), utterances outside 0.5–30s duration dropped. Split: train=12,300 / val=1,484 / test=1,675 (80/10/10 by speaker, confirmed **zero speaker overlap** between splits).

**⚠️ Gender label caveat (must be read before Section 6/7's gender-classification results):** OpenSLR54 ships **no gender metadata whatsoever** — verified against both the downloaded corpus files and the live openslr.org/54 listing. The `gender` field used throughout this project is an **acoustic pseudo-label**, derived per-speaker from mean fundamental frequency (F0) via librosa pYIN, thresholded at 165Hz (below → male, above → female — the standard heuristic midpoint between typical adult male ~85–180Hz and female ~165–255Hz ranges). This is **not verified ground truth**. Every gender-classification number in this report measures agreement with this heuristic, not true gender. Resulting distribution: 8,146 male / 7,313 female (15,459 total).

## 4. Methodology

- **ASR fine-tuning:** Whisper-small, `Seq2SeqTrainer`, AdamW, batch size 2, gradient accumulation 4 (effective batch 8), learning rate 1e-4, FP16 disabled locally (no CUDA on this dev machine — Apple Silicon MPS/CPU only), 300 training steps (~16% of one epoch over the 12,300-utterance train split; full 5-epoch training was not run locally due to CPU-only wall-clock constraints — see Limitations).
- **Gender classification:** mean-pooled Whisper encoder hidden states (last layer) as fixed-length embeddings, logistic regression head, trained on 800 train-split utterances, evaluated on the full 1,675-utterance test split.
- **Ablation:** a "leaky" utterance-random split (ignoring speaker identity) was built from the same 15,459 records, to test whether speaker-disjoint splitting matters for this dataset.
- **Deployment:** fine-tuned checkpoint converted to CTranslate2 (int8, CPU), served via a Streamlit app with transcription + gender prediction + latency display.

## 5. Experimental Setup

All experiments ran locally on Apple Silicon (CPU/MPS, no CUDA), Python 3.13, in a local virtualenv per `requirements.txt` (with `ctranslate2` locally installed as 4.8.1 rather than the Colab-pinned 4.5.0, which has no Python 3.13 wheel). This departs from the original Colab T4 GPU plan; wall-clock constraints on CPU training are the primary reason later steps (full 5-epoch training, larger gender-classifier training set) were scaled down rather than run to the roadmap's original scope.

## 6. Results

### 6.1 ASR: zero-shot baseline vs. fine-tuned

| Model | WER | CER | Utterances |
|---|---|---|---|
| Whisper-small, zero-shot | 221.76% | 147.39% | 150 (test split) |
| Whisper-small, fine-tuned (300 steps, speaker-disjoint) | **70.14%** | **21.98%** | 150 (same subset) |

Fine-tuning produced a large, genuine improvement — CER dropped by roughly 6.7×. Absolute WER remains high by high-resource-language standards, consistent with only 300 steps (~16% of an epoch) of fine-tuning on a 15-hour subset.

### 6.2 Speaker-disjoint vs. leaky-split ablation

| Split | Train steps | WER | CER |
|---|---|---|---|
| Speaker-disjoint (real) | 300 | 70.14% | 21.98% |
| Leaky (utterance-random) | 100 | 85.14% | 28.94% |

**Caveat:** the leaky run was trained for only 100 steps vs. 300 for the disjoint run (a local tractability tradeoff), so this −15-point WER gap **conflates speaker leakage with unequal training budget** and should not be read as a clean leakage effect — if anything, leakage would be expected to *help* the leaky split's apparent score, and it did not, because the step-count deficit dominated. A controlled version of this ablation (equal steps) is needed before drawing a leakage conclusion; this is flagged as future work, not a finding.

### 6.3 Gender classification (second task)

| Model | Accuracy | Precision (female) | Recall (female) | F1 (female) |
|---|---|---|---|---|
| Majority-class baseline (predict "male" always) | 75.16% | — | — | — |
| Logistic regression on encoder embeddings | **44.78%** | 15.37% | 27.16% | 19.64% |

The trained classifier **underperformed the trivial majority-class baseline** by 30.4 points. Plausible causes: (1) the Whisper encoder was only fine-tuned for 300 steps, so its representations may not yet be strongly speaker-discriminative; (2) the F0 pseudo-labels are themselves noisy (see Section 3 caveat) — the classifier may be learning a real acoustic signal that simply doesn't align well with the noisy heuristic label it's being scored against; (3) the 800-utterance training set is small relative to the embedding dimensionality. This negative result is reported as-is and discussed further in Section 7.

### 6.4 Error analysis

Of 150 fine-tuned-model test transcriptions, categorized heuristically:

| Category | Count | % |
|---|---|---|
| OOV / rare vocabulary | 140 | 93.3% |
| Phonetic confusion | 134 | 89.3% |
| Noise/degradation (near-total mismatch) | 58 | 38.7% |
| Dialect/accent (heavy deletions) | 3 | 2.0% |
| Other | 8 | 5.3% |

(Note: the underlying script's markdown template still carries a stale caption referring to "the T-003 smoke checkpoint" — that refers to an earlier development milestone; the actual numbers above are from the real 300-step fine-tuned checkpoint, not a 3-step smoke test.)

Example (real, from `reports/error_samples.md`):

| Reference | Hypothesis | Categories |
|---|---|---|
| अतिरिक्त सुविधाहरूको | अतिरिक्त सूइदाहरूको | OOV, phonetic confusion |
| अधिक भारी बाउन्ने | अदिक बारी बाउन्ने | OOV, phonetic confusion |

### 6.5 Own-voice test (real self-recorded clips)

9 short Nepali phrases, recorded on iPhone, hand-transcribed, run through the fine-tuned checkpoint:

| Clip | WER | CER |
|---|---|---|
| Timro naam k ho ("तिम्रो नाम के हो") | 0.50 | 0.06 |
| Maile kaam paina | 0.67 | 0.15 |
| Parsi ghar farkini ho | 0.75 | 0.32 |
| Mero daad dukhekl xa | 0.75 | 0.35 |
| Voli chutti ho | 0.67 | 0.43 |
| Ka jan lako | 1.00 | 0.53 |
| Ma ghar jana lako | 1.00 | 0.60 |
| Voli kata jane | 1.00 | 0.46 |
| para prabhu timro (long passage) | 1.00 | 0.94 |

Short, simple phrases transcribe recognizably; the long passage performs worst, consistent with the model being fine-tuned on short read-speech utterances rather than long spontaneous passages.

### 6.6 Deployment benchmark

CTranslate2 (int8, CPU) inference latency on 3 real clips: **1.11–1.31 seconds** per clip (short phrases, ~1-3s audio), a substantial speedup over the unconverted Hugging Face `generate()` path used elsewhere in this report. The CT2 conversion required temporarily bumping `transformers` past the Colab-pinned 4.49.0 (a version incompatibility between locally-installed `ctranslate2==4.8.1` and `transformers==4.49.0` — the pin was restored immediately after conversion).

## 7. Discussion

Fine-tuning substantially improves Nepali ASR over zero-shot Whisper-small even at a fraction of the planned training budget, confirming that domain-specific fine-tuning is worthwhile even under significant compute constraints. However, three limitations should shape how the other results are read:

1. **Gender labels are pseudo-labels, not ground truth** (Section 3). The gender-classification accuracy (Section 6.3) and any downstream discussion of "misclassified speakers" measures agreement with an F0 heuristic, not true gender — a fundamental ceiling on what can be claimed from this task, not just a footnote.
2. **The speaker-leakage ablation is confounded by training budget** (Section 6.2) — the reported WER gap cannot be attributed to leakage alone.
3. **The gender classifier's negative result** (underperforming a trivial baseline) is itself informative: it suggests either the pseudo-labels are too noisy to learn from with this little data, or 300 fine-tuning steps is insufficient for the encoder to develop speaker-discriminative representations — both worth flagging as an open question rather than a solved second task.

## 8. Conclusion

This project fine-tuned Whisper-small on a real, speaker-disjoint 15-hour Nepali subset, reducing CER from 147.4% to 22.0% relative to zero-shot. It built a gender-classification second task on encoder embeddings against an explicitly-labeled acoustic pseudo-label (since no ground truth exists in the source corpus), which did not beat a trivial baseline — a genuine negative result reported transparently rather than concealed. Deployment via CTranslate2 achieved sub-1.5-second CPU inference latency. The main avenues for improvement are: completing the originally planned full 5-epoch/2000-step training run (compute-permitting, ideally on the Colab T4 GPU as originally scoped), running the speaker-leakage ablation at matched step counts, and — if the gender task is retained — either manually verifying a sample of the F0 pseudo-labels or increasing classifier training data.

## References

- Radford, A. et al. "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper), 2022.
- Kjartansson, O. et al. "Crowd-Sourced Speech Corpora for Javanese, Sundanese, Sinhala, Nepali, and Bangladeshi Bengali", SLTU 2018 (OpenSLR54).
- Hugging Face, "Fine-Tune Whisper for Multilingual ASR with Transformers" (blog, adapted in `src/training.py`).
- jiwer library documentation (character alignment used in `src/error_analysis.py`).

## Appendix pointers (for Step 14 continuation / evidence assembly)

- `reports/wer_cer_results.md`, `reports/baseline_asr_examples.jsonl` — Section 6.1
- `reports/ablation_wer_gap.json` — Section 6.2
- `reports/baseline_gender.json`, `reports/gender_classifier_results.json` — Section 6.3
- `reports/error_categories.md`, `reports/error_samples.md` — Section 6.4
- `reports/own_voice_test.json` — Section 6.5
- `reports/ctranslate2_benchmark.json` — Section 6.6
- `src/*.py`, `scripts/*.py` — full code, all adapted-from-tutorial sections cited in module docstrings
