# NSTT-Lite: Nepali Speech-to-Text with Whisper-Small Fine-Tuning and Speaker Gender Classification

*Draft — Step 14, compiled from real experimental results in `reports/`. Word count target for final submission: 4500 words; this draft prioritizes completeness of real numbers over brevity and should be trimmed before submission.*

## 1. Introduction

Automatic speech recognition (ASR) for Nepali, a low-resource language with limited annotated speech data, remains substantially behind high-resource languages in off-the-shelf model quality. This project fine-tunes OpenAI's Whisper-small on a subset of the OpenSLR54 Nepali ASR corpus and evaluates the improvement over zero-shot performance. As a second, distinct task, it also builds a speaker-gender classifier on top of the fine-tuned model's encoder representations, and benchmarks a production-oriented deployment path via CTranslate2/Faster-Whisper.

## 2. Related Work

**Whisper and multilingual ASR.** Whisper (Radford et al., 2022) is a multilingual, multitask sequence-to-sequence ASR model trained on 680k hours of weakly supervised audio scraped from the web. Its training mix is heavily skewed toward high-resource languages (English, Spanish, and other languages with abundant transcribed audio); Nepali receives a comparatively small share, so weak zero-shot performance on Nepali is an expected consequence of the pretraining distribution rather than a surprising result. This project's zero-shot baseline (Section 6.1, WER 221.76%) is consistent with that expectation and with prior reports of Whisper-small performing poorly on Devanagari-script, morphologically rich low-resource languages without fine-tuning.

**Fine-tuning recipes.** The standard adaptation path for a specific low-resource language is supervised fine-tuning on paired audio-transcript data using the Hugging Face `Seq2SeqTrainer`, as documented in Hugging Face's "Fine-Tune Whisper for Multilingual ASR" tutorial (cited in `src/training.py`, since this project's training loop follows that recipe directly: encoder-decoder cross-entropy loss, WER computed via greedy/beam decoding at eval time, and gradient accumulation to simulate a larger batch size than a single consumer GPU or CPU can hold in memory). This project does not deviate from that recipe algorithmically — the departures from the original plan are entirely about compute budget (Section 5), not method.

**Speaker-disjoint evaluation.** A recurring risk in speech corpora built from a small number of speakers reading many utterances each (as OpenSLR54 is) is that a random utterance-level train/test split lets the model partially memorize speaker-specific acoustic characteristics (accent, recording device, background noise profile) rather than learning transcription in general. The standard mitigation, used throughout the speech recognition literature, is to hold entire speakers out of training rather than splitting at the utterance level — the approach taken in `src/splits.py`'s `speaker_disjoint_split`. This project's Step 9 ablation (Section 6.2) is a direct, if confounded, empirical test of whether that mitigation matters for this specific corpus and scale.

**Gender labeling from acoustic features.** In the absence of annotated gender metadata, using mean fundamental frequency (F0) as a proxy for speaker gender is a long-standing heuristic in speech processing, grounded in the physiological fact that adult male vocal folds are typically longer and heavier than adult female vocal folds, producing a lower average F0 (commonly cited ranges: ~85–180Hz male, ~165–255Hz female, with substantial overlap and individual variation). This project uses that heuristic (via `librosa.pyin`, Section 3) precisely because OpenSLR54 provides no alternative; the well-known limitation of the heuristic — the overlapping ranges mean a nontrivial fraction of speakers near the boundary will be mislabeled regardless of threshold choice — is discussed as a likely contributor to the gender classifier's negative result in Section 7.

**Deployment.** CTranslate2 (Klein et al.) is a C++/Python inference engine that re-implements common transformer architectures with quantization and kernel fusion for faster CPU/GPU inference than the original PyTorch/Hugging Face implementation; Faster-Whisper wraps it specifically for Whisper models. This project's deployment benchmark (Section 6.6) uses this runtime as the roadmap specifies, to demonstrate the latency difference between a training-oriented `transformers.generate()` call and a deployment-oriented inference path.

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

The trained classifier **underperformed the trivial majority-class baseline** by 30.4 points. The confusion matrix (`reports/gender_classifier_results.json`) makes the failure mode concrete:

| True \ Predicted | Male | Female |
|---|---|---|
| Male (n=1,259) | 637 | 622 |
| Female (n=416) | 303 | 113 |

The classifier gets barely over half of true male utterances right (637/1,259 = 50.6% male recall) — worse than a coin flip would suggest given the class imbalance — and misses nearly three-quarters of true female utterances (only 113/416 = 27.2% recall), despite predicting "female" for 735 of the 1,675 test utterances overall (622 + 113). In other words, the model is not simply defaulting to one class; it is making genuinely inconsistent predictions that happen to land worse than always guessing the majority class. This pattern — errors spread roughly evenly rather than concentrated in one direction — is more consistent with the classifier picking up noise than with it learning a real but imperfectly-labeled signal. Plausible causes, in descending order of how much of the gap each is likely to explain: (1) the Whisper encoder was fine-tuned for only 300 steps (~16% of one epoch), likely insufficient for its representations to become speaker-discriminative on top of whatever discriminative structure the pretrained encoder already had; (2) the F0 pseudo-labels are themselves noisy — Section 2's discussion of overlapping male/female F0 ranges means a nontrivial fraction of the 800 training labels and 1,675 test labels are likely wrong, which caps the achievable accuracy regardless of classifier quality, and could plausibly explain a same-order-of-magnitude accuracy loss; (3) the 800-utterance training set, projected into a several-hundred-dimensional mean-pooled embedding space, is a small sample-to-dimensionality ratio for logistic regression, risking overfitting to noise in the training labels specifically. This negative result is reported as-is rather than tuned until it looks better, and discussed further in Section 7.

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

Fine-tuning substantially improves Nepali ASR over zero-shot Whisper-small even at a fraction of the planned training budget: CER fell from 147.4% to 22.0%, and WER from 221.8% to 70.1%, after only 300 steps (~16% of one epoch) on 12,300 speaker-disjoint training utterances. This confirms the core premise of the project — that domain-specific fine-tuning on even a modest amount of in-language data closes a large fraction of the zero-shot gap for a low-resource language — while also showing that absolute performance at this training budget remains far from usable for a real transcription product. The two results are not in tension: they describe different points on the same learning curve, and the roadmap's original full 5-epoch/2000-step target (Section 5) sits much further along that curve than what was run here.

The own-voice test (Section 6.5) adds a qualitatively different data point: short, simple phrases similar in style to SLR54's read-speech utterances transcribe recognizably (WER as low as 0.50 on "तिम्रो नाम के हो"), while a long, prosodically different passage performs far worse (WER 1.00). This is consistent with a model that has partially learned the *acoustic-to-text mapping for short read Nepali utterances specifically*, rather than Nepali speech recognition in general — an important distinction for anyone evaluating this checkpoint against out-of-domain audio (spontaneous speech, phone-call audio, regional accents not represented in the 146-speaker training subset).

Three limitations should shape how the remaining results are read, in order of how much they constrain the claims this report can make:

1. **Gender labels are pseudo-labels, not ground truth** (Section 3). The gender-classification accuracy (Section 6.3) and any statement about "which speakers get misclassified" measures agreement with an F0 heuristic, not true gender. This is a ceiling on what the second task can demonstrate, not a footnote to caveat and move past: even a hypothetically perfect classifier could only ever be shown to perfectly recover the *heuristic*, not to have learned true gender. Any claim in this report about gender classification should be read as "classification of the F0-threshold pseudo-label" throughout, including in the Conclusion.
2. **The speaker-leakage ablation is confounded by training budget** (Section 6.2). Because the leaky-split run received a third of the training steps of the speaker-disjoint run, the observed 15-point WER gap cannot be attributed to leakage — if leakage inflates apparent performance as the literature generally expects, the true leakage effect is being *masked* by the leaky run's training deficit, meaning the real leakage-only gap (at matched step counts) could be larger, smaller, or in a different direction than what is reported here. This ablation should be treated as a demonstration that the harness for measuring leakage exists and is correctly wired (zero vs. nonzero `count_speaker_overlap`), not as evidence for or against the leakage hypothesis itself.
3. **The gender classifier's negative result is itself informative**, and its confusion-matrix breakdown (Section 6.3) points more toward "insufficient discriminative signal in a lightly fine-tuned encoder plus noisy labels" than toward "correct signal, wrong evaluation." Both remaining explanations — undertrained encoder representations and noisy pseudo-labels — point toward the same two remedies: more fine-tuning steps (closing the gap with remedy #1 above) and, independently, a manual verification pass on a random sample of speakers to estimate the pseudo-label error rate directly rather than inferring it indirectly from downstream classifier performance.

A further limitation, not specific to any one result: all evaluation numbers in Sections 6.1–6.4 are computed on a fixed 150-utterance subset of the 1,675-utterance test split (chosen for CPU wall-clock tractability, Section 5), not the full split. This keeps the different evaluations comparable to each other (same subset used throughout) but means the reported WER/CER carry more sampling variance than a full-test-set evaluation would, and should be read as indicative rather than final.

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
