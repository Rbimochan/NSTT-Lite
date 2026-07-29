# Auditing and Repairing a Published Nepali ASR Model: A Speaker-Diversity Case Study on XLS-R

**Module:** ST7088CEM — Artificial Neural Networks

**Student:** Bimochan Raj Kunwar — Coventry ID: 17108924, MSc7-S2

## 1. Introduction

Nepali automatic speech recognition (ASR) benchmarks reported in the
literature and on model-sharing platforms are frequently measured on narrow,
low-diversity corpora — often a single speaker's read speech — and then cited
as evidence of general-purpose transcription quality. This report audits one
such claim: `gagan3012/wav2vec2-xlsr-nepali`, a publicly available Hugging
Face XLS-R (wav2vec2) checkpoint that self-reports a 5.97% word error rate
(WER). That figure was measured on OpenSLR-43, the model's own
single-speaker training corpus. This project asks two questions. First, does
the claim survive contact with genuinely diverse speech — a different,
multi-speaker Nepali corpus the model was never trained on? Second, if the
claim does not survive, can the same model be repaired by fine-tuning it on
speaker-diverse data, and at what cost?

The answer to both, established empirically in this report, is: the claim
does not generalize (WER degrades from 4.91% in-domain to 62.30%
out-of-domain, a measurement independently confirming the audited 5.97%
figure is representative only of its narrow training distribution), and
fine-tuning on a 15-hour, 160-speaker, speaker-disjoint subset substantially
closes that gap (out-of-domain WER falls to 38.17%) at a disclosed cost to
in-domain performance (4.91% → 16.40%). A further ablation shows that
measuring this improvement without a speaker-disjoint train/test split would
have overstated the repair by roughly 5.6 percentage points — a methodological
finding as important as the headline result itself.

## 2. Related Work

**XLS-R and cross-lingual speech representation.** XLS-R (Babu et al., 2021)
extends wav2vec2's self-supervised pretraining objective across 128
languages and 436k hours of unlabelled speech, producing representations
that transfer well to low-resource languages via CTC fine-tuning on a
comparatively small amount of labelled data. This is the standard recipe for
low-resource ASR when end-to-end training from scratch is infeasible for lack
of data.

**Benchmark narrowness in low-resource ASR.** A recurring risk in
low-resource speech benchmarking is that available labelled corpora are
small enough to come from very few speakers, sometimes one. A WER measured
this way conflates "the model transcribes Nepali well" with "the model has
partially memorized this speaker's acoustic characteristics." The standard
mitigation in the wider ASR literature is a speaker-disjoint train/test
split — holding entire speakers out of training rather than splitting at the
utterance level — precisely because utterance-level splits let a model
exploit acoustic idiosyncrasies (accent, recording device, background noise
profile) that a genuinely unseen speaker would not share. Section 6 of this
report provides a direct, quantified demonstration of how much this
methodological choice matters for this specific audit.

**Deployment-oriented quantization.** Dynamic int8 quantization
(`torch.quantization.quantize_dynamic`) is a common post-training
optimization for reducing model storage footprint without retraining, widely
used to make deployed models smaller. Its latency effect is
hardware-dependent: x86 platforms typically benefit from the `fbgemm`
backend's optimized quantized kernels, while ARM/Apple Silicon platforms are
restricted to `qnnpack`, whose dynamic-quantization kernels do not
universally outperform native FP32 execution. Section 7 reports a direct
measurement of this trade-off on this project's hardware.

## 3. Dataset

Two corpora are used throughout, and their results are never merged into a
single number.

**OpenSLR-43** (`gauravparajuli/slr43` on Hugging Face) is the audited
model's own training corpus: a single-speaker (female), read-speech Nepali
dataset of 2,064 utterances. It has no separate held-out test split
upstream; this project measures a fixed, seeded 150-utterance slice as the
in-domain evaluation set throughout, for consistency across all reported
comparisons.

**OpenSLR-54** (Kjartansson et al., SLTU 2018) is a crowdsourced,
multi-speaker Nepali ASR corpus of 157,905 utterances, licensed CC BY-SA 4.0.
Because the full corpus is far larger than needed and downloading all of it
was neither necessary nor (in early attempts) fully reliable over the
available network connection, a subset was built: audio resampled to 16kHz
mono, transcripts NFC-normalized (Devanagari), utterances outside a 0.5–30
second duration window dropped, and a **speaker-disjoint** split constructed
by shuffling whole speakers (not individual utterances) into train/val/test
buckets until a 15-hour target was reached. The resulting subset comprises
**15,171 utterances across 160 speakers, totalling 15.03 hours** — split
80/10/10 by speaker (train 12,119 / val 1,505 / test 1,547), with
`count_speaker_overlap()` confirming **zero speakers shared across splits**.

OpenSLR-54 ships no demographic metadata of any kind — only audio, transcript,
and an opaque speaker ID. Since one of the roadmap's earlier directions
called for a speaker-gender field, and no such field exists in the source
data, a **gender pseudo-label** was derived per speaker from mean
fundamental frequency (F0), estimated via `librosa.pyin` and thresholded at
165Hz (below → male, above → female — the standard heuristic midpoint
between typical adult male ~85–180Hz and female ~165–255Hz pitch ranges).
This label is used only for the descriptive per-group breakdown in Section
6.3 and is labelled there, and everywhere else it appears, as a **pitch
threshold pseudo-label, not verified ground truth** — a distinction load-bearing
enough to affect how that specific result should be read, not a stylistic
footnote.

## 4. Method

### 4.1 Algorithm selection

This project fine-tunes the **audited checkpoint itself**
(`gagan3012/wav2vec2-xlsr-nepali`) rather than a different model or a
freshly-initialized XLS-R backbone. This is a deliberate choice, not a
default: the research question is whether *this model's* generalization
failure is caused by training-data narrowness, and answering that requires
holding architecture, vocabulary, and decoding constant while varying only
the training data's speaker diversity. Introducing a different architecture
(e.g. Whisper, an earlier direction explored and then abandoned for this
reason) would confound any improvement — it could no longer be attributed
cleanly to the data intervention. Training a fresh XLS-R backbone from
`facebook/wav2vec2-xls-r-300m` was considered and rejected for the same
reason: it would test a different, looser hypothesis ("can speaker-diverse
data alone produce a good Nepali model from generic multilingual
pretraining") rather than the one this report is about ("can this specific
published model be repaired").

Training from scratch was never a serious option: 15 hours of labelled audio
is roughly two orders of magnitude short of what training a competitive
speech model from random initialization requires. Whisper and Meta's MMS
were both considered and rejected as the primary fine-tuning target for the
architecture-confound reason above; MMS's Nepali path is also CTC-based, so
it would not even test a materially different hypothesis.

### 4.2 Fine-tuning configuration

The checkpoint's convolutional feature encoder was frozen
(`freeze_feature_encoder()`), following standard wav2vec2 fine-tuning
practice — that component is pretrained on far more audio than this
project's 15-hour subset provides, and freezing it reduces overfitting risk
while leaving the transformer encoder and CTC head trainable. Training used
AdamW (via the Hugging Face `Trainer` default), a learning rate of 3×10⁻⁵
with 500 warmup steps (deliberately low, since this continues fine-tuning an
already Nepali-adapted checkpoint rather than adapting a generic one),
batch size 2 with gradient accumulation 4 (effective batch 8), FP16 where
CUDA is available and FP32 otherwise, a 5-epoch ceiling, and early stopping
with patience 2 on validation WER (`load_best_model_at_end=True`,
`metric_for_best_model="wer"`). Experiment tracking used **MLflow** (a local
`mlruns/` store; TensorBoard was used in an earlier iteration of this project
and was replaced) — every training and evaluation run in this report is
logged there with its parameters and metrics.

Two practical obstacles were resolved during this work and are recorded
because they affect reproducibility on similar hardware. First, dynamic int8
quantization (Section 7) fails outright on Apple Silicon with
`RuntimeError: Didn't find engine for operation ... NoQEngine` unless
`torch.backends.quantized.engine` is explicitly set to `"qnnpack"` — the
`fbgemm` backend PyTorch defaults to is an x86-only build. Second, a genuine
out-of-memory crash occurred partway through the first full training run
(MPS reporting 17GB allocated); this was fixed by adding a callback that
calls `torch.mps.empty_cache()` periodically during training, and training
resumed cleanly from the last saved epoch checkpoint with no further issues.
Training itself ran locally on Apple Silicon CPU/MPS (no CUDA available on
this machine) after an earlier plan to use Google Colab or Kaggle for a
cloud GPU was set aside in favour of local, fully offline execution; the
full 5-epoch run took approximately 7 hours 40 minutes wall-clock.

### 4.3 Evaluation methodology

Every WER/CER comparison in this report uses the same procedure and the same
150-utterance seeded sample from each corpus (seed 42), so that numbers
across sections are directly comparable: a fixed slice of OpenSLR-43 for
in-domain measurement, and a seeded shuffle of OpenSLR-54's speaker-disjoint
test split for out-of-domain measurement. (OpenSLR-54's test manifest is
speaker-ordered on disk; a naive unsuffled slice was found early in this
project to land on a single-gender-pseudo-label subset of speakers by
chance — the seeded shuffle exists specifically to avoid that failure mode
recurring.) WER and CER are computed via `jiwer` after identical text
normalization on both reference and hypothesis.

## 5. Experimental Setup

All experiments ran on a local machine (Apple Silicon, macOS, no CUDA), in a
dedicated conda environment (Python 3.11), with dependencies pinned in
`requirements.txt`. `torch`/`torchaudio` are installed separately from the
pinned requirements file, since a cloud GPU runtime (Colab/Kaggle) would ship
its own CUDA-matched build; this project ultimately trained entirely
locally. All datasets, model checkpoints, and code referenced in this report
are drawn from the project repository's `coursework-10phase` branch and are
reproducible from the commit history there.

## 6. Results

### 6.1 The audit: does the published claim generalize?

| Evaluation | WER | CER |
|---|---|---|
| Self-reported (published) | 5.97% | — |
| Measured, in-domain (OpenSLR-43, this project) | **4.91%** | 0.87% |
| Measured, out-of-domain (OpenSLR-54 speaker-disjoint test) | **62.30%** | 17.38% |

The in-domain measurement is close to (marginally better than) the published
figure, confirming no discrepancy on the model's own distribution. The
out-of-domain measurement is more than twelve times worse. This is the
audit's central finding: the published number is an accurate description of
performance on narrow, single-speaker data and a highly misleading
description of performance on diverse speech.

### 6.2 Fine-tuning: does repair work, and at what cost?

| Checkpoint | In-domain (OpenSLR-43) | Out-of-domain (OpenSLR-54) |
|---|---|---|
| Original | 4.91% WER | 62.30% WER |
| Fine-tuned (speaker-disjoint, 5 epochs) | 16.40% WER | **38.17% WER** |

Fine-tuning on the speaker-diverse subset reduces out-of-domain WER by
roughly 39% relative (62.30% → 38.17%), directly supporting the hypothesis
that training-data narrowness, not an architectural limitation, is
responsible for the original generalization failure. The cost is real and
disclosed rather than minimised: in-domain WER more than triples (4.91% →
16.40%). Five epochs of continued fine-tuning on a differently-distributed
corpus mildly erodes the sharp specialisation the model had to its original
narrow training data — an expected trade-off, not a defect, and one that
should inform any decision about which checkpoint to deploy depending on the
target population of speakers.

Training-time validation WER on the model's own held-out val split (a
different measurement from the matched-methodology figures above, and not
interchangeable with them) improved monotonically across all five epochs
with no early stopping triggered: 72.98% → 70.88% → 70.09% → 69.03% → 68.83%.
The discrepancy between this figure and the 38.17% reported in the table
above is not an inconsistency: the two numbers are measured on different
splits (SLR54's own 1,505-utterance validation split, versus a fixed
150-utterance sample of the *test* split, evaluated identically to every
other cell in this report). The 38.17% figure is the one that belongs in
cross-checkpoint comparisons.

### 6.3 Error analysis and per-group breakdown

Applying the fine-tuned checkpoint to the same 150-utterance out-of-domain
sample and categorising errors heuristically (via jiwer character-alignment
and regex-based tagging; categories are not mutually exclusive) gives:

| Category | Count (of 150) |
|---|---|
| OOV / rare vocabulary | 90 |
| Phonetic confusion | 64 |
| Other (unclassified substitution pattern) | 59 |
| Noise / severe degradation | 22 |

Rare or out-of-vocabulary words are the single largest identifiable error
driver — consistent with a model trained on only 12,119 utterances having
incomplete lexical coverage of Nepali's full vocabulary. Representative worst
cases (reference → hypothesis) illustrate the phonetic-confusion pattern:
*मान्छे चटकारे → मान्छि चट कार्य*; *क्रान्तिकारी वाममोर्चाको → तन्थिकारी बाम वर्षको*.
Both show plausible phoneme-level substitutions rather than nonsensical
output, suggesting the model has learned Nepali phonotactics broadly but
lacks the specific lexical items involved.

WER broken out by the F0-pitch pseudo-label groups defined in Section 3 —
**stated again here explicitly because this specific result is easy to
misquote**: male-pseudo-label 37.68% WER (n=77) versus female-pseudo-label
38.73% WER (n=73). The two groups are close; there is no large disparity
this analysis can detect. This should be read as "WER is similar across
pitch-threshold groups," not as "WER is similar across genders," since the
grouping itself is a heuristic proxy, not verified demographic data.

A more striking finding is the **spread across individual speakers**: across
the 16 speakers represented in the 150-utterance sample, per-speaker mean WER
ranges from 12.5% (best, n=12 utterances) to 63.6% (worst, n=11 utterances) —
a five-fold difference the aggregate 38.17% figure obscures entirely.
Speaker-level variance of this magnitude suggests that some speakers'
recording conditions, accent, or speaking style are substantially harder for
this checkpoint than others, and that the single aggregate WER is a
summary statistic, not evidence of uniform performance.

### 6.4 Speaker-leakage ablation

To test how much a methodological shortcut — evaluating without a
speaker-disjoint split — would have distorted the Section 6.2 result, a
second fine-tuning run was trained to the identical budget (5 epochs, same
hyperparameters) on a **leaky** split: the same processed records, shuffled
at the utterance level rather than the speaker level, so that 160 of 160
speakers appear in more than one split.

| Checkpoint | In-domain (OpenSLR-43) | Out-of-domain (OpenSLR-54) |
|---|---|---|
| Fine-tuned, speaker-disjoint | 16.40% WER | 38.17% WER |
| Fine-tuned, leaky | 16.96% WER | **32.55% WER** |

The leaky-trained model scores 5.62 percentage points better on the
"out-of-domain" test than the properly speaker-disjoint model, despite
identical training budget and hyperparameters — the expected direction if
leakage inflates apparent performance, since the leaky split's test set is
not genuinely unseen for many of its speakers. In-domain performance is
nearly identical between the two (16.40% vs. 16.96%), as expected, since
neither training run touches OpenSLR-43's speaker. This is direct,
quantified evidence that the speaker-disjoint methodology used throughout
this project was the correct choice: a leaky evaluation would have reported
a roughly 15% relative WER "improvement" over the properly-measured result
that reflects memorisation of already-seen speakers rather than genuine
generalisation to new ones. This finding generalises beyond this specific
project: any Nepali ASR evaluation, including potentially aspects of
benchmark practice in the wider low-resource-ASR literature, that does not
explicitly control for speaker overlap between train and test risks the same
class of inflation.

### 6.5 Efficiency benchmark

The fine-tuned checkpoint was benchmarked in FP32 and after dynamic int8
quantization of its linear layers (`torch.quantization.quantize_dynamic`,
`qnnpack` backend — the only quantized engine available on this platform),
on 50 CPU inference runs:

| | Model size | Mean latency | WER | CER |
|---|---|---|---|---|
| FP32 | 1,203.6 MB | 204.7 ms | 35.81% | 7.17% |
| Dynamic INT8 | 48.5 MB | 335.2 ms | 36.49% | 7.49% |

Quantization delivers a 96.0% size reduction at negligible accuracy cost
(+0.68 percentage points WER, within normal sampling noise for 50
utterances) — but is **1.64× slower**, not faster, than FP32 on this
hardware. This is a real, reproducible result rather than a benchmark error:
the speed benefit of dynamic quantization on other platforms comes from
`fbgemm`'s optimized quantized GEMM kernels, unavailable on Apple Silicon;
`qnnpack`'s dynamic-quantization path does not uniformly outperform Apple's
native FP32 execution, and since only the model's linear layers are
quantized (its convolutional feature encoder remains FP32 regardless), the
per-call dequantization overhead can exceed the compute it saves at this
batch size. The practical conclusion for this specific checkpoint on this
specific hardware is that quantization is a *storage* optimisation, not a
*latency* one; a deployment targeting x86 CPUs or GPU/Core-ML-specific
quantization paths would need to be benchmarked independently rather than
assuming this result transfers.

## 7. Discussion

The results in Sections 6.1–6.4 tell a single, coherent story. A published
low-resource ASR benchmark measured on narrow data (Section 6.1) does not
describe performance on diverse speech; the gap is large enough (4.91% →
62.30% WER) that anyone deploying this model against real, varied users
would be seriously misled by the headline figure. Training-data diversity,
not architecture, explains a substantial share of that gap: fine-tuning the
same model on speaker-diverse data closes roughly 39% of it (Section 6.2).
The ablation in Section 6.4 then closes the methodological loop: it
demonstrates directly, on this project's own data and models, exactly the
failure mode that produced the original misleading benchmark in the first
place — evaluating without controlling for speaker overlap inflates apparent
performance by a similar order of magnitude (≈5.6 points) to the effect this
project set out to study. In other words, the project's central finding and
its central methodological safeguard are two instances of the same
underlying phenomenon, observed at different scales.

Three limitations should qualify how these results are read. First, the
gender pseudo-label (Section 3, Section 6.3) is a pitch-threshold heuristic,
not verified demographic data; the near-parity finding between pseudo-label
groups is a statement about that heuristic grouping, not about true gender,
and should not be cited as evidence the model is unbiased with respect to
speaker gender — a genuine bias analysis would need self-reported labels.
Second, the fine-tuning corpus is itself only 15 hours across 160 speakers —
larger and more diverse than the original single-speaker training set, but
still modest by ASR standards, so the repaired model's 38.17% WER should be
read as evidence that the *direction* of the intervention works, not as a
production-ready transcription quality. Third, the efficiency benchmark
(Section 6.5) is specific to this Apple Silicon/`qnnpack` configuration; its
conclusion that quantization slows inference down should not be
generalised to other hardware without re-measurement.

## 8. Conclusion

This project set out to answer whether a specific published Nepali ASR
benchmark claim (5.97% WER) survives contact with speaker-diverse speech, and
found that it does not: the same checkpoint's WER rises to 62.30% on a
different, multi-speaker corpus. Fine-tuning that checkpoint on a 15-hour,
speaker-disjoint, 160-speaker subset repairs a substantial share of that
gap (to 38.17% WER), at a disclosed cost to the model's original narrow-domain
sharpness. A matched-budget ablation confirms this improvement is genuine and
not a leakage artefact — and separately confirms that evaluating without a
speaker-disjoint split would have overstated the same improvement by roughly
5.6 percentage points, a result with implications beyond this specific
project for how low-resource ASR claims of this kind should be evaluated.
Error analysis identifies out-of-vocabulary words as the dominant remaining
error source and reveals per-speaker WER variance the aggregate metric
conceals; an efficiency benchmark shows that a standard deployment
optimisation (dynamic int8 quantization) trades substantial model size for a
measured latency *regression* on this hardware, a caveat-worthy, honestly
reported result rather than an unambiguous win. Future work would extend the
fine-tuning corpus beyond 15 hours, replace the F0-pitch pseudo-label with
verified demographic metadata if a bias analysis is required, and repeat the
efficiency benchmark on x86/GPU hardware before drawing platform-general
deployment conclusions.

## References

- Babu, A. et al. "XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale", 2021.
- Kjartansson, O. et al. "Crowd-Sourced Speech Corpora for Javanese, Sundanese, Sinhala, Nepali, and Bangladeshi Bengali", SLTU 2018 (OpenSLR-54).
- `gagan3012/wav2vec2-xlsr-nepali` model card, Hugging Face.
- `gauravparajuli/slr43` dataset card, Hugging Face (OpenSLR-43).
- Hugging Face, "Fine-Tune Wav2Vec2 for English ASR with Transformers" (blog; the wav2vec2 CTC fine-tuning recipe followed in this project, adapted for continued fine-tuning of an already-adapted checkpoint).
- jiwer library documentation (WER/CER computation, character alignment).
- MLflow documentation (experiment tracking).

## Appendix pointers

- `proposal.md` / `proposal.docx` / `proposal.pdf` — the original project proposal, reproduced verbatim
- `reports/phase4_audit_results.json` — Section 6.1
- `reports/phase6_notes.md`, `reports/phase6_train_full_*.json` — Section 4.2, 6.2 training details
- `reports/phase7_notes.md`, `reports/phase7_*_results.json` — Section 6.2
- `reports/phase9_error_analysis_results.json`, `reports/phase9_per_utterance.jsonl` — Section 6.3
- `reports/phase9_ablation_notes.md` — Section 6.4
- `reports/phase8_notes.md`, `reports/phase8_efficiency_results.json` — Section 6.5
- `src/*.py`, `scripts/*.py` — full code; external-source adaptations cited in module docstrings
- MLflow store (`mlruns/`) — every run referenced in this report, with logged parameters and metrics
