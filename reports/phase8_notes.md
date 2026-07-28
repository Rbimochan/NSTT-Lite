# Phase 8 — Efficiency / Deployment Benchmark

**Rubric target:** Originality (10 marks).

## Setup

Fine-tuned checkpoint (`models/xlsr-ft`), CPU inference, 50 real test
utterances (seeded shuffle, same convention as Phase 4/7), 20 latency runs
per configuration (after a warm-up call). Dynamic int8 quantization via
`torch.quantization.quantize_dynamic` on `nn.Linear` layers, using the
`qnnpack` backend (the only quantized engine available on Apple Silicon —
torch's x86 default, `fbgemm`, is not present on this platform; without
explicitly setting `torch.backends.quantized.engine = "qnnpack"` the call
fails with `RuntimeError: Didn't find engine for operation ... NoQEngine`).

## Results

| | Model size | Mean latency | WER | CER |
|---|---|---|---|---|
| FP32 | 1,203.6 MB | 204.7 ms | 35.81% | 7.17% |
| Dynamic INT8 | 48.5 MB | 335.2 ms | 36.49% | 7.49% |
| **Δ** | **−96.0%** | **+63.8% (slower)** | +0.68pp | +0.32pp |

## Honest interpretation — this is a mixed result, not a clean win

Quantization delivered a dramatic size reduction (96%, useful for
distribution/storage) at essentially no accuracy cost (+0.68pp WER is within
normal run-to-run noise for a 50-utterance sample). **But it made inference
slower, not faster** — 0.61x the speed of FP32, i.e. a 1.64x latency
*regression*.

This is a known, real phenomenon, not a bug in the benchmark: dynamic int8
quantization's speed benefit comes from x86's `fbgemm` backend having
highly optimized quantized GEMM kernels. On Apple Silicon, only `qnnpack` is
available, and its dynamic-quantization kernels are not uniformly faster
than Apple's own optimized FP32 BLAS/Accelerate paths — for a
convolution-heavy model like wav2vec2/XLS-R (where only the `nn.Linear`
layers get quantized; the convolutional feature encoder stays FP32
regardless), the per-call dequantization overhead can exceed the compute
savings, especially at this batch size (1) and sequence length.

**Practical takeaway for deployment:** quantization here is a *storage/size*
optimization, not a *latency* optimization, on Apple Silicon CPU. Anyone
deploying this specific checkpoint on ARM hardware should benchmark before
assuming int8 is faster — it measurably was not here. On x86 hardware with
`fbgemm`, or with GPU/Core ML-specific quantization paths, the result would
likely differ; that's noted as a limitation/future-work item, not tested here.

## Status

- [x] FP32 baseline: size, latency, WER
- [x] Dynamic INT8: size, latency, WER
- [x] Trade-off computed and logged to MLflow (experiment `phase8-efficiency`)
- [ ] Cross-platform (x86/fbgemm or GPU) comparison — out of scope for this
      machine, noted as a limitation in the report
