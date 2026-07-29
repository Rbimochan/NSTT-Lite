"""Generates the report's Figure 1 (system/experimental pipeline diagram)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(9, 6.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

box_style = dict(boxstyle="round,pad=0.3,rounding_size=0.15", linewidth=1.4)

def box(x, y, w, h, text, fc="#EAF2FB", ec="#2F5496", fontsize=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=fc, edgecolor=ec, **box_style))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             weight=weight, wrap=True)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                  linewidth=1.3, color="#333333"))

# Audited checkpoint
box(3.5, 8.8, 3.0, 0.9, "Audited checkpoint\ngagan3012/wav2vec2-xlsr-nepali", fc="#FDEBD0", ec="#B9770E", weight="bold")

# Task 1: audit
box(0.4, 6.8, 4.2, 1.3, "Task 1 -- Benchmark Audit\nzero-shot on OpenSLR-43 (in-domain)\nand OpenSLR-54 (out-of-domain)", fc="#EAF2FB")
box(5.4, 6.8, 4.2, 1.3, "Task 2 -- Fine-tuning\nCTC fine-tune on 15hr, 160-speaker\nspeaker-disjoint OpenSLR-54 subset", fc="#EAF2FB")

arrow(5.0, 8.8, 2.5, 8.1)
arrow(5.0, 8.8, 7.5, 8.1)

# Generalization re-evaluation
box(0.4, 4.8, 4.2, 1.2, "Generalization re-evaluation\nfour-cell: {original, fine-tuned} x\n{in-domain, out-of-domain}", fc="#EAF2FB")
box(5.4, 4.8, 4.2, 1.2, "Speaker-leakage ablation\nmatched-budget: speaker-disjoint\nvs. leaky (utterance-random) split", fc="#EAF2FB")

arrow(2.5, 6.8, 2.5, 6.0)
arrow(7.5, 6.8, 7.5, 6.0)
arrow(4.6, 5.4, 5.4, 5.4)

# Error analysis
box(2.9, 2.9, 4.2, 1.2, "Error analysis\nOOV/phonetic/noise categories,\nper-speaker + gender-pseudo-label WER", fc="#EAF2FB")
arrow(2.5, 4.8, 4.0, 4.1)
arrow(7.5, 4.8, 6.0, 4.1)

# Efficiency benchmark
box(2.9, 1.0, 4.2, 1.2, "Efficiency benchmark\nFP32 vs. dynamic int8 quantization\n(size / latency / WER trade-off)", fc="#EAF2FB")
arrow(5.0, 2.9, 5.0, 2.2)

plt.tight_layout()
plt.savefig("appendix_screenshots/figure1_pipeline.png", dpi=150, bbox_inches="tight")
print("saved")
