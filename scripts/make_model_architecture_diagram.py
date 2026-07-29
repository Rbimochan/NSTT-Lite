"""Generates the report's Figure 1: XLS-R (wav2vec2) model architecture,
showing the actual internal layers (CNN feature encoder, transformer
encoder, CTC head) and which components are frozen vs. fine-tuned in
Task 2 -- as opposed to a generic process-flow diagram.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(8.5, 11))
ax.set_xlim(0, 10)
ax.set_ylim(0, 15.5)
ax.axis("off")

FROZEN = "#D6D6D6"
FROZEN_EDGE = "#7A7A7A"
TUNED = "#FBD9A5"
TUNED_EDGE = "#B9770E"
NEUTRAL = "#EAF2FB"
NEUTRAL_EDGE = "#2F5496"

box_style = dict(boxstyle="round,pad=0.25,rounding_size=0.12", linewidth=1.5)

def box(x, y, w, h, text, fc, ec, fontsize=9, weight="normal", style="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=fc, edgecolor=ec, **box_style))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            weight=weight, style=style, wrap=True)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                  linewidth=1.4, color="#333333"))

# --- Title ---
ax.text(5, 15.1, "XLS-R (wav2vec2) Architecture — Frozen vs. Fine-tuned Components",
        ha="center", va="center", fontsize=11, weight="bold")

# --- 1. Raw waveform input ---
wf_y = 13.7
ax.plot([1.3 + 0.02*i for i in range(300)],
        [wf_y + 0.18*__import__("math").sin(i*0.35)*__import__("math").exp(-((i-150)**2)/12000) for i in range(300)],
        color="#555555", linewidth=0.8)
box(1.0, 13.35, 7.0, 0.7, "", "#FFFFFF", "#999999")
ax.text(8.3, 13.7, "raw audio\nwaveform", ha="left", va="center", fontsize=8, style="italic")

arrow(4.5, 13.35, 4.5, 12.9)

# --- 2. CNN Feature Encoder (frozen) ---
box(1.0, 11.7, 7.0, 1.2,
    "CNN Feature Encoder\n7 temporal convolution blocks (stride/kernel schedule per wav2vec2)\nraw waveform → latent speech representations z (50 Hz frame rate)",
    FROZEN, FROZEN_EDGE, fontsize=8.7)
ax.text(8.3, 12.3, "FROZEN\n(pretrained,\nnot updated)", ha="left", va="center", fontsize=7.8,
        color=FROZEN_EDGE, weight="bold")

arrow(4.5, 11.7, 4.5, 11.25)

# --- 3. Positional conv + Transformer Encoder (fine-tuned) ---
box(1.0, 8.7, 7.0, 2.4,
    "Transformer Encoder (24 layers, 1024-dim, 16 heads — XLS-R 0.3B config)\n"
    "relative positional convolutional embedding\n"
    "multi-head self-attention + feed-forward, per layer\n"
    "latent representations z → contextualised representations c",
    TUNED, TUNED_EDGE, fontsize=8.7)
ax.text(8.3, 9.9, "FINE-TUNED\n(weights updated\nin Task 2)", ha="left", va="center", fontsize=7.8,
        color=TUNED_EDGE, weight="bold")

arrow(4.5, 8.7, 4.5, 8.25)

# --- 4. Linear projection + CTC head (fine-tuned) ---
box(1.0, 7.05, 7.0, 1.1,
    "Linear Projection + CTC Head\ncontextualised representations c → per-frame vocabulary logits\n(Nepali character/token vocabulary, unchanged from audited checkpoint)",
    TUNED, TUNED_EDGE, fontsize=8.7)
ax.text(8.3, 7.6, "FINE-TUNED", ha="left", va="center", fontsize=7.8, color=TUNED_EDGE, weight="bold")

arrow(4.5, 7.05, 4.5, 6.6)

# --- 5. CTC decoding ---
box(1.0, 5.6, 7.0, 0.9, "Greedy CTC Decoding\nper-frame logits → collapsed character sequence", NEUTRAL, NEUTRAL_EDGE, fontsize=8.8)

arrow(4.5, 5.6, 4.5, 5.15)

box(1.0, 4.55, 7.0, 0.9, "Output transcript (Devanagari text)", NEUTRAL, NEUTRAL_EDGE, fontsize=9, weight="bold")

# --- Training signal (backprop arrow, dashed, only through tuned blocks) ---
ax.annotate("", xy=(2.3, 7.05), xytext=(2.3, 9.9),
            arrowprops=dict(arrowstyle="-|>", linestyle="dashed", color="#B9770E", linewidth=1.3,
                             connectionstyle="arc3,rad=0.35"))
ax.text(0.55, 8.4, "CTC loss\nbackprop\n(Task 2 only)", ha="center", va="center", fontsize=7.3,
        color="#B9770E", rotation=90)

# --- Legend ---
legend_elems = [
    Rectangle((0, 0), 1, 1, facecolor=FROZEN, edgecolor=FROZEN_EDGE, label="Frozen (pretrained, unchanged in both Task 1 and Task 2)"),
    Rectangle((0, 0), 1, 1, facecolor=TUNED, edgecolor=TUNED_EDGE, label="Fine-tuned in Task 2 (weights updated via CTC loss)"),
    Rectangle((0, 0), 1, 1, facecolor=NEUTRAL, edgecolor=NEUTRAL_EDGE, label="Inference-only step (no learnable weights)"),
]
ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.045),
          fontsize=7.8, frameon=False, ncol=1)

# --- Note box: Task 1 vs Task 2 ---
ax.text(5, 3.55,
    "Task 1 (Benchmark Audit) runs this exact architecture end-to-end with every block frozen (zero-shot).\n"
    "Task 2 (Fine-tuning Repair) keeps the CNN feature encoder frozen and updates only the transformer\n"
    "encoder and CTC head via continued CTC training on the speaker-disjoint OpenSLR-54 subset.",
    ha="center", va="top", fontsize=8.3, style="italic",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F7F7F7", edgecolor="#AAAAAA"))

plt.tight_layout()
plt.savefig("appendix_screenshots/figure1_architecture.png", dpi=160, bbox_inches="tight")
print("saved")
