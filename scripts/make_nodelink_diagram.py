"""Generates a classic node-link 'deep neural network' style diagram
(matching the standard textbook illustration: input layer -> multiple
hidden layers -> output layer, fully connected, color-coded per layer)
applied to this project's XLS-R architecture, for the appendix.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6.2)
ax.axis("off")
ax.set_facecolor("white")

INPUT_C = "#1A56E8"
HIDDEN_C = "#2FD9C9"
OUTPUT_C = "#8FA0F5"
LINE_C = "#8FA0F5"

ax.text(0.3, 5.85, "Deep Neural Network — XLS-R (wav2vec2) as a Layered Network",
        fontsize=16, weight="bold", color="#1A1A1A", ha="left")

layers = [
    {"x": 1.0, "n": 5, "color": INPUT_C, "label": "Input Layer", "sub": "audio waveform\nsamples"},
    {"x": 3.2, "n": 5, "color": HIDDEN_C, "label": "", "sub": ""},
    {"x": 5.0, "n": 5, "color": HIDDEN_C, "label": "Hidden Layers", "sub": "CNN feature encoder (frozen)\n+ 24 transformer blocks (fine-tuned)"},
    {"x": 6.8, "n": 5, "color": HIDDEN_C, "label": "", "sub": ""},
    {"x": 8.6, "n": 4, "color": OUTPUT_C, "label": "Output Layer", "sub": "CTC character\nprobabilities"},
]

y_center = 3.15
spacing = 0.75

positions = []
for layer in layers:
    n = layer["n"]
    ys = [y_center + (i - (n - 1) / 2) * spacing for i in range(n)]
    positions.append(ys)

# draw connections (fully connected, faded)
for li in range(len(layers) - 1):
    x1, x2 = layers[li]["x"], layers[li + 1]["x"]
    for y1 in positions[li]:
        for y2 in positions[li + 1]:
            ax.add_patch(FancyArrowPatch((x1 + 0.22, y1), (x2 - 0.22, y2),
                                          arrowstyle="-|>", mutation_scale=6,
                                          linewidth=0.5, color=LINE_C, alpha=0.35, zorder=1))

# draw nodes
for li, layer in enumerate(layers):
    for y in positions[li]:
        ax.add_patch(Circle((layer["x"], y), 0.22, facecolor=layer["color"], edgecolor="white", linewidth=1.5, zorder=2))

# labels
ax.text(layers[0]["x"], 5.15, "Input Layer", fontsize=12.5, weight="bold", ha="center", color="#1A1A1A")
ax.text(layers[0]["x"], 4.85, "audio waveform samples", fontsize=9, ha="center", color="#555555")

ax.text((layers[1]["x"] + layers[3]["x"]) / 2, 5.15, "Multiple Hidden Layers", fontsize=12.5, weight="bold", ha="center", color="#1A1A1A")
ax.text((layers[1]["x"] + layers[3]["x"]) / 2, 4.85, "CNN feature encoder (frozen) + 24 transformer\nencoder blocks (fine-tuned in Task 2)", fontsize=9, ha="center", color="#555555")

ax.text(layers[4]["x"], 5.15, "Output Layer", fontsize=12.5, weight="bold", ha="center", color="#1A1A1A")
ax.text(layers[4]["x"], 4.85, "CTC character\nprobabilities", fontsize=9, ha="center", color="#555555")

# bottom note
ax.text(5.5, 0.55,
        "Simplified node-link view of the layered network in Figure 1 (5 nodes per layer shown for clarity; actual widths: "
        "512-dim conv channels, 1024-dim transformer, and full Nepali character vocabulary).",
        fontsize=8.5, ha="center", style="italic", color="#666666")

plt.tight_layout()
plt.savefig("appendix_screenshots/figure_nodelink_architecture.png", dpi=160, bbox_inches="tight", facecolor="white")
print("saved")
