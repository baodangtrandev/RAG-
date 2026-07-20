import matplotlib.pyplot as plt
import numpy as np

# Set plot style for academic paper
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

# 1. Pareto Frontier Plot: Latency vs Correctness
fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

configs = [
    ("Vector Baseline", 0.60, 19.60, "red", "o"),
    ("BM25 Baseline", 0.97, 29.20, "orange", "s"),
    ("Hybrid Baseline", 1.15, 33.60, "purple", "^"),
    ("T-RAG v1 Balanced G1", 1.02, 33.80, "brown", "D"),
    ("T-RAG v2 Standard", 0.84, 34.67, "blue", "P"),
    ("OPT: Low Latency", 0.76, 35.27, "cyan", "h"),
    ("Targeted H (Speed King v2)", 0.84, 34.60, "teal", "*"),
    ("OPT: Recall+Sparse (Best)", 1.02, 36.80, "green", "X"),
    ("Grid Tau = 0.10", 0.88, 36.67, "magenta", "v"),
    ("Targeted D (Alpha Sweet Spot)", 0.96, 36.00, "olive", ">")
]

for label, lat, corr, color, marker in configs:
    ax.scatter(lat, corr, color=color, marker=marker, s=120, label=label, edgecolors='black', alpha=0.9)

# Draw Pareto Frontier curve line
# Sorted by latency: Vector (0.60, 19.60) -> OPT Low Lat (0.76, 35.27) -> T-RAG v2 Std (0.84, 34.67) [not on frontier] 
# -> Grid Tau 0.10 (0.88, 36.67) -> OPT Recall+Sparse (1.02, 36.80)
frontier_x = [0.60, 0.76, 0.88, 1.02]
frontier_y = [19.60, 35.27, 36.67, 36.80]
ax.plot(frontier_x, frontier_y, "g--", alpha=0.5, label="Pareto Frontier")

ax.set_xlabel("Total Latency (seconds)")
ax.set_ylabel("Correctness Score (%)")
ax.set_title("Latency vs. Correctness Trade-off (Pareto Frontier)")
ax.set_xlim(0.5, 1.35)
ax.set_ylim(15, 40)

# Put legend on the bottom right or outside
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, facecolor='white', edgecolor='black')
plt.tight_layout()
plt.savefig("fig_pareto.png", bbox_inches='tight')
plt.close()

# 2. Dense vs. Sparse Weight Curve
fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=300)

dense_weights = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
correctness = np.array([36.00, 36.60, 35.80, 34.67, 28.26, 25.85, 24.25])

ax.plot(dense_weights, correctness, marker='o', color='royalblue', linewidth=2.5, markersize=8, label="Correctness vs. Dense Weight")
ax.fill_between(dense_weights, correctness, alpha=0.15, color='royalblue')

ax.set_xlabel("Dense Vector Weight ($W_{dense}$)")
ax.set_ylabel("Correctness Score (%)")
ax.set_title("Impact of Dense-Sparse Weight Ratios\n(BM25 Weight $W_{sparse} = 1.0 - W_{dense}$)")
ax.set_xticks(dense_weights)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(20, 40)

for x, y in zip(dense_weights, correctness):
    ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("fig_dense_sparse.png")
plt.close()

print("Plots generated successfully!")
