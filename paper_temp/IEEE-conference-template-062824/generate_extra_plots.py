import matplotlib.pyplot as plt
import numpy as np
import os

# Set output directory to current folder
out_dir = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# Plot 1: Performance by Query Category (Grouped Bar Chart)
# ==========================================
categories = [
    'Basic', 'Semantic', 'Intra-doc', 'Project', 'Constrained', 
    'Conflicting', 'Completeness', 'Misc.', 'High-level', 'Not Found'
]
hybrid_bl = [40.6, 14.4, 45.0, 17.5, 20.0, 50.0, 15.0, 60.0, 40.0, 100.0]
trag_optimal = [45.1, 16.8, 32.5, 12.5, 30.0, 45.0, 15.0, 80.0, 50.0, 100.0]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, hybrid_bl, width, label='Hybrid Baseline', color='#e74c3c')
rects2 = ax.bar(x + width/2, trag_optimal, width, label='T-RAG v2 (Optimal)', color='#2ecc71')

ax.set_ylabel('Correctness (%)', fontsize=12)
ax.set_title('Performance Comparison Across Query Categories', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=11)

# Add grid lines behind bars
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_ylim(0, 110)

fig.tight_layout()
fig_query_types_path = os.path.join(out_dir, 'fig_query_types.png')
plt.savefig(fig_query_types_path, dpi=300)
plt.close()
print(f"Saved {fig_query_types_path}")

# ==========================================
# Plot 2: Context Window (Top-K) Impact on Refusal Rate
# ==========================================
k_values = [1, 3, 5]
correctness = [22.40, 30.60, 34.67]
refused = [40.60, 21.60, 17.00]

fig, ax1 = plt.subplots(figsize=(7, 5))

color1 = '#2980b9'
ax1.set_xlabel('Context Window Size (Top-K)', fontsize=12)
ax1.set_ylabel('Correctness (%)', color=color1, fontsize=12)
line1 = ax1.plot(k_values, correctness, marker='o', markersize=8, color=color1, linewidth=2.5, label='Correctness')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(15, 45)
ax1.set_xticks(k_values)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color2 = '#c0392b'
ax2.set_ylabel('Refusal Rate (%)', color=color2, fontsize=12)
line2 = ax2.plot(k_values, refused, marker='s', markersize=8, color=color2, linewidth=2.5, linestyle='--', label='Refusal Rate')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(10, 45)

# Adding grid and title
ax1.grid(True, linestyle=':', alpha=0.6)
plt.title('Impact of Context Window (K) on Generation', fontsize=14, fontweight='bold')

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=11)

fig.tight_layout()
fig_context_k_path = os.path.join(out_dir, 'fig_context_k.png')
plt.savefig(fig_context_k_path, dpi=300)
plt.close()
print(f"Saved {fig_context_k_path}")
