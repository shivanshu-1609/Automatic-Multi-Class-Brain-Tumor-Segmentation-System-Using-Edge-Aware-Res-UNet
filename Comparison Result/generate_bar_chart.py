import matplotlib.pyplot as plt
import numpy as np
import os

# Data
metrics = ['Dice Score', 'IoU', 'Sensitivity']
simple_unet = [0.7355, 0.6455, 0.7563]
ea_unet = [0.7181, 0.6187, 0.7935]

x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, simple_unet, width, label='Model 1: Simple UNet', color='#4a90e2')
rects2 = ax.bar(x + width/2, ea_unet, width, label='Model 2: Ea+UNet', color='#e24a4a')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Metric Score (0 to 1)', fontsize=12)
ax.set_title('Overall Performance Comparison', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.0)

# Move legend below the graph
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)

# Add value labels on top of the bars
ax.bar_label(rects1, padding=3, fmt='%.4f', fontsize=10)
ax.bar_label(rects2, padding=3, fmt='%.4f', fontsize=10)

fig.tight_layout()

# Save the plot
out_path = r"d:\Major Project\Comparison Result\performance_bar_chart.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Graph successfully generated and saved at: {out_path}")
