#!/usr/bin/env python3
"""
Create model comparison visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Load results
with open('artifacts/baseline_metrics.json', 'r') as f:
    baseline_metrics = json.load(f)

with open('artifacts/tgnn_metrics.json', 'r') as f:
    tgnn_metrics = json.load(f)

with open('artifacts/diffusion_prediction_results.json', 'r') as f:
    diffusion_metrics = json.load(f)

# Create comparison visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Classification Metrics Comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
baseline_values = [
    baseline_metrics['accuracy'],
    baseline_metrics['precision'], 
    baseline_metrics['recall'],
    baseline_metrics['f1_score'],
    baseline_metrics['auc_roc']
]
tgnn_values = [
    tgnn_metrics['accuracy'],
    tgnn_metrics['precision'],
    tgnn_metrics['recall'], 
    tgnn_metrics['f1'],
    tgnn_metrics['auc']
]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, baseline_values, width, label='TF-IDF + LR (Baseline)', alpha=0.8, color='skyblue')
bars2 = ax1.bar(x + width/2, tgnn_values, width, label='TGNN', alpha=0.8, color='lightcoral')

ax1.set_xlabel('Metrics')
ax1.set_ylabel('Score')
ax1.set_title('Classification Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, rotation=45)
ax1.legend()
ax1.set_ylim(0.9, 1.0)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 2. Performance Improvement
improvements = [tgnn_values[i] - baseline_values[i] for i in range(len(metrics))]
colors = ['green' if imp > 0 else 'red' for imp in improvements]

bars3 = ax2.bar(metrics, improvements, color=colors, alpha=0.7)
ax2.set_xlabel('Metrics')
ax2.set_ylabel('Improvement')
ax2.set_title('TGNN Performance Improvement over Baseline')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, rotation=45)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Add value labels
for bar, imp in zip(bars3, improvements):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height > 0 else -0.002),
             f'{imp:+.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

# 3. Diffusion Prediction Metrics (TGNN only)
diffusion_metrics_names = ['Hit@1', 'Hit@5', 'Hit@10', 'MRR', 'Jaccard']
diffusion_values = [
    diffusion_metrics['hit_at_k']['1'],
    diffusion_metrics['hit_at_k']['5'], 
    diffusion_metrics['hit_at_k']['10'],
    diffusion_metrics['mrr'],
    diffusion_metrics['jaccard']
]

bars4 = ax3.bar(diffusion_metrics_names, diffusion_values, alpha=0.8, color='lightgreen')
ax3.set_xlabel('Metrics')
ax3.set_ylabel('Score')
ax3.set_title('TGNN Diffusion Prediction Performance')
ax3.set_ylim(0, 1.0)

# Add value labels
for bar, val in zip(bars4, diffusion_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# 4. Model Capabilities Comparison
capabilities = ['Text Classification', 'Graph Structure', 'Temporal Dynamics', 'Diffusion Prediction', 'Interpretability', 'Speed']
baseline_cap = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]  # Baseline capabilities
tgnn_cap = [1.0, 1.0, 1.0, 1.0, 0.3, 0.4]  # TGNN capabilities

x_cap = np.arange(len(capabilities))
bars5 = ax4.bar(x_cap - width/2, baseline_cap, width, label='TF-IDF + LR', alpha=0.8, color='skyblue')
bars6 = ax4.bar(x_cap + width/2, tgnn_cap, width, label='TGNN', alpha=0.8, color='lightcoral')

ax4.set_xlabel('Capabilities')
ax4.set_ylabel('Score (0-1)')
ax4.set_title('Model Capabilities Comparison')
ax4.set_xticks(x_cap)
ax4.set_xticklabels(capabilities, rotation=45)
ax4.legend()
ax4.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Model comparison visualization saved to figures/model_comparison.png")
