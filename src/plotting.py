"""
Visualization functions for feature map comparison experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_accuracy_comparison(results_dict, save_dir='results'):
    """Bar chart comparing accuracy across feature maps and datasets."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    # results_dict: {dataset_name: {fm_name: accuracy}}
    ds_names = list(results_dict.keys())
    fm_names = list(next(iter(results_dict.values())).keys())

    x = np.arange(len(fm_names))
    width = 0.25
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    for i, ds in enumerate(ds_names):
        accs = [results_dict[ds].get(fm, 0.5) for fm in fm_names]
        ax.bar(x + i * width, accs, width, label=ds, color=colors[i % len(colors)],
               alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Test Accuracy')
    ax.set_title('Classification Accuracy by Feature Map')
    ax.set_xticks(x + width)
    ax.set_xticklabels(fm_names, rotation=15)
    ax.legend()
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path / 'accuracy_comparison.png', dpi=150)
    plt.close()


def plot_kernel_matrices(kernel_matrices_dict, save_dir='results'):
    """Heatmaps of quantum kernel matrices for each feature map."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    n = len(kernel_matrices_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for idx, (name, K) in enumerate(kernel_matrices_dict.items()):
        im = axes[idx].imshow(K, cmap='viridis', vmin=0, vmax=1)
        axes[idx].set_title(name)
        plt.colorbar(im, ax=axes[idx], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path / 'kernel_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_kernel_alignment_bars(alignment_scores, save_dir='results'):
    """Bar chart of kernel target alignment scores."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(alignment_scores.keys())
    values = [alignment_scores[n] for n in names]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    ax.bar(names, values, color=colors[:len(names)], alpha=0.85,
           edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Kernel Target Alignment')
    ax.set_title('KTA by Feature Map')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path / 'kernel_alignment.png', dpi=150)
    plt.close()


def plot_expressibility_histograms(expr_results, save_dir='results'):
    """Fidelity distribution histograms for each feature map."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    n = len(expr_results)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    for idx, (name, data) in enumerate(expr_results.items()):
        axes[idx].hist(data['fidelities'], bins=30, density=True, alpha=0.7,
                      color=colors[idx % len(colors)], edgecolor='black', linewidth=0.5)
        axes[idx].set_title(f"{name}\nKL = {data['kl_divergence']:.4f}")
        axes[idx].set_xlabel('Fidelity')
        axes[idx].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(save_path / 'expressibility.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_circuit_resources(feature_maps_dict, save_dir='results'):
    """Print circuit resource comparison table."""
    for name, fm in feature_maps_dict.items():
        decomposed = fm.decompose()
        ops = decomposed.count_ops()
        print(f"{name}: depth={decomposed.depth()}, "
              f"gates={sum(ops.values())}, cx={ops.get('cx', 0)}, "
              f"params={fm.num_parameters}")
