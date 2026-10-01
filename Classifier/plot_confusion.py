"""Plot the confusion matrices of every results_<tag>.npy.

Reads every results_*.npy under CLASSIFIER.results_dir and writes a single PNG grid: rows = tags, columns = backbones. 
Each cell shows raw counts plus row-normalised percentages.

Usage:
    python Classifier/plot_confusion.py
"""

import glob
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import CLASSIFIER


def main():
    results_dir = CLASSIFIER.results_dir
    files = sorted(glob.glob(os.path.join(results_dir, 'results_*.npy')))
    if not files:
        print(f"No results_*.npy found in {results_dir}")
        return

    runs = {Path(f).stem.replace('results_', ''):
            np.load(f, allow_pickle=True).tolist() for f in files}

    n_rows = len(runs)
    n_cols = max(len(results) for results in runs.values())
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 4.2 * n_rows),
                             squeeze=False)

    for i, (tag, results) in enumerate(runs.items()):
        for j, r in enumerate(results):
            cm = np.array(r['confusion'])
            cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
            ax = axes[i, j]
            ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)

            for ii in range(2):
                for jj in range(2):
                    color = 'white' if cm_pct[ii, jj] > 50 else 'black'
                    ax.text(jj, ii,
                            f"{cm[ii, jj]}\n({cm_pct[ii, jj]:.1f}%)",
                            ha='center', va='center',
                            color=color, fontsize=11)

            ax.set_xticks([0, 1], ['N', 'AFIB'])
            ax.set_yticks([0, 1], ['N', 'AFIB'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f"{tag} / {r['model']}\n"
                         f"acc={r['accuracy']:.3f}  "
                         f"f1={r['f1']:.3f}  "
                         f"auc={r['auc']:.3f}+/-{r['auc_std']:.3f}",
                         fontsize=10)
        for j in range(len(results), n_cols):
            axes[i, j].set_visible(False)

    plt.tight_layout()
    out = os.path.join(results_dir, 'confusion_grid.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
