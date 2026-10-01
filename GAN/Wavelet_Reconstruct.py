"""This script loads a trained GAN generator and reconstructs ECGs from the CWT scalogram tensors.

Usage:
    python GAN/Wavelet_Reconstruct.py
    python GAN/Wavelet_Reconstruct.py --n-samples 10 --out my.png

--data-dir must hold the arrays the generator was trained on.
"""

import argparse
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cwt_utils import scalogram_to_signal, reconstruction_constant, fs_from_scales
from config import GAN, WAVELET, SEED


def generate(n_samples=5, out_file=None, model_file=None, data_dir=None):
    """Save a figure of n_samples generated scalograms and their ICWT."""
    data_dir = data_dir or os.path.dirname(GAN.scale_file) or '.'
    seg_scales = np.load(os.path.join(data_dir, os.path.basename(GAN.scale_file)))
    scales = np.load(os.path.join(data_dir, os.path.basename(GAN.scales_file)))
    C_psi   = reconstruction_constant(WAVELET.fb, WAVELET.fc)
    fs = fs_from_scales(scales, WAVELET.f_min, WAVELET.fc)

    model_file = model_file or GAN.gen_model_file
    if out_file is None:
        out_file = f'synthetic_ecg_{Path(model_file).stem}.png'

    model = tf.keras.models.load_model(model_file)

    noise = tf.random.normal([n_samples, GAN.latent_dim])

    rng = np.random.default_rng(SEED)
    drawn = rng.choice(seg_scales, size=n_samples)
    scalograms = model.predict(noise, verbose=0) * drawn[:, None, None, None]

    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i, sc in enumerate(scalograms):
        ecg = scalogram_to_signal(sc, scales, C_psi)
        t   = np.arange(len(ecg)) / fs

        axes[i, 0].imshow(np.hypot(sc[:, :, 0], sc[:, :, 1]), aspect='auto', cmap='magma', origin='lower')
        axes[i, 0].set_title(f"Sample {i}: |W|")

        axes[i, 1].imshow(sc[:, :, 0], aspect='auto', cmap='RdBu_r', origin='lower')
        axes[i, 1].set_title(f"Sample {i}: Real")

        axes[i, 2].imshow(sc[:, :, 1], aspect='auto', cmap='RdBu_r', origin='lower')
        axes[i, 2].set_title(f"Sample {i}: Imag")

        axes[i, 3].plot(t, ecg, color='#CC0000', lw=1.2)
        axes[i, 3].set_title(f"Sample {i}: ECG ({len(ecg)} samples)")
        axes[i, 3].set_xlabel("Time (s)")
        axes[i, 3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(SEED)
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', default=GAN.gen_model_file,
                   help="generator .keras file")
    p.add_argument('--n-samples', type=int, default=5,
                   help="number of samples to plot")
    p.add_argument('--data-dir', default=None,
                   help="directory of the set the generator was trained on")
    p.add_argument('--out', default=None,
                   help="output PNG")
    args = p.parse_args()
    generate(n_samples=args.n_samples, out_file=args.out, model_file=args.model, data_dir=args.data_dir)
