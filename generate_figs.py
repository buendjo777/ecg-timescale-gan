"""Regenerate the data-driven thesis figures.

    python generate_figs.py --out ./figures

Produces:

    fig_2_3.png   normal vs AF strips with R peaks and the R-R series
    fig_5_1.png   normal segment + its scalogram
    fig_5_2.png   AF segment + its scalogram
    fig_5_3.png   round-trip reconstruction

The ECG channel of each record is the one the classifier uses (AFDB_BEST).
Nothing here writes to the pipeline's artefacts; it only reads the databases.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pywt
import wfdb

from config import WAVELET, CLASSIFIER, AFDB_BEST
from cwt_utils import (make_scales, signal_to_scalogram, scalogram_to_signal,
                       reconstruction_constant, bandpass_filter, select_lead,
                       fit_length, rhythm_index)

# Colours and fonts of the thesis figures.
NAVY = "#0A2A4E"
CORAL = "#F26A6A"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "text.color": "black", "axes.edgecolor": "black",
    "axes.labelcolor": "black", "axes.titlecolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "axes.grid": True, "grid.color": "#EFEFEF", "grid.linewidth": 0.8,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})


def title(ax, text, fs=11.5):
    ax.set_title(text, loc="left", color="black", fontsize=fs, pad=8)


def figure_title(fig, text, fs=13.0):
    fig.text(0.010, 0.998, text, ha="left", va="top", fontsize=fs,
             color="black")


# Data access

def load_afdb_window(record, start, seconds=10.0, path=None):
    """A single labelled window plus its R peaks."""
    path = path or CLASSIFIER.data_path
    fs = CLASSIFIER.fs
    n = int(seconds * fs)
    rec = wfdb.rdrecord(f"{path.rstrip('/')}/{record}",
                        sampfrom=start, sampto=start + n)
    sig = select_lead(rec, db="afdb", channel=AFDB_BEST[record])
    if WAVELET.apply_bandpass:
        sig = bandpass_filter(sig, fs, WAVELET.f_min, WAVELET.f_max)
    try:
        qrs = wfdb.rdann(f"{path.rstrip('/')}/{record}", "qrs",
                         sampfrom=start, sampto=start + n)
        peaks = np.asarray(qrs.sample) - start
        peaks = peaks[(peaks >= 0) & (peaks < n)]
    except Exception:
        peaks = np.array([], dtype=int)
    return fit_length(sig, n), peaks, fs


def rhythm_of(record, sample, path=None):
    """The .atr rhythm label covering a sample index, and where it starts."""
    path = path or CLASSIFIER.data_path
    ann = wfdb.rdann(f"{path.rstrip('/')}/{record}", "atr")
    r_samples, r_labels = rhythm_index(ann)
    entry = int(np.searchsorted(r_samples, sample, side="right"))
    if entry == 0:
        return None, None
    return r_labels[entry - 1], int(r_samples[entry - 1])


# Figures

def strip(ax, sig, fs, peaks, colour=NAVY):
    t = np.arange(len(sig)) / fs
    ax.plot(t, sig, color=colour, lw=1.1)
    if len(peaks):
        ax.plot(peaks / fs, sig[peaks] + 0.06 * np.ptp(sig), marker="v",
                linestyle="none", markersize=6, color="black", zorder=5)
    ax.set_xlim(0, t[-1])
    ax.set_ylabel("mV")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def rr_note(ax, peaks, fs):
    if len(peaks) < 2:
        return ""
    rr = np.diff(peaks) / fs
    txt = (f"R-R  mean {rr.mean():.3f} s   sd {rr.std():.3f} s   "
           f"min {rr.min():.2f} s   max {rr.max():.2f} s   ({len(rr)} intervals)")
    ax.text(0.0, -0.30, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.0, color="black")
    return txt


def scalogram_panel(ax, sig, fs, scales):
    coefs, _ = pywt.cwt(sig - sig.mean(), scales, WAVELET.wavelet)
    freqs = WAVELET.fc * fs / scales
    t = np.arange(len(sig)) / fs
    ax.pcolormesh(t, freqs, np.abs(coefs), cmap="magma", shading="auto")
    ax.set_yscale("log")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.grid(False)
    return coefs


def fig_2_3(out, norm, afib):
    (ns, npk, fs), (as_, apk, _) = norm, afib
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 5.4),
                             gridspec_kw=dict(hspace=0.62))
    title(axes[0], f"Normal rhythm - afdb record {NORM_REC}")
    strip(axes[0], ns, fs, npk)
    rr_note(axes[0], npk, fs)
    title(axes[1], f"Atrial fibrillation - afdb record {AFIB_REC}")
    strip(axes[1], as_, fs, apk)
    rr_note(axes[1], apk, fs)
    axes[1].set_xlabel("Time (s)")
    figure_title(fig, "Normal rhythm and atrial fibrillation compared")
    fig.subplots_adjust(top=0.86)
    fig.savefig(f"{out}/fig_2_3.png")
    plt.close(fig)


def fig_signal_and_scalogram(out, name, sig, fs, peaks, scales, heading,
                             fig_title):
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 6.0),
                             gridspec_kw=dict(height_ratios=[1.0, 1.55],
                                              hspace=0.42))
    title(axes[0], heading)
    strip(axes[0], sig, fs, peaks)
    title(axes[1], "Magnitude of the continuous wavelet transform")
    scalogram_panel(axes[1], sig, fs, scales)
    figure_title(fig, fig_title)
    fig.subplots_adjust(top=0.88)
    fig.savefig(f"{out}/{name}.png")
    plt.close(fig)


def fig_5_3(out, sig, fs, scales, heading):
    """Three panels: the signal, the magnitude of its scalogram, and the
    original with the reconstruction superimposed."""
    tensor, mean = signal_to_scalogram(sig, scales, WAVELET.wavelet)
    K = reconstruction_constant(WAVELET.fb, WAVELET.fc)
    rec = scalogram_to_signal(tensor, scales, K) + mean
    r = float(np.corrcoef(sig, rec)[0, 1])
    nrmse = float(np.sqrt(np.mean((sig - rec) ** 2)) / np.std(sig))

    W_in = np.abs(tensor[:, :, 0] + 1j * tensor[:, :, 1])
    vmax = float(W_in.max())

    t = np.arange(len(sig)) / fs
    freqs = WAVELET.fc * fs / scales
    fig, axes = plt.subplots(3, 1, figsize=(10.4, 7.4),
                             gridspec_kw=dict(
                                 height_ratios=[1.0, 1.45, 1.0],
                                 hspace=0.46))

    title(axes[0], heading)
    axes[0].plot(t, sig, color=NAVY, lw=1.1)
    axes[0].set_xlim(0, t[-1]); axes[0].set_ylabel("mV")
    for s in ("top", "right"):
        axes[0].spines[s].set_visible(False)

    title(axes[1], "|W(a, t)| of the segment")
    axes[1].pcolormesh(t, freqs, W_in, cmap="magma", shading="auto",
                       vmin=0, vmax=vmax)
    axes[1].set_yscale("log"); axes[1].set_ylabel("Frequency (Hz)")
    axes[1].grid(False)

    title(axes[2], "Original and reconstruction superimposed")
    axes[2].plot(t, sig, color="#9EC3D8", lw=2.2, label="original")
    axes[2].plot(t, rec, color=CORAL, lw=1.0, label="reconstructed")
    axes[2].legend(frameon=False, loc="upper right", fontsize=9.5, ncol=2)
    axes[2].set_xlim(0, t[-1]); axes[2].set_ylabel("mV")
    axes[2].set_xlabel("Time (s)")
    for s in ("top", "right"):
        axes[2].spines[s].set_visible(False)
    axes[2].text(0.0, -0.30,
                 f"Pearson r = {r:.4f}     normalised RMSE = {nrmse:.4f}",
                 transform=axes[2].transAxes, fontsize=10.0, color="black")

    figure_title(fig, "Round-trip fidelity of the wavelet transform")
    fig.subplots_adjust(top=0.94)
    fig.savefig(f"{out}/fig_5_3.png")
    plt.close(fig)
    return r, nrmse


NORM_REC, NORM_START = "07910", 5250000
AFIB_REC, AFIB_START = "04746", 2950000


def main():
    global NORM_REC, AFIB_REC
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", default="./figures",
                   help="output folder")
    p.add_argument("--norm", default=f"{NORM_REC}:{NORM_START}",
                   help="normal-rhythm window as RECORD:START_SAMPLE")
    p.add_argument("--afib", default=f"{AFIB_REC}:{AFIB_START}",
                   help="AFIB window as RECORD:START_SAMPLE")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    NORM_REC, ns = args.norm.split(":")[0], int(args.norm.split(":")[1])
    AFIB_REC, as_ = args.afib.split(":")[0], int(args.afib.split(":")[1])

    for rec, start in ((NORM_REC, ns), (AFIB_REC, as_)):
        lab, at = rhythm_of(rec, start)
        print(f"{rec} @{start}: enclosing .atr rhythm = {lab} (set at {at})")

    norm = load_afdb_window(NORM_REC, ns)
    afib = load_afdb_window(AFIB_REC, as_)
    scales = make_scales(CLASSIFIER.fs, WAVELET.f_min, WAVELET.f_max,
                         WAVELET.n_scales, WAVELET.fc)

    fig_2_3(args.out, norm, afib)
    fig_signal_and_scalogram(
        args.out, "fig_5_1", norm[0], norm[2], norm[1], scales,
        f"Ten seconds of normal rhythm - afdb record {NORM_REC}",
        "A normal-rhythm segment and its scalogram")
    fig_signal_and_scalogram(
        args.out, "fig_5_2", afib[0], afib[2], afib[1], scales,
        f"Ten seconds of atrial fibrillation - afdb record {AFIB_REC}",
        "An atrial fibrillation segment and its scalogram")
    r, nrmse = fig_5_3(
        args.out, afib[0], afib[2], scales,
        f"Round-trip reconstruction - afdb record {AFIB_REC}")
    print(f"fig_5_3 round trip: r={r:.4f}  nrmse={nrmse:.4f}")
    print(f"Done. Figures in {args.out}")


if __name__ == "__main__":
    main()
