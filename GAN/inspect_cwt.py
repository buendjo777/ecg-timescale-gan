"""Inspection and diagnostic tools for scalograms and generators.

Subcommands:
  verify     CWT -> ICWT round trip on one afdb record
  browse     one PNG per stored scalogram; with --source clf also the raw
             ECG with its R peaks and the ICWT
  roundtrip  whether generated scalograms are valid CWTs, and their rhythm
  collapse   diversity and memorisation check for a generator

Subcommands that draw scalograms take --view: abs, logabs, re or im.

Usage:
    python GAN/inspect_cwt.py verify --record 0
    python GAN/inspect_cwt.py browse --source clf --max-samples 40
    python GAN/inspect_cwt.py roundtrip --data-dir /data/gan_afib_upright \\
        --generator wavelet_generator_final_upright.keras
"""

import argparse
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import wfdb

from cwt_utils import (signal_to_scalogram, scalogram_to_signal, make_scales, reconstruction_constant, fs_from_scales, select_lead, bandpass_filter, fit_length, rhythm_index)
from config import GAN, CLASSIFIER, WAVELET, SEED, AFDB_BEST

VIEWS = ('abs', 'logabs', 're', 'im')
LABEL_NAMES = {0: 'N', 1: 'AFIB'}
BLUE, RED = '#004C99', '#CC0000'

# Median standard deviation of the R-R intervals (s) in real afdb N and
# AFIB windows. The roundtrip subcommand draws them as reference lines.
AFDB_RR_SD = {'N': 0.015, 'AFIB': 0.135}

def _render_scalogram(ax, sample, view='abs', cmap=None):
    """Draw a (n_scales, T, 2) Re/Im array on ax and return the image."""
    re, im = sample[..., 0], sample[..., 1]
    if view == 'abs':
        data, default_cmap = np.hypot(re, im), 'magma'
    elif view == 'logabs':
        data, default_cmap = np.log1p(np.hypot(re, im)), 'magma'
    elif view == 're':
        data, default_cmap = re, 'RdBu_r'
    elif view == 'im':
        data, default_cmap = im, 'RdBu_r'
    else:
        raise ValueError(f"Unknown view: {view!r}")
    return ax.imshow(data, aspect='auto', origin='lower', cmap=cmap or default_cmap)


def _source_path(source, data_dir):
    if source == 'gan':
        return os.path.join(data_dir, os.path.basename(GAN.scalogram_file))
    return os.path.join(CLASSIFIER.output_dir, 'X_scalograms.npy')


def _load_gan_set(data_dir):
    """Return the GAN set's scalograms (memory-mapped), per-window divisors and scales."""
    def path(cfg_path):
        return os.path.join(data_dir, os.path.basename(cfg_path))
    return (np.load(path(GAN.scalogram_file), mmap_mode='r'), np.load(path(GAN.scale_file)), np.load(path(GAN.scales_file)))


def cmd_verify(args):
    """Transform one afdb window, invert it and compare with the original."""
    rec_name = CLASSIFIER.records[args.record]
    record = wfdb.rdrecord(os.path.join(CLASSIFIER.data_path, rec_name))
    fs = record.fs
    sig = select_lead(record, db='afdb', channel=AFDB_BEST[rec_name])
    if WAVELET.apply_bandpass:
        sig = bandpass_filter(sig, fs, WAVELET.f_min, WAVELET.f_max)
    seg = fit_length(sig[:int(CLASSIFIER.seg_seconds * fs)], CLASSIFIER.seg_length)

    # Linear grid over the same scale range, for comparison.
    if args.spacing == 'log':
        scales = make_scales(fs, WAVELET.f_min, WAVELET.f_max, WAVELET.n_scales, WAVELET.fc)
    else:
        scales = np.linspace(WAVELET.fc * fs / WAVELET.f_max, WAVELET.fc * fs / WAVELET.f_min, WAVELET.n_scales)
    K = reconstruction_constant(WAVELET.fb, WAVELET.fc)

    scal, mean = signal_to_scalogram(seg, scales, WAVELET.wavelet)
    rec = scalogram_to_signal(scal, scales, K) + mean
    mid = slice(len(seg) // 4, 3 * len(seg) // 4)
    corr = np.corrcoef(seg[mid], rec[mid])[0, 1]
    rmse = np.sqrt(np.mean((seg[mid] - rec[mid]) ** 2))
    amp = np.mean(np.abs(rec[mid])) / max(np.mean(np.abs(seg[mid])), 1e-12)
    print(f"spacing={args.spacing}  view={args.view}  length={len(seg)}  "
          f"fs={fs}  scales=[{scales[0]:.2f}..{scales[-1]:.2f}]  "
          f"K={K:.4e}  corr={corr:.4f}  rmse={rmse:.4f}  "
          f"|rec|/|sig|={amp:.3f}")

    if args.view == 'both':
        panels = [('re', "Re channel"), ('im', "Im channel")]
    else:
        panels = [(args.view,
                   f"Scalogram  view={args.view}  spacing={args.spacing}")]
    n_rows = len(panels) + 2
    t = np.arange(len(seg)) / fs
    fig, ax = plt.subplots(n_rows, 1, figsize=(14, 3 * n_rows))
    ax[0].plot(t, seg, color=BLUE)
    ax[0].set_title("Original ECG")
    for a, (view, title) in zip(ax[1:-1], panels):
        _render_scalogram(a, scal, view=view)
        a.set_title(title)
    ax[-1].plot(t, seg, alpha=0.4, color=BLUE, label='original')
    ax[-1].plot(t, rec, color=RED, label=f'reconstructed (corr={corr:.4f})')
    ax[-1].legend()
    ax[-1].set_xlabel("Time (s)")
    for a in ax:
        a.grid(alpha=0.3)

    fig.suptitle(f"verify  spacing={args.spacing}  view={args.view}  "
                 f"record={rec_name}")
    plt.tight_layout()
    out = f'verify_{args.spacing}_{args.view}.png'
    plt.savefig(out, dpi=300)
    print(f"Saved {out}")


def _load_clf_context(out_dir):
    """Load the arrays needed to trace a classifier row back to its raw ECG."""
    return {
        'bounds': np.load(os.path.join(out_dir, 'window_bounds.npy')),
        'records': np.load(os.path.join(out_dir, 'record_ids.npy')),
        'scales': np.load(os.path.join(out_dir, 'scales.npy')),
        'means': np.load(os.path.join(out_dir, 'segment_means.npy')),
        'seg_scales': np.load(os.path.join(out_dir, 'segment_scales.npy')),
        'K': reconstruction_constant(WAVELET.fb, WAVELET.fc),
    }


def _rhythm_context(rhythm_ann, start, end, fs):
    """Return (rhythm at start, its annotation sample, seconds to the nearest rhythm change)."""
    r_samples, r_labels = rhythm_index(rhythm_ann)
    entry = int(np.searchsorted(r_samples, start, side='right'))
    if entry:
        label, at = r_labels[entry - 1], int(r_samples[entry - 1])
    else:
        label, at = None, None
    edges = r_samples[r_samples > 0]
    if len(edges):
        dist = float(np.min(np.abs(edges[:, None] - np.array([start, end])))) / fs
    else:
        dist = float('nan')
    return label, at, dist


def _plot_raw_window(ax, seg, peaks, fs, title):
    """Plot a raw ECG window with its R peaks and the R-R intervals below."""
    t = np.arange(len(seg)) / fs
    ax.plot(t, seg, color=BLUE, lw=0.9)
    if len(peaks):
        rail = float(np.max(seg)) + 0.12 * float(np.ptp(seg) or 1.0)
        ax.plot(peaks / fs, np.full(len(peaks), rail), 'v', color=RED, ms=5, label=f"{len(peaks)} R peaks")
        ax.legend(loc='upper right', fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("mV")
    ax.grid(alpha=0.3)
    rr = np.diff(peaks) / fs
    if len(rr):
        rr_txt = ' '.join(f"{v:.2f}" for v in rr)
        ax.set_xlabel(f"RR (s): {rr_txt}\nmean {rr.mean():.3f}  "
                      f"sd {rr.std():.3f}  min {rr.min():.2f}  "
                      f"max {rr.max():.2f}", fontsize=7.5)
    else:
        ax.set_xlabel("RR (s): fewer than 2 R peaks in window", fontsize=7.5)


def _render_clf_panels(data, chosen, y, ctx, out_dir, view):
    """Plot the raw ECG, stored tensor and ICWT of each chosen classifier row."""
    fs = CLASSIFIER.fs
    order = sorted(chosen, key=lambda i: (str(ctx['records'][i]), int(ctx['bounds'][i][0])))
    current_rec = None
    for done, i in enumerate(order, 1):
        i = int(i)
        rec = str(ctx['records'][i])
        start, end = (int(v) for v in ctx['bounds'][i])
        if rec != current_rec:
            path = os.path.join(CLASSIFIER.data_path, rec)
            sig_raw = select_lead(wfdb.rdrecord(path), db='afdb', channel=AFDB_BEST[rec])
            beats = np.asarray(wfdb.rdann(path, 'qrs').sample)
            rhythm_ann = wfdb.rdann(path, 'atr')
            current_rec = rec

        assigned = LABEL_NAMES[int(y[i])]
        ann_label, ann_at, boundary_s = _rhythm_context(rhythm_ann, start, end, fs)
        peaks = beats[(beats >= start) & (beats < end)] - start

        sample = np.asarray(data[i]).astype(np.float64)
        sample_scaled = sample * float(ctx['seg_scales'][i])
        recon = scalogram_to_signal(sample_scaled, ctx['scales'], ctx['K']) + float(ctx['means'][i])

        fig, ax = plt.subplots(3, 1, figsize=(13, 9))
        _plot_raw_window(ax[0], sig_raw[start:end], peaks, fs, f"raw ECG from {rec}.dat  [{start}, {end})")

        _render_scalogram(ax[1], sample, view=view)
        ax[1].set_title(f"stored tensor, view={view}  shape={sample.shape}")
        ax[1].set_ylabel("Scale index (low freq at top)")
        # Label the time axis in seconds.
        xt = np.linspace(0, sample.shape[1] - 1, 6)
        ax[1].set_xticks(xt)
        ax[1].set_xticklabels([f"{v / fs:.0f}" for v in xt])
        ax[1].set_xlabel("Time (s)")

        ax[2].plot(np.arange(len(recon)) / fs, recon, color=RED, lw=0.9)
        ax[2].set_title("ICWT of the stored tensor")
        ax[2].set_xlabel("Time (s)")
        ax[2].set_ylabel("mV")
        ax[2].grid(alpha=0.3)

        fig.suptitle(f"{rec}  [{start}, {end})   assigned={assigned}   "
                     f".atr={ann_label} @{ann_at}   "
                     f"nearest rhythm boundary {boundary_s:.1f} s",
                     fontsize=11)
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        plt.savefig(os.path.join(out_dir, f"sample_{i:06d}_{rec}_{assigned}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        if done % 25 == 0 or done == len(order):
            print(f"  {done}/{len(order)}", flush=True)


def _render_plain(data, indices, out_dir, view):
    """Save one PNG per scalogram."""
    indices = list(indices)
    pad = max(5, len(str(max(len(indices) - 1, 0))))
    for k, i in enumerate(indices, 1):
        sample = np.asarray(data[i])
        fig, ax = plt.subplots(figsize=(12, 4))
        image = _render_scalogram(ax, sample, view=view)
        ax.set_title(f"#{i}  view={view}  shape={sample.shape}")
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Scale index (low freq at top)")
        fig.colorbar(image, ax=ax, shrink=0.85)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sample_{i:0{pad}d}.png"), dpi=120)
        plt.close(fig)
        if k % 50 == 0 or k == len(indices):
            print(f"  {k}/{len(indices)}")


def cmd_browse(args):
    """Write PNGs of the first GAN rows, or of a balanced sample of classifier rows."""
    npy_path = _source_path(args.source, args.data_dir)
    out_dir = (args.out_dir or f"{os.path.splitext(npy_path)[0]}_pngs_{args.view}")
    os.makedirs(out_dir, exist_ok=True)
    data = np.load(npy_path, mmap_mode='r')
    print(f"Loaded {data.shape}  dtype={data.dtype}  from {npy_path}")

    if args.source == 'gan':
        n = min(args.max_samples, len(data))
        print(f"Rendering {n}/{len(data)} samples (view={args.view}) "
              f"-> {out_dir}")
        _render_plain(data, range(n), out_dir, args.view)
        print(f"Done. {n} PNGs in {out_dir}")
        return

    y = np.load(os.path.join(CLASSIFIER.output_dir, 'y_labels.npy'))
    if len(y) != len(data):
        raise SystemExit(f"label count {len(y)} != scalogram count "
                         f"{len(data)}")
    n_idx, af_idx = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    n_per_class = args.max_samples // 2
    if n_per_class > min(len(n_idx), len(af_idx)):
        n_per_class = min(len(n_idx), len(af_idx))
        print(f"  clipping to {n_per_class} per class")
    rng = np.random.default_rng(SEED)
    chosen = np.concatenate([rng.choice(n_idx, size=n_per_class, replace=False), rng.choice(af_idx, size=n_per_class, replace=False),])
    print(f"Rendering {len(chosen)}/{len(data)} samples (view={args.view}, "
          f"balanced {n_per_class} N + {n_per_class} AFIB) -> {out_dir}")

    ctx = _load_clf_context(CLASSIFIER.output_dir)
    _render_clf_panels(data, chosen, y, ctx, out_dir, args.view)
    print(f"Done. {len(chosen)} PNGs in {out_dir}")


def _detect_r_peaks(sig, fs):
    """Find the R peaks of a reconstructed signal, upright QRS or inverted."""
    from scipy.signal import find_peaks
    x = sig if abs(sig.max()) >= abs(sig.min()) else -sig
    prominence = max(0.5 * float(np.std(x)), 1e-6)
    peaks, _ = find_peaks(x, distance=int(0.3 * fs), prominence=prominence)
    return peaks


def _rr_stats(sig, fs):
    """Return (n_beats, mean R-R, sd R-R, CV) in seconds, NaN with fewer than 3 peaks."""
    peaks = _detect_r_peaks(sig, fs)
    if len(peaks) < 3:
        return len(peaks), np.nan, np.nan, np.nan
    rr = np.diff(peaks) / fs
    mean, sd = float(rr.mean()), float(rr.std(ddof=1))
    return len(peaks), mean, sd, (sd / mean if mean > 0 else np.nan)


def _consistency(tensor, scales, K):
    """Correlate a scalogram with the CWT of its own ICWT. Returns (correlation, signal)."""
    sig = scalogram_to_signal(tensor, scales, K)
    back, _ = signal_to_scalogram(sig, scales, WAVELET.wavelet)
    n = min(tensor.shape[1], back.shape[1])
    corr = np.corrcoef(tensor[:, :n, :2].ravel(), back[:, :n, :2].ravel())
    return float(corr[0, 1]), sig


def _roundtrip_rows(tensors, scales, K, fs):
    corrs, rr = [], []
    for tensor in tensors:
        c, sig = _consistency(tensor, scales, K)
        corrs.append(c)
        rr.append(_rr_stats(sig, fs))
    return np.array(corrs), np.array(rr, dtype=float)


def cmd_roundtrip(args):
    """Compare round-trip consistency and R-R stats of real and generated scalograms."""
    reals, seg_scales, scales = _load_gan_set(args.data_dir)
    K = reconstruction_constant(WAVELET.fb, WAVELET.fc)
    fs = fs_from_scales(scales, WAVELET.f_min, WAVELET.fc)
    rng = np.random.default_rng(args.seed)

    idx = rng.choice(len(reals), size=min(args.n, len(reals)), replace=False)
    real = (np.asarray(reals[i]).astype(np.float64) * float(seg_scales[i])
            for i in idx)
    rows = {'real': _roundtrip_rows(real, scales, K, fs)}

    if args.generator:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(SEED)
        gen = tf.keras.models.load_model(args.generator, compile=False)
        noise = tf.random.normal([args.n, GAN.latent_dim], seed=args.seed)
        amplitude = rng.choice(seg_scales, size=args.n)
        fake = gen.predict(noise, verbose=0) * amplitude[:, None, None, None]
        rows['generated'] = _roundtrip_rows(fake.astype(np.float64), scales, K, fs)

    print(f"\n{'set':<11}{'n':>5}{'consistency':>26}{'beats':>8}"
          f"{'meanRR':>9}{'sdRR':>9}{'cv':>8}")
    print('-' * 76)
    for name, (c, rr) in rows.items():
        with np.errstate(invalid='ignore'):
            print(f"{name:<11}{len(c):>5}"
                  f"{c.mean():>10.4f} +/-{c.std():.4f} min {c.min():>6.4f}"
                  f"{np.nanmedian(rr[:, 0]):>8.1f}"
                  f"{np.nanmedian(rr[:, 1]):>9.3f}"
                  f"{np.nanmedian(rr[:, 2]):>9.3f}"
                  f"{np.nanmedian(rr[:, 3]):>8.3f}")
    print(f"\nreference RR sd on afdb: real N median {AFDB_RR_SD['N']} s, "
          f"real AFIB median {AFDB_RR_SD['AFIB']} s")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, (c, rr) in rows.items():
        ax[0].hist(c, bins=30, alpha=0.6, label=name)
        ax[1].hist(rr[:, 2][~np.isnan(rr[:, 2])], bins=30, alpha=0.6, label=name)
    ax[0].set_xlabel("scalogram round-trip correlation")
    ax[1].set_xlabel("R-R standard deviation (s)")
    ax[1].axvline(AFDB_RR_SD['N'], color='g', ls='--', lw=1)
    ax[1].axvline(AFDB_RR_SD['AFIB'], color='r', ls='--', lw=1)
    for a in ax:
        a.legend()
        a.set_ylabel("count")
        a.grid(alpha=0.3)
    name = Path(args.generator).stem if args.generator else 'real only'
    fig.suptitle(f"round-trip consistency and rhythm, {name}")
    plt.tight_layout()
    plt.savefig(args.out, dpi=160, bbox_inches='tight')
    print(f"Saved {args.out}")


def _pairwise_l2(a, b):
    """Return the L2 distance between every row of a and every row of b."""
    sq = ((a ** 2).sum(axis=1)[:, None] + (b ** 2).sum(axis=1)[None, :] - 2.0 * (a @ b.T))
    return np.sqrt(np.maximum(sq, 0.0))


def cmd_collapse(args):
    """Check a generator for mode collapse and memorisation."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading generator from {args.generator}")
    gen = tf.keras.models.load_model(args.generator, compile=False)
    noise = tf.random.normal([args.n, GAN.latent_dim])
    fakes = gen(noise, training=False).numpy()

    flat = fakes.reshape(args.n, -1).astype(np.float32)
    upper = np.triu_indices(args.n, k=1)
    fake_dist = _pairwise_l2(flat, flat)[upper]

    real_path = os.path.join(args.data_dir, os.path.basename(GAN.scalogram_file))
    print(f"Loading reals from {real_path}")
    reals = np.load(real_path, mmap_mode='r')
    rng = np.random.default_rng(SEED)
    n_real = min(args.n_real, len(reals))
    real_idx = rng.choice(len(reals), size=n_real, replace=False)
    real_flat = np.asarray(reals[real_idx]).astype(np.float32)
    real_flat = real_flat.reshape(n_real, -1)
    nearest = _pairwise_l2(flat, real_flat).min(axis=1)
    real_upper = np.triu_indices(n_real, k=1)
    real_dist = _pairwise_l2(real_flat, real_flat)[real_upper]

    # Plot the Re channel only, in the largest square grid that n allows.
    side = int(np.floor(np.sqrt(args.n)))
    if side * side < args.n:
        print(f"Warning: n={args.n} is not a perfect square; plotting the "
              f"first {side * side} samples in a {side}x{side} grid.")
    fig, axes = plt.subplots(side, side, figsize=(2 * side, 2 * side))
    for i, ax in enumerate(np.atleast_1d(axes).flatten()):
        ax.imshow(fakes[i, :, :, 0], aspect='auto', cmap='RdBu_r', origin='lower')
        ax.set_title(f"#{i}", fontsize=7)
        ax.axis('off')
    plt.tight_layout()
    grid_path = os.path.join(args.out, 'collapse_grid.png')
    plt.savefig(grid_path, dpi=120)
    plt.close(fig)

    print(f"[collapse] n={args.n}  "
          f"mean_pairwise_L2={fake_dist.mean():.4f}  "
          f"min_pairwise_L2={fake_dist.min():.4f}  "
          f"mean_pixel_std={fakes.std(axis=0).mean():.4f}")
    print(f"[memorize] mean_NN={nearest.mean():.4f}  "
          f"min_NN={nearest.min():.4f}  "
          f"ref_real_pairwise={real_dist.mean():.4f}  -> {grid_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    fmt = argparse.ArgumentDefaultsHelpFormatter
    gan_dir = os.path.dirname(GAN.scalogram_file) or '.'

    sp = sub.add_parser('verify', help="CWT round trip on one afdb record", formatter_class=fmt)
    sp.add_argument('--record', type=int, default=0,
                    help="index into CLASSIFIER.records")
    sp.add_argument('--spacing', default='log', choices=['log', 'linear'],
                    help="scale spacing")
    sp.add_argument('--view', default='abs', choices=VIEWS + ('both',),
                    help="scalogram view; 'both' plots Re and Im separately")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser('browse', help="one PNG per stored scalogram", formatter_class=fmt)
    sp.add_argument('--source', default='gan', choices=['gan', 'clf'],
                    help="which set to read")
    sp.add_argument('--data-dir', default=gan_dir,
                    help="GAN set to read with --source gan")
    sp.add_argument('--max-samples', type=int, default=100,
                    help="number of PNGs")
    sp.add_argument('--view', default='abs', choices=VIEWS, help="scalogram view")
    sp.add_argument('--out-dir', default=None,
                    help="output folder")
    sp.set_defaults(func=cmd_browse)

    sp = sub.add_parser('roundtrip', help="validity and rhythm of samples", formatter_class=fmt)
    sp.add_argument('--data-dir', default=gan_dir,
                    help="directory of the real GAN set")
    sp.add_argument('--generator', default=None,
                    help="generator .keras file to compare with the real set")
    sp.add_argument('--n', type=int, default=64,
                    help="samples per set")
    sp.add_argument('--seed', type=int, default=SEED, help="RNG seed")
    sp.add_argument('--out', default='roundtrip.png', help="output PNG")
    sp.set_defaults(func=cmd_roundtrip)

    sp = sub.add_parser('collapse', help="diversity and memorisation check", formatter_class=fmt)
    sp.add_argument('--generator', required=True,
                    help="generator .keras file")
    sp.add_argument('--out', required=True, help="output directory")
    sp.add_argument('--n', type=int, default=64,
                    help="generated samples")
    sp.add_argument('--data-dir', default=gan_dir,
                    help="directory of the real GAN set")
    sp.add_argument('--n-real', type=int, default=500,
                    help="real samples used as reference")
    sp.set_defaults(func=cmd_collapse)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
