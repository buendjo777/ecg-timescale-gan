"""Build the AF classifier's training data from afdb.

Each afdb record is cut into 10 s windows, and a window is kept only if a
single rhythm covers all of it (strict purity, see cwt_utils.pure_rhythm).
Every kept window becomes a 2-channel CWT tensor (real and imaginary parts) labelled with its rhythm.

Usage:
    python Classifier/Classifier_Preprocessor.py

Input: afdb records (.dat, .hea, .atr) in CLASSIFIER.data_path.
Output, in CLASSIFIER.output_dir:
    X_scalograms.npy    (n, 64, 2500, 2)  CWT tensors, float16
    y_labels.npy        (n,)              0 = N, 1 = AFIB
    record_ids.npy      (n,)              record each window comes from
    window_bounds.npy   (n, 2)            [start, end) in samples
    segment_scales.npy  (n,)              normalisation divisor of each window
    segment_means.npy   (n,)              mean removed before the CWT
    scales.npy          (64,)             CWT scale grid
"""

import os
import sys
from collections import Counter
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import wfdb

from cwt_utils import (make_scales, signal_to_scalogram, bandpass_filter, select_lead, window_starts, rhythm_index, pure_rhythm)
from config import (CLASSIFIER, WAVELET, AFDB_BEST, SATURATION_MV)

SCALES = make_scales(CLASSIFIER.fs, WAVELET.f_min, WAVELET.f_max, WAVELET.n_scales, WAVELET.fc)

LABEL_MAP = {'N': 0, 'AFIB': 1}


def segment_indices(sig_len):
    """Returns the (start, end) sample indices of every window in the signal."""
    win = int(CLASSIFIER.seg_seconds * CLASSIFIER.fs)
    return [(i, i + win)
            for i in window_starts(sig_len, win, CLASSIFIER.overlap)]


def _read_record(rec):
    """Reads one afdb record and returns (filtered signal, rhythm annotations).
    Returns (None, None) if the record cannot be read."""
    path = os.path.join(CLASSIFIER.data_path, rec)
    try:
        record     = wfdb.rdrecord(path)
        rhythm_ann = wfdb.rdann(path, 'atr')
    except Exception as e:
        print(f"  {rec}: error: {e}")
        return None, None

    sig = select_lead(record, db='afdb', channel=AFDB_BEST[rec])
    if WAVELET.apply_bandpass:
        sig = bandpass_filter(sig, CLASSIFIER.fs, WAVELET.f_min, WAVELET.f_max)
    return sig, rhythm_ann


def _plan_record(sig, rhythm_ann):
    """Chooses which windows of one record to keep, without applying the CWT.
    Returns (kept, counts) where kept is a list (start, end, label), and counts
    holds the kept windows per class and the skipped windows with the reason for it."""
    r_samples, r_labels = rhythm_index(rhythm_ann)
    kept, counts = [], Counter()

    for start, end in segment_indices(len(sig)):
        # No rhythm annotation yet (usually the record's first window).
        if len(r_samples) == 0 or start < r_samples[0]:
            counts['skip_nolabel'] += 1
            continue

        label = pure_rhythm(r_samples, r_labels, start, end)
        if label is None:
            counts['skip_mixed'] += 1
            continue
        if label not in LABEL_MAP:
            counts['skip_other_rhythm'] += 1
            continue

        seg = sig[start:end]
        # Drop windows with NaNs or a band-passed amplitude above SATURATION_MV.
        if np.any(np.isnan(seg)) or np.max(np.abs(seg)) > SATURATION_MV:
            counts['skip_badsignal'] += 1
            continue

        kept.append((start, end, LABEL_MAP[label]))
        counts[label] += 1

    return kept, counts


def build_dataset():
    """Run the preprocessing and write the arrays to CLASSIFIER.output_dir."""
    out = CLASSIFIER.output_dir
    os.makedirs(out, exist_ok=True)

    records = list(CLASSIFIER.records)

    print(f"[length={CLASSIFIER.seg_length}  overlap={CLASSIFIER.overlap}]  "
          f"{len(records)} records")

    # Pass 1: decide which windows to keep.
    plan, stats = {}, Counter()
    for rec in records:
        sig, rhythm_ann = _read_record(rec)
        if sig is None:
            continue
        kept, counts = _plan_record(sig, rhythm_ann)
        plan[rec] = kept
        stats += counts
        skipped = (counts['skip_nolabel'] + counts['skip_mixed'] + counts['skip_other_rhythm'] + counts['skip_badsignal'])
        print(f"  {rec}: N={counts['N']:4d}  AFIB={counts['AFIB']:4d}  "
              f"skipped={skipped:4d}")

    n_rows = sum(len(v) for v in plan.values())
    if n_rows == 0:
        raise RuntimeError(f"No windows survived in {CLASSIFIER.data_path}.")
    x_dtype = np.float16
    shape = (n_rows, WAVELET.n_scales, CLASSIFIER.seg_length, 2)
    size_gb = np.prod(shape) * np.dtype(x_dtype).itemsize / 1e9
    print(f"\nPass 1 kept {n_rows} windows -> allocating {shape} "
          f"({size_gb:.1f} GB) on disk")

    # Pass 2: compute the CWT of every kept window.
    X = np.lib.format.open_memmap(os.path.join(out, 'X_scalograms.npy'), mode='w+', dtype=x_dtype, shape=shape)
    y      = np.empty(n_rows, dtype=np.int32)
    recs   = np.empty(n_rows, dtype='<U5')
    means  = np.empty(n_rows, dtype=np.float32)
    scales_per_seg = np.empty(n_rows, dtype=np.float32)
    bounds = np.empty((n_rows, 2), dtype=np.int64)

    row = 0
    for rec in records:
        kept = plan.get(rec)
        if not kept:
            continue
        sig, _ = _read_record(rec)
        if sig is None:
            continue
        for start, end, label in kept:
            seg = sig[start:end].astype(np.float32)
            tensor, mean = signal_to_scalogram(seg, SCALES, WAVELET.wavelet)
            scale = float(np.max(np.hypot(tensor[..., 0], tensor[..., 1])))
            if scale > 0:
                tensor = tensor / scale
            else:
                scale = 1.0

            X[row]              = tensor
            y[row]              = label
            recs[row]           = rec
            means[row]          = mean
            scales_per_seg[row] = scale
            bounds[row]         = (start, end)
            row += 1
        print(f"  {rec}: {len(kept)} transformed  ({row}/{n_rows})", flush=True)
    assert row == n_rows, f"wrote {row} rows, planned {n_rows}"
    X.flush()
    np.save(os.path.join(out, 'y_labels.npy'),       y)
    np.save(os.path.join(out, 'record_ids.npy'),     recs)
    np.save(os.path.join(out, 'scales.npy'),         SCALES)
    np.save(os.path.join(out, 'segment_means.npy'),  means)
    np.save(os.path.join(out, 'segment_scales.npy'), scales_per_seg)
    np.save(os.path.join(out, 'window_bounds.npy'),  bounds)

    print(f"\n{shape}  N={int(np.sum(y == 0))}  AFIB={int(np.sum(y == 1))}"
          f"  -> {out}")
    print(f"Skipped: no-label={stats['skip_nolabel']}, "
          f"mixed={stats['skip_mixed']}, "
          f"other-rhythm={stats['skip_other_rhythm']}, "
          f"bad-signal={stats['skip_badsignal']}")


if __name__ == "__main__":
    build_dataset()
