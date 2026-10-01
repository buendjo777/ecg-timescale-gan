"""Builds the WGAN training set: 10 s ECG windows turned into scalograms.

The windows come from afdb and are only used if they are within an AFIB window.
Records left with fewer than GAN.min_segments_per_record windows are filtered out. 
Each window is divided by its own max |W|.

Usage:
    python GAN/Wavelet_Preprocessor.py
    python GAN/Wavelet_Preprocessor.py --lead-group inverted \\
        --out-dir /data/gan_afib_inverted
"""

import argparse
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import wfdb

from cwt_utils import (make_scales, signal_to_scalogram, bandpass_filter, select_lead, window_starts, rhythm_index, pure_rhythm)
from config import (GAN, CLASSIFIER, WAVELET, DATA_DIR, SATURATION_MV, AFDB_LEAD_GROUPS)


def record_channels(leads, rec):
    """Return the channels to read from a record."""
    chans = leads.get(rec)
    if isinstance(chans, str):
        return (chans,)
    return chans or (None,)


def extract_rhythm_segments(sig, rhythm_ann, rhythm):
    """Return the windows that lie entirely inside a `rhythm` episode."""
    r_samples, r_labels = rhythm_index(rhythm_ann)
    win = CLASSIFIER.seg_length
    segs = []
    for i in window_starts(len(sig), win, GAN.overlap):
        if pure_rhythm(r_samples, r_labels, i, i + win) != rhythm:
            continue
        seg = sig[i:i + win]
        if np.any(np.isnan(seg)) or np.max(np.abs(seg)) > SATURATION_MV:
            continue
        segs.append(seg)
    return segs


def collect_segments(records, rhythm, leads):
    """Read every record and return (segments, per_record)."""
    fs = CLASSIFIER.fs
    segments, per_record = [], {}
    for rec in records:
        path = os.path.join(CLASSIFIER.data_path, rec)
        try:
            record = wfdb.rdrecord(path)
            ann = wfdb.rdann(path, 'atr')
        except Exception as e:
            print(f"  {rec}: skipped ({e})")
            continue

        n_before = len(segments)
        for chan in record_channels(leads, rec):
            sig = select_lead(record, db='afdb', channel=chan)
            if WAVELET.apply_bandpass:
                sig = bandpass_filter(sig, fs, WAVELET.f_min, WAVELET.f_max)
            segments.extend(extract_rhythm_segments(sig, ann, rhythm))
        n_added = len(segments) - n_before

        if n_added < GAN.min_segments_per_record:
            del segments[n_before:]
            per_record[rec] = 0
            print(f"  {rec}: dropped ({n_added} windows < "
                  f"{GAN.min_segments_per_record})")
        else:
            per_record[rec] = n_added
            print(f"  {rec}: +{n_added} segments", flush=True)
    return segments, per_record


def build_dataset(out_dir, rhythm='AFIB', records=None, lead_group='upright'):
    """Select windows, compute their scalograms and save them to out_dir."""
    fs, seg_length = CLASSIFIER.fs, CLASSIFIER.seg_length
    leads = AFDB_LEAD_GROUPS[lead_group]
    # TODO: Slight concern on whether synthetic beats from the GAN (trained on all patients) means data leakage for
    # the Classifier, maybe train the GAN without the test patients?
    records = list(records) if records else sorted(leads)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[fs={fs}  length={seg_length}  overlap={GAN.overlap}  "
          f"rhythm={rhythm}  leads={lead_group}]  "
          f"reading {len(records)} records", flush=True)
    segments, per_record = collect_segments(records, rhythm, leads)

    n_records_kept = sum(1 for n in per_record.values() if n)
    print(f"\nKept {n_records_kept}/{len(records)} records -> "
          f"{len(segments)} segments total")

    n = len(segments)
    if n == 0:
        raise RuntimeError(f"No {rhythm} segments survived in {CLASSIFIER.data_path}.")
    shape = (n, WAVELET.n_scales, seg_length, 2)
    print(f"\nComputing {n} CWTs -> {shape} "
          f"({np.prod(shape) * 4 / 1e9:.1f} GB, fp32)", flush=True)

    scales = make_scales(fs, WAVELET.f_min, WAVELET.f_max, WAVELET.n_scales, WAVELET.fc)
    paths = {key: os.path.join(out_dir, os.path.basename(cfg_path))
             for key, cfg_path in (('scalogram', GAN.scalogram_file),
                                   ('scales', GAN.scales_file),
                                   ('means', GAN.means_file),
                                   ('scale', GAN.scale_file))}
    X = np.lib.format.open_memmap(paths['scalogram'], mode='w+', dtype=np.float32, shape=shape)
    means = np.empty(n, dtype=np.float32)
    seg_scales = np.empty(n, dtype=np.float32)

    for i, seg in enumerate(segments):
        tensor, means[i] = signal_to_scalogram(seg, scales, WAVELET.wavelet)
        scale = float(np.max(np.hypot(tensor[..., 0], tensor[..., 1])))
        if scale <= 0:
            scale = 1.0
        X[i] = tensor / scale
        seg_scales[i] = scale
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}", flush=True)

    X.flush()
    np.save(paths['scales'], scales)
    np.save(paths['means'], means)
    np.save(paths['scale'], seg_scales)

    print(f"\nSaved {shape} -> {paths['scalogram']}  "
          f"(per-segment scales: median {float(np.median(seg_scales)):.4f}, "
          f"max {float(seg_scales.max()):.4f})")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--rhythm', default='AFIB',
                   help="rhythm to keep")
    p.add_argument('--records', default=None,
                   help="comma-separated subset of records")
    p.add_argument('--out-dir', default=None,
                   help="output directory, gan_afib_<lead-group> in DATA_DIR if not given")
    p.add_argument('--lead-group', default='upright',
                   choices=sorted(AFDB_LEAD_GROUPS),
                   help="channel map from config.AFDB_LEAD_GROUPS")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(DATA_DIR, f'gan_afib_{args.lead_group}')
    build_dataset(out_dir,
                  rhythm=args.rhythm,
                  records=args.records.split(',') if args.records else None,
                  lead_group=args.lead_group)


if __name__ == "__main__":
    main()
