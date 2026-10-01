"""This script includes all of the functions to do with the continous wavelet transform used by the CNN and the GAN.

The forward transform is pywt's complex Morlet CWT. The inverse is the
single-integral reconstruction derived mathematically:

    x(t) = 2 / K * Re( sum_j W(a_j, t) * da_j / a_j^(3/2) )

discretised over a log-spaced grid of scales a_j, with K given by the function reconstruction_constant(). """

import numpy as np
import pywt
from scipy.signal import butter, filtfilt

DEFAULT_LEAD = {'mitdb': 'MLII', 'afdb': 'ECG1'}

# All four afdb rhythms are indexed even though only N and AFIB are kept.
RHYTHMS = {'AFIB', 'AFL', 'J', 'N'}


def make_scales(fs, f_min, f_max, n_scales, fc):
    """Makes log-spaced scales covering [f_min, f_max] Hz at sampling rate fs.
    Scale a corresponds to frequency fc * fs / a, so the smallest scale maps to f_max and the largest to f_min.
    Was used at first because MITDB and AFDB bases are sampled at different frequencies."""
    return np.geomspace(fc * fs / f_max, fc * fs / f_min, n_scales)


def fs_from_scales(scales, f_min, fc):
    """Sampling rate that a grid from make_scales() was built for."""
    return float(scales[-1]) * f_min / fc


def reconstruction_constant(fb, fc, n=8192):
    """Constant K of the single-integral inverse CWT for cmor(fb, fc).
    K = integral over w > 0 of psi_hat(w) / w, with psi_hat(w) = exp(-fb * (w - 2 pi fc)^2 / 4)."""
    omega = np.geomspace(1e-4, 200.0, n)
    psi_hat = np.exp(-fb * (omega - 2.0 * np.pi * fc) ** 2 / 4.0)
    trapezoid = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(trapezoid(psi_hat / omega, omega))


def signal_to_scalogram(sig, scales, wavelet):
    """CWT of a 1-D signal as a (n_scales, len(sig), 2) array of Re and Im channels.
    The mean is removed before applying the transform and returned later with the result, see scalogram_to_signal()"""
    mean = float(np.mean(sig))
    coefs, _ = pywt.cwt(sig - mean, scales, wavelet)
    return np.stack([coefs.real, coefs.imag], axis=-1), mean


def _midpoint_widths(scales):
    """Integration widths, da_j, for log-spaced scales.
    Interior widths are half the distance between neighbours.
    The end points extend half a step past the grid."""
    delta_a = np.empty(len(scales))
    delta_a[1:-1] = (scales[2:] - scales[:-2]) / 2.0
    r_left = scales[1] / scales[0]
    r_right = scales[-1] / scales[-2]
    delta_a[0] = (scales[1] - scales[0] / r_left) / 2.0
    delta_a[-1] = (scales[-1] * r_right - scales[-2]) / 2.0
    return delta_a


def scalogram_to_signal(tensor, scales, C_psi):
    """Discrete inverse CWT of a (n_scales, T, 2) Re/Im array.

        x(t) = 2 / K * Re( sum_j W(a_j, t) * da_j / a_j^(3/2) )

    with C_psi = K from reconstruction_constant(). 
    The result has zero mean; add back the mean returned by signal_to_scalogram()."""
    coefs = tensor[:, :, 0] + 1j * tensor[:, :, 1]
    delta_a = _midpoint_widths(scales)
    weighted = coefs / scales[:, None] ** 1.5 * delta_a[:, None]
    summed = weighted.sum(axis=0)
    return 2.0 * np.real(summed) / C_psi


def bandpass_filter(sig, fs, low=0.5, high=40.0, order=4):
    """Zero-phase Butterworth band-pass filter.
    Standard filter for ECGs."""
    b, a = butter(order, [low, high], btype='band', fs=fs)
    return filtfilt(b, a, sig)


def select_lead(record, db='mitdb', channel=None):
    """Return one channel of a wfdb record as a 1-D array.
    The channel is chosen by name: 'channel' if given, if not MLII for mitdb and ECG1 for afdb."""
    if db not in DEFAULT_LEAD:
        raise ValueError(f"Unknown db: {db}")
    names = list(record.sig_name)
    if channel is not None and channel not in names:
        print(f"  WARNING: {channel} not in {names}, falling back")
    for name in (channel, DEFAULT_LEAD[db]):
        if name in names:
            return record.p_signal[:, names.index(name)]
    print(f"  WARNING: {record.record_name} has no {DEFAULT_LEAD[db]}, "
          f"using {names[0]}")
    return record.p_signal[:, 0]


def window_starts(sig_len, win, overlap=0.0):
    """Start index of every complete window of 'win' samples."""
    step = max(1, int(win * (1.0 - overlap)))
    return range(0, sig_len - win + 1, step)


def fit_length(sig, target):
    """Crop or pad a 1-D signal to exactly 'target' samples."""
    if len(sig) >= target:
        return sig[:target]
    return np.pad(sig, (0, target - len(sig)), mode='reflect')


def rhythm_index(rhythm_ann):
    """Turn a record's .atr annotations into a list of rhythm changes.
    Returns (samples, labels): the sample where each rhythm starts and the rhythm's name."""
    samples, labels = [], []
    for s, aux in zip(rhythm_ann.sample, rhythm_ann.aux_note):
        # aux_note looks like "(AFIB", so strip it down to the rhythm name.
        aux = aux.strip().strip('(').strip(')')
        if aux in RHYTHMS:
            samples.append(int(s))
            labels.append(aux)
    return np.asarray(samples, dtype=np.int64), labels


def pure_rhythm(r_samples, r_labels, start, end):
    """Returns the rhythm covering all of [start, end), or None.
    The rhythm at 'start' is set by the last annotation at or before it.
    None means the rhythm changes inside the window, or the window starts before the first annotation."""
    entry = int(np.searchsorted(r_samples, start, side='right'))
    before_end = int(np.searchsorted(r_samples, end, side='left'))
    if before_end > entry:
        return None
    if entry == 0:
        return None
    return r_labels[entry - 1]
