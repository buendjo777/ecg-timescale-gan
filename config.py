"""Configuration shared throughout the GAN and classifier scripts.

All parameters are declared here and cannot be changed during trainings.
Import the instances from config import WAVELET, GAN, CLASSIFIER
"""

import os
from dataclasses import dataclass

SEED = 42

# Root of all data paths below. Override it with the DATA_DIR environment variable.
DATA_DIR = os.environ.get('DATA_DIR', '/data')


@dataclass(frozen=True)
class WaveletConfig:
    """Complex Morlet wavelet and scale grid used for every CWT."""
    wavelet: str   = 'cmor1.5-1.0'   # pywt name, must match fb and fc
    fb:      float = 1.5             # bandwidth
    fc:      float = 1.0             # centre frequency
    f_min:   float = 0.5             # the scales cover [f_min, f_max] Hz
    f_max:   float = 40.0
    n_scales: int  = 64
    # Bandpass the ECG to [f_min, f_max] before the CWT.
    apply_bandpass: bool = True


@dataclass(frozen=True)
class GANConfig:
    """WGAN-GP data and training parameters."""
    # GAN windows. fs and the window length come from CLASSIFIER.
    overlap:     float = 0.5
    min_segments_per_record: int = 10

    latent_dim: int   = 128
    batch_size: int   = 128
    lr:         float = 5e-5
    lambda_gp:  float = 10.0
    n_critic:   int   = 5            # critic updates per generator update

    # Default GAN set: afdb AFIB windows, upright group.
    scalogram_file: str = f'{DATA_DIR}/gan_afib_upright/processed_scalograms.npy'
    scales_file:    str = f'{DATA_DIR}/gan_afib_upright/cwt_scales.npy'
    means_file:     str = f'{DATA_DIR}/gan_afib_upright/segment_means.npy'    # mean removed before the CWT
    scale_file:     str = f'{DATA_DIR}/gan_afib_upright/segment_scales.npy'   # per-window divisor
    gen_model_file: str = 'wavelet_generator_final.keras'
    results_dir:    str = 'WGAN_results'


@dataclass(frozen=True)
class ClassifierConfig:
    """AF classifier data and training parameters."""
    data_path: str   = f'{DATA_DIR}/mitdb_atf/'
    # 00735 and 03665 are excluded as they have no signal provided, only annotations.
    records:   tuple = (
        '04015', '04043', '04048', '04126', '04746', '04908', '04936',
        '05091', '05121', '05261', '06426', '06453', '06995', '07162',
        '07859', '07879', '07910', '08215', '08219', '08378', '08405',
        '08434', '08455',
    )
    fs:        int   = 250

    seg_seconds: float = 10.0
    seg_length:  int   = 2500        # seg_seconds * fs
    overlap:     float = 0.0

    n_folds:       int   = 5
    batch_size:    int   = 64
    # Phase 1 trains only the head, phase 2 also the top n_unfreeze layers of the backbone.
    phase1_lr:     float = 1e-3
    phase1_epochs: int   = 15
    phase2_lr:     float = 1e-5
    phase2_epochs: int   = 10
    n_unfreeze:    int   = 20

    output_dir:  str = f'{DATA_DIR}/af_classification/'
    results_dir: str = 'AF_Classification_Results'


WAVELET    = WaveletConfig()
GAN        = GANConfig()
CLASSIFIER = ClassifierConfig()

# afdb records that are AFIB for their whole length.
PURE_AFIB_RECORDS = ('07162', '07859')

# Windows whose band-passed amplitude exceeds this (mV) are dropped.
SATURATION_MV = 10


# Channel was chosen with the help of a doctor as leads are not explicitely stated.
AFDB_BEST = {
    '04015': 'ECG2', '04043': 'ECG1', '04048': 'ECG2', '04126': 'ECG2',
    '04746': 'ECG2', '04908': 'ECG2', '04936': 'ECG2', '05091': 'ECG2',
    '05121': 'ECG2', '05261': 'ECG1', '06426': 'ECG2', '06453': 'ECG2',
    '06995': 'ECG1', '07162': 'ECG1', '07859': 'ECG2', '07879': 'ECG2',
    '07910': 'ECG1', '08215': 'ECG2', '08219': 'ECG2', '08378': 'ECG2',
    '08405': 'ECG1', '08434': 'ECG2', '08455': 'ECG2',
}

# The two GAN training groups, picked by eye from 5 s strips of both channels.
# 04908 Not included due to constant noise in the recording.
AFDB_UPRIGHT = {
    '04015': ('ECG1',),         '04043': ('ECG1',),
    '04048': ('ECG1',),         '04126': ('ECG2',),
    '04746': ('ECG1', 'ECG2'),  '04936': ('ECG2',),
    '05091': ('ECG2',),         '05121': ('ECG2',),
    '05261': ('ECG1',),         '06995': ('ECG1', 'ECG2'),
    '07162': ('ECG2',),         '07859': ('ECG2',),
    '07879': ('ECG1', 'ECG2'),  '07910': ('ECG1',),
    '08215': ('ECG2',),         '08219': ('ECG1', 'ECG2'),
    '08405': ('ECG1',),         '08434': ('ECG2',),
    '08455': ('ECG1',),
}

AFDB_INVERTED = {
    '04015': ('ECG2',),         '04043': ('ECG2',),
    '04126': ('ECG1',),         '04936': ('ECG1',),
    '05091': ('ECG1',),         '05121': ('ECG1',),
    '05261': ('ECG2',),         '06426': ('ECG2',),
    '06453': ('ECG2',),         '07162': ('ECG1',),
    '07859': ('ECG1',),         '07910': ('ECG2',),
    '08215': ('ECG1',),         '08378': ('ECG1', 'ECG2'),
    '08405': ('ECG2',),         '08434': ('ECG1',),
    '08455': ('ECG2',),
}

AFDB_LEAD_GROUPS = {'best': AFDB_BEST, 'upright': AFDB_UPRIGHT,
                    'inverted': AFDB_INVERTED}
