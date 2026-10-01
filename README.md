# Generative Adversarial Networks in the Time-Scale Domain for ECG Augmentation and Diagnosis

Bachelor's thesis (TFG) code, ICAI - Universidad Pontificia Comillas.

The project asks whether synthetic ECG data can help an atrial
fibrillation (AF) classifier. Ten-second ECG windows are turned into
complex Morlet wavelet scalograms, a Wasserstein GAN with gradient
penalty learns to generate new ones, and an ImageNet CNN is trained to
tell AF from non-AF windows, with and without the synthetic data.

## Pipeline

1. **Preprocessing.**
   `Classifier/Classifier_Preprocessor.py` cuts the MIT-BIH Atrial
   Fibrillation Database (afdb) into 10 s windows that lie entirely
   inside a single rhythm, N or AFIB, and stores the CWT of each one.
   `GAN/Wavelet_Preprocessor.py` builds the GAN training set from afdb
   AFIB episodes with the same purity rule, using windows that overlap
   by 50% and one of the channel groups in `config.py`.
2. **Generation.**
   `GAN/Wavelet_WGAN.py` trains the WGAN-GP on those scalograms.
   `GAN/Wavelet_Reconstruct.py` draws samples and inverts them back to
   ECG with the inverse CWT.
3. **Classification.**
   `Classifier/AFIB_CNN_Classifier.py` fine-tunes ResNet50 and
   EfficientNetB0 under cross-validation grouped by patient, optionally
   adding synthetic windows to the training folds.


`GAN/inspect_cwt.py` holds the diagnostics and tools: CWT round trips, 
dataset browsing, and validity and mode-collapse checks for trained generators.

## Layout

```
config.py                  all parameters (wavelet, GAN, classifier, afdb channels)
cwt_utils.py               forward and inverse CWT, filtering, windowing, afdb rhythm lookup
generate_figs.py           thesis figures 2.3 and 5.1-5.3
Classifier/
  Classifier_Preprocessor.py
  AFIB_CNN_Classifier.py
  RR_Baseline.py           R-R interval baseline, no CWT
  plot_confusion.py        confusion-matrix grid of every results_*.npy
GAN/
  Wavelet_Preprocessor.py
  Wavelet_WGAN.py
  Wavelet_Reconstruct.py
  inspect_cwt.py
```

## Setup

```bash
pip install -r requirements.txt
```

The afdb database is public on PhysioNet:

```bash
python -c "import wfdb; wfdb.dl_database('afdb', 'data/mitdb_atf')"
```

All paths in `config.py` start at `$DATA_DIR`, which defaults to `/data`.
Without Docker, run the scripts with `DATA_DIR=./data`. 
With Docker, `docker compose up -d` mounts `./data` (or `$DATA_DIR` if set) at `/data`.

## Usage

```bash
# classifier data and training
python Classifier/Classifier_Preprocessor.py
python Classifier/AFIB_CNN_Classifier.py --backbone EfficientNetB0
python Classifier/RR_Baseline.py
python Classifier/plot_confusion.py

# GAN data and training
python GAN/Wavelet_Preprocessor.py --lead-group upright
python GAN/Wavelet_WGAN.py --tag upright

# samples and checks
python GAN/Wavelet_Reconstruct.py --model wavelet_generator_final_upright.keras
python GAN/inspect_cwt.py roundtrip --generator wavelet_generator_final_upright.keras

# figures
python generate_figs.py --out figures
```

## Results

## Use of AI tools
AI coding assistants were used to assist in generating the figures used (generate_figs.py)