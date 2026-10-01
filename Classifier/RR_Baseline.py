"""Classical AF detector built on R-R interval statistics alone: No CWT, no scalogram, no GPU.

The windows, the strict-purity labels and their order come from 
Classifier_Preprocessor (_read_record and _plan_record), and the folds from
the same StratifiedGroupKFold call and seed as AFIB_CNN_Classifier.
R peaks come from the .qrs annotations.

Usage:
    python Classifier/RR_Baseline.py

Writes results_rr_baseline.npy into CLASSIFIER.results_dir.
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import wfdb

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, confusion_matrix, roc_auc_score)

from config import CLASSIFIER, SEED
import Classifier.Classifier_Preprocessor as prep


FEATURE_NAMES = (
    'n_beats', 'mean_rr', 'sd_rr', 'cv_rr', 'rmssd', 'rmssd_norm',
    'pnn50', 'pnn20', 'median_abs_drr', 'range_rr',
    'sd1', 'sd2', 'sd1_sd2', 'rr_entropy',
)


def rr_features(peaks, fs):
    """Interval statistics for one window. NaN where a window is too short."""
    out = np.full(len(FEATURE_NAMES), np.nan)
    out[0] = len(peaks)
    if len(peaks) < 2:
        return out

    rr = np.diff(peaks) / fs
    out[1] = rr.mean()
    if len(rr) < 2:
        return out

    drr = np.diff(rr)
    sd_rr = rr.std(ddof=1)
    rmssd = np.sqrt(np.mean(drr ** 2))
    sd_drr = drr.std(ddof=1) if len(drr) > 1 else 0.0

    out[2] = sd_rr
    out[3] = sd_rr / rr.mean() if rr.mean() > 0 else np.nan
    out[4] = rmssd
    out[5] = rmssd / rr.mean() if rr.mean() > 0 else np.nan
    out[6] = np.mean(np.abs(drr) > 0.05)
    out[7] = np.mean(np.abs(drr) > 0.02)
    out[8] = np.median(np.abs(drr))
    out[9] = rr.max() - rr.min()

    # Poincare descriptors. sd1 is short-term variability, sd2 long-term.
    sd1 = np.sqrt(0.5) * sd_drr
    sd2_sq = 2.0 * sd_rr ** 2 - 0.5 * sd_drr ** 2
    sd2 = np.sqrt(sd2_sq) if sd2_sq > 0 else 0.0
    out[10] = sd1
    out[11] = sd2
    out[12] = sd1 / sd2 if sd2 > 0 else np.nan

    # Shannon entropy over a fixed RR grid.
    hist, _ = np.histogram(rr, bins=np.arange(0.2, 2.01, 0.05))
    p = hist[hist > 0] / hist.sum()
    out[13] = -np.sum(p * np.log(p))
    return out


def build_table():
    """Feature table over the same windows the CNN uses. Returns (X, y, groups)."""
    fs = CLASSIFIER.fs
    rows, labels, groups = [], [], []

    for rec in CLASSIFIER.records:
        sig, rhythm_ann = prep._read_record(rec)
        if sig is None:
            continue
        kept, counts = prep._plan_record(sig, rhythm_ann)
        beat_ann = wfdb.rdann(os.path.join(CLASSIFIER.data_path, rec), 'qrs')
        peaks = np.asarray(beat_ann.sample)

        for start, end, label in kept:
            in_win = peaks[(peaks >= start) & (peaks < end)]
            rows.append(rr_features(in_win, fs))
            labels.append(label)
            groups.append(rec)

        print(f"  {rec}: N={counts['N']:4d}  AFIB={counts['AFIB']:4d}",
              flush=True)

    X = np.vstack(rows)
    y = np.asarray(labels, dtype=np.int32)
    groups = np.asarray(groups)
    print(f"\n{X.shape}  N={int((y == 0).sum())}  AFIB={int((y == 1).sum())}")
    return X, y, groups


def cross_validate(name, model, X, y, groups):
    """Patient-grouped CV, reported exactly like AFIB_CNN_Classifier does."""
    print(f"\n{name}")
    gkf = StratifiedGroupKFold(n_splits=CLASSIFIER.n_folds, shuffle=True,
                               random_state=SEED)
    all_true, all_pred, fold_aucs = [], [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        all_true.extend(y[te]); all_pred.extend(pred)
        auc = (roc_auc_score(y[te], prob)
               if len(np.unique(y[te])) > 1 else float('nan'))
        fold_aucs.append(auc)
        print(f"  Fold {fold}: acc={accuracy_score(y[te], pred):.4f}  "
              f"f1={f1_score(y[te], pred, zero_division=0):.4f}  "
              f"auc={auc:.4f}")

    y_true, y_pred = map(np.array, (all_true, all_pred))
    fold_aucs = np.array(fold_aucs, dtype=float)
    return dict(
        model       = name,
        accuracy    = accuracy_score(y_true, y_pred),
        f1          = f1_score(y_true, y_pred, zero_division=0),
        sensitivity = recall_score(y_true, y_pred, zero_division=0),
        specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        precision   = precision_score(y_true, y_pred, zero_division=0),
        auc         = float(np.nanmean(fold_aucs)),
        # TODO: ddof=1?
        auc_std     = float(np.nanstd(fold_aucs)),
        auc_folds   = fold_aucs,
        confusion   = confusion_matrix(y_true, y_pred),
    )


def main():
    print(f"Building RR feature table from {CLASSIFIER.data_path}")
    X, y, groups = build_table()

    models = [
        ('LogReg', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000,
                                       class_weight='balanced',
                                       random_state=SEED)),
        ])),
        ('HistGradBoost', HistGradientBoostingClassifier(
            random_state=SEED, class_weight='balanced')),
    ]
    results = [cross_validate(n, m, X, y, groups) for n, m in models]

    print(f"\n{'Model':<16}{'Acc':>8}{'F1':>8}{'Sens':>8}{'Spec':>8}"
          f"{'AUC':>8}{'std':>7}")
    for r in results:
        print(f"{r['model']:<16}{r['accuracy']:>8.4f}{r['f1']:>8.4f}"
              f"{r['sensitivity']:>8.4f}{r['specificity']:>8.4f}"
              f"{r['auc']:>8.4f}{r['auc_std']:>7.4f}")
        n_used = int(np.sum(~np.isnan(r['auc_folds'])))
        print(f"{'':<16}{n_used}/{len(r['auc_folds'])} folds in the mean AUC")

    os.makedirs(CLASSIFIER.results_dir, exist_ok=True)
    out = os.path.join(CLASSIFIER.results_dir, 'results_rr_baseline.npy')
    np.save(out, np.array(results, dtype=object), allow_pickle=True)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
