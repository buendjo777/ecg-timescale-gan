"""Train and cross-validate the AF classifier on CWT scalograms.
An ImageNet backbone (ResNet50 or EfficientNetB0) with a small pooling head is fine-tuned in two phases:

  Phase 1: the backbone is frozen and only the GAP + Dense(1) head trains.
  Phase 2: the top CLASSIFIER.n_unfreeze backbone layers (except BatchNorm)
           are unfrozen and training continues at a much lower learning rate.

Evaluation is 5-fold cross-validation grouped by record (StratifiedGroupKFold).
The validation split inside each training fold is grouped the same way.

Usage:
    python Classifier/AFIB_CNN_Classifier.py
    python Classifier/AFIB_CNN_Classifier.py --exclude-pure-afib --tag drop_pure
    CUDA_VISIBLE_DEVICES=N python Classifier/AFIB_CNN_Classifier.py

Reads the arrays written by Classifier_Preprocessor.py and writes the
comparison plot and raw results to CLASSIFIER.results_dir.
"""

import argparse
import math
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.applications import resnet as resnet_pp
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score)

from cwt_utils import make_scales
from config import CLASSIFIER, WAVELET, SEED, PURE_AFIB_RECORDS


# Each backbone with the preprocessing its ImageNet weights expect.
# EfficientNetB0 normalises its input internally.
BACKBONES = (('ResNet50',       ResNet50,       resnet_pp.preprocess_input), ('EfficientNetB0', EfficientNetB0, None),)

def _to_display_range(t):
    """Rescale each sample to [0, 255] over all three channels together."""
    lo = tf.reduce_min(t, axis=[1, 2, 3], keepdims=True)
    hi = tf.reduce_max(t, axis=[1, 2, 3], keepdims=True)
    return (t - lo) / tf.maximum(hi - lo, 1e-6) * 255.0

def build_model(input_shape, backbone_fn, name, preprocess_fn=None):
    """Builds the ImageNet backbone with a global-pooling head on the |W| representation.
    Returns (model, base); unfreeze_top works on the base model in phase 2."""
    inputs = layers.Input(shape=input_shape)   # (64, 2500, 2)

    mag = layers.Lambda(lambda t: tf.sqrt(tf.square(t[..., 0:1]) + tf.square(t[..., 1:2])), name='abs_W',)(inputs)
    x = layers.Concatenate(axis=-1, name='abs_3ch')([mag, mag, mag])

    # TODO: Explore using Anti-aliasing instead of simply resizing (diff input)
    resized = layers.Resizing(64, 512)(x)
    scaled = layers.Lambda(_to_display_range, name='to_display_range')(resized)
    if preprocess_fn is not None:
        scaled = layers.Lambda(preprocess_fn, name='backbone_preprocess')(scaled)

    base = backbone_fn(weights='imagenet', include_top=False, input_tensor=scaled)
    base.trainable = False                      # unfreeze_top() in phase 2

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name=name), base


def unfreeze_top(base, n):
    """Makes the last n layers of the backbone trainable, except for BatchNorm."""
    base.trainable = True
    for layer in base.layers[:-n]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def class_weights(y):
    """Calculates balanced class weights, n_samples / (2 * n_class)."""
    total, n_neg, n_pos = len(y), np.sum(y == 0), np.sum(y == 1)
    return {0: total / (2 * max(n_neg, 1)), 1: total / (2 * max(n_pos, 1))}


class _RowSubset:
    """Read-only view of some rows of a large array, without copying."""

    def __init__(self, base, idx):
        self.base = base
        self.idx = np.asarray(idx)
        self.shape = (len(self.idx),) + tuple(base.shape[1:])
        self.dtype = base.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, i):
        return self.base[self.idx[i]]


class _ConcatRows:
    """Read-only view of two arrays stacked by rows, without copying."""

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = len(a)
        self.shape = (len(a) + len(b),) + tuple(a.shape[1:])
        self.dtype = a.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, i):
        return self.a[i] if i < self.n_a else self.b[i - self.n_a]


def inner_split(ytr, groups_tr, n_splits=5):
    """Splits a training fold into train/validation indices by record.
    Splits whose validation part lacks one of the classes are skipped."""
    placeholder = np.zeros(len(ytr))
    n_splits = min(n_splits, len(np.unique(groups_tr)))
    if n_splits < 2:
        raise RuntimeError("No grouped validation split: the training fold has one patient.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr_idx, val_idx in splitter.split(placeholder, ytr, groups_tr):
        if len(np.unique(ytr[val_idx])) > 1:
            return tr_idx, val_idx
    raise RuntimeError("The training fold is too record-poor to validate on.")


def fit_phase(model, Xtr, ytr, groups_tr, lr, epochs, cw, n_real=None):
    """Compiles and trains the model."""
    model.compile(optimizer=optimizers.Adam(lr), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    callbacks = [
        EarlyStopping(monitor='val_auc', mode='max', patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=2, verbose=1),
    ]

    if n_real is None or n_real >= len(ytr):
        tr_idx, val_idx = inner_split(ytr, groups_tr)
    else:
        tr_idx, val_idx = inner_split(ytr[:n_real], groups_tr[:n_real])
        tr_idx = np.concatenate([tr_idx, np.arange(n_real, len(ytr))])

    def make_ds(indices, shuffle):
        """Return a dataset that streams (x, y) pairs from Xtr/ytr by index."""
        def gen():
            order = indices.copy()
            if shuffle:
                np.random.shuffle(order)
            for i in order:
                yield Xtr[i], ytr[i]
        ds = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(Xtr.shape[1:], tf.float32), tf.TensorSpec((), tf.int32)),)
        return (ds.batch(CLASSIFIER.batch_size).repeat().prefetch(tf.data.AUTOTUNE))

    train_ds = make_ds(tr_idx, shuffle=True)
    val_ds   = make_ds(val_idx, shuffle=False)

    steps_per_epoch  = math.ceil(len(tr_idx)  / CLASSIFIER.batch_size)
    validation_steps = math.ceil(len(val_idx) / CLASSIFIER.batch_size)

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, class_weight=cw, callbacks=callbacks, verbose=2)


def evaluate_fold(backbone_fn, name, Xtr, ytr, groups_tr, Xte, preprocess_fn=None, n_real=None):
    """Trains a fresh model on one fold and return its test probabilities."""
    model, base = build_model(Xtr.shape[1:], backbone_fn, name, preprocess_fn=preprocess_fn)
    # TODO: Try doing it again without reweighting after adding synthetic data, keep real weights only.
    cw = class_weights(ytr)

    fit_phase(model, Xtr, ytr, groups_tr, CLASSIFIER.phase1_lr, CLASSIFIER.phase1_epochs, cw, n_real=n_real)
    unfreeze_top(base, CLASSIFIER.n_unfreeze)
    fit_phase(model, Xtr, ytr, groups_tr, CLASSIFIER.phase2_lr, CLASSIFIER.phase2_epochs, cw, n_real=n_real)

    def gen_pred():
        for x in Xte:
            yield x
    test_ds = tf.data.Dataset.from_generator(gen_pred, output_signature=tf.TensorSpec(Xte.shape[1:], tf.float32),).batch(CLASSIFIER.batch_size).prefetch(tf.data.AUTOTUNE)
    y_prob = model.predict(test_ds, verbose=0).flatten()
    tf.keras.backend.clear_session()
    return y_prob


def cross_validate(backbone_fn, name, X, y, groups, preprocess_fn=None, synth=None):
    """Runs grouped, stratified cross-validation of one backbone."""
    print(f"\n{name}")
    # With fewer patients than folds, use one patient per test fold.
    n_splits = min(CLASSIFIER.n_folds, len(np.unique(groups)))
    if n_splits < CLASSIFIER.n_folds:
        print(f"  {len(np.unique(groups))} patients: {n_splits}-fold CV", flush=True)
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    all_true, all_pred, fold_aucs = [], [], []
    placeholder = np.zeros(len(y))

    for fold, (tr, te) in enumerate(gkf.split(placeholder, y, groups), 1):
        ytr_local, yte_local = y[tr], y[te]
        print(f"  Fold {fold} | "
              f"train: N={int((ytr_local==0).sum())} AFIB={int((ytr_local==1).sum())} | "
              f"test: N={int((yte_local==0).sum())} AFIB={int((yte_local==1).sum())} | "
              f"records train={len(np.unique(groups[tr]))} test={len(np.unique(groups[te]))}")

        Xtr, ytr, gtr, n_real = _RowSubset(X, tr), y[tr], groups[tr], None
        if synth is not None:
            Xs, ys, gs = synth
            n_real = len(ytr)
            Xtr = _ConcatRows(Xtr, Xs)
            ytr = np.concatenate([ytr, ys])
            gtr = np.concatenate([gtr, gs])
            print(f"    + {len(ys)} synthetic rows into train "
                  f"(real {n_real}); test fold untouched")

        y_prob = evaluate_fold(backbone_fn, name, Xtr, ytr, gtr, _RowSubset(X, te), preprocess_fn=preprocess_fn, n_real=n_real)
        # TODO: Tune the threshold on the validation split internally within folds, using something like:
        # def pick_threshold(y_val, p_val):
        #    fpr, tpr, thr = roc_curve(y_val, p_val)
        #    return float(thr[np.argmax(tpr - fpr)])

        y_pred = (y_prob >= 0.5).astype(int)
        all_true.extend(y[te]); all_pred.extend(y_pred)
        fold_auc = (roc_auc_score(y[te], y_prob)
                    if len(np.unique(y[te])) > 1 else float('nan'))
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold}: "
              f"acc={accuracy_score(y[te], y_pred):.4f}  "
              f"f1={f1_score(y[te], y_pred, zero_division=0):.4f}  "
              f"auc={fold_auc:.4f}")

    y_true, y_pred = map(np.array, (all_true, all_pred))
    fold_aucs = np.array(fold_aucs, dtype=float)
    return dict(
        model       = name,
        accuracy    = accuracy_score(y_true, y_pred),
        f1          = f1_score(y_true, y_pred, zero_division=0),
        # Sensitivity is the AFIB recall and specificity the N recall.
        sensitivity = recall_score(y_true, y_pred, zero_division=0),
        specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        precision   = precision_score(y_true, y_pred, zero_division=0),
        auc         = float(np.nanmean(fold_aucs)),
        # TODO: ddof=1
        auc_std     = float(np.nanstd(fold_aucs)),
        auc_folds   = fold_aucs,
        confusion   = confusion_matrix(y_true, y_pred),
    )


def plot_results(results, tag='abs'):
    """Saves a bar chart of the main metrics, one group of bars per backbone."""
    metrics = ['accuracy', 'f1', 'sensitivity', 'specificity', 'auc']
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(results):
        ax.bar(x + i * width, [r[m] for m in metrics], width, label=r['model'])

    # TODO: I cant get the ticks to be centered
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    ax.set_title(f'Backbone comparison: record-grouped CV  (tag={tag})')
    plt.tight_layout()
    out = os.path.join(CLASSIFIER.results_dir, f'comparison_{tag}.png')
    plt.savefig(out, dpi=300)
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--exclude-pure-afib', action='store_true',
                   help="drop the 100%% AFIB records "
                        f"({', '.join(PURE_AFIB_RECORDS)}) right after loading")
    p.add_argument('--synth-dir', default=None,
                   help="directory with synthetic_scalograms.npy to add to "
                        "every training fold")
    p.add_argument('--synth-n', type=int, default=None,
                   help="use only the first N synthetic rows")
    p.add_argument('--n-patients', type=int, default=None,
                   help="keep only this many patients")
    p.add_argument('--patient-seed', type=int, default=SEED,
                   help="seed for the patient draw")
    p.add_argument('--backbone', default=None,
                   choices=['ResNet50', 'EfficientNetB0'],
                   help="train only this backbone")
    p.add_argument('--tag', default='abs',
                   help="suffix for comparison_<tag>.png and results_<tag>.npy")
    args = p.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    os.makedirs(CLASSIFIER.results_dir, exist_ok=True)

    print("Loading X_scalograms.npy", flush=True)
    X = np.load(os.path.join(CLASSIFIER.output_dir, 'X_scalograms.npy'), mmap_mode='r')
    print(f"  X shape={X.shape}  dtype={X.dtype}  nbytes={X.nbytes/1e9:.1f} GB", flush=True)
    print("Loading y_labels.npy and record_ids.npy", flush=True)
    y = np.load(os.path.join(CLASSIFIER.output_dir, 'y_labels.npy'))
    groups = np.load(os.path.join(CLASSIFIER.output_dir, 'record_ids.npy'))

    # Check that the stored arrays match config.py.
    scales = np.load(os.path.join(CLASSIFIER.output_dir, 'scales.npy'))
    expected_scales = make_scales(CLASSIFIER.fs, WAVELET.f_min, WAVELET.f_max, WAVELET.n_scales, WAVELET.fc)
    matches = (X.shape[1:] == (WAVELET.n_scales, CLASSIFIER.seg_length, 2)
               and len(y) == len(X) == len(groups)
               and scales.shape == expected_scales.shape
               and np.allclose(scales, expected_scales))
    if not matches:
        raise SystemExit(f"The arrays in {CLASSIFIER.output_dir} do not match config.py.")

    if args.exclude_pure_afib:
        mask = ~np.isin(groups, PURE_AFIB_RECORDS)
        X, y, groups = _RowSubset(X, np.flatnonzero(mask)), y[mask], groups[mask]
        print(f"Excluded 100%-AFIB records ({', '.join(PURE_AFIB_RECORDS)}): "
              f"records={len(np.unique(groups))}",
              flush=True)

    print(f"X={X.shape}  N={np.sum(y==0)}  AFIB={np.sum(y==1)}  "
          f"records={len(np.unique(groups))}  tag={args.tag}",
          flush=True)

    if args.n_patients:
        recs = np.unique(groups)
        if args.n_patients > len(recs):
            raise SystemExit(
                f"--n-patients {args.n_patients} > {len(recs)} available")
        rng = np.random.default_rng(args.patient_seed)
        keep = rng.choice(recs, size=args.n_patients, replace=False)
        for _ in range(200):
            m = np.isin(groups, keep)
            if len(np.unique(y[m])) > 1:
                break
            keep = rng.choice(recs, size=args.n_patients, replace=False)
        else:
            raise SystemExit("no subset of that size holds both classes")
        mask = np.isin(groups, keep)
        idx = np.flatnonzero(mask)
        X, y, groups = _RowSubset(X, idx), y[idx], groups[idx]
        print(f"n-patients={args.n_patients} seed={args.patient_seed} "
              f"records={sorted(keep.tolist())}", flush=True)
        print(f"  subset: {X.shape}  N={int((y==0).sum())} "
              f"AFIB={int((y==1).sum())}", flush=True)

    synth = None
    if args.synth_dir:
        Xs = np.load(os.path.join(args.synth_dir, 'synthetic_scalograms.npy'), mmap_mode='r')
        if args.synth_n:
            Xs = Xs[:args.synth_n]
        if Xs.shape[1:] != X.shape[1:]:
            raise SystemExit(
                f"synthetic shape {Xs.shape[1:]} != real {X.shape[1:]}")
        # The generator only produces AFIB windows.
        ys = np.ones(len(Xs), dtype=y.dtype)
        gs = np.full(len(Xs), 'synth')
        synth = (Xs, ys, gs)
        print(f"synthetic: {Xs.shape} from {args.synth_dir}", flush=True)

    results = [cross_validate(fn, label, X, y, groups, preprocess_fn=pp, synth=synth)
        for label, fn, pp in BACKBONES
        if args.backbone is None or label == args.backbone
    ]

    # Summary table.
    print(f"\n{'Model':<18}{'Acc':>8}{'F1':>8}{'Sens':>8}{'Spec':>8}"
          f"{'AUC':>8}{'std':>7}")
    for r in results:
        print(f"{r['model']:<18}{r['accuracy']:>8.4f}{r['f1']:>8.4f}"
              f"{r['sensitivity']:>8.4f}{r['specificity']:>8.4f}"
              f"{r['auc']:>8.4f}{r['auc_std']:>7.4f}")
        folds = '  '.join(f"{a:.3f}" for a in r['auc_folds'])
        n_used = int(np.sum(~np.isnan(r['auc_folds'])))
        print(f"{'':<18}per-fold AUC: {folds}  "
              f"({n_used}/{len(r['auc_folds'])} folds in the mean)")

    plot_results(results, tag=args.tag)
    out_npy = os.path.join(CLASSIFIER.results_dir, f'results_{args.tag}.npy')
    np.save(out_npy, results, allow_pickle=True)
    print(f"Saved {out_npy}")


if __name__ == "__main__":
    main()
