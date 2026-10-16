"""
Train a drone/noise classifier on pipeline4 metadata.

Key changes from train_model3.py (addressing root cause analysis):
  1. Simpler classifier head: 128 -> 1 (single dense layer).
  2. Mean-only pooling (1024-dim) – drops std/max to avoid source artifacts.
  3. Label smoothing 0.1 – prevents score saturation near 0/1.
  4. Cosine annealing LR schedule – smoother than ReduceLROnPlateau.
  5. Source-balanced sampling (alpha=0.5 default) – equal loss per source.
  6. Stronger augmentation: ±6dB gain, SNR=20dB noise, time masking.
  7. Precision-first threshold: --min_precision 0.95 for low false alarms.
  8. Frozen backbone only – no fine-tuning phase initially.
  9. Monitor val_loss for early stopping (more stable than val_pr_auc).
 10. Reports per-split, per-folder, and per-source diagnostics.

Usage:
    python train_model4.py --metadata output4/metadata.csv --audio_root output4
    python train_model4.py --metadata output4/metadata.csv --audio_root output4 --epochs 50 --patience 15
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

SR = 16_000
CLIP_SAMPLES = 32_000  # 2 seconds at 16kHz

# Augmentation parameters (stronger than v3)
SHIFT_MAX = int(CLIP_SAMPLES * 0.10)
GAIN_DB = 6.0          # was 3.0 — more volume variation
NOISE_SNR_DB = 20.0    # was 35.0 — more additive noise
TIME_MASK_MAX = 0.10   # mask up to 10% of the clip

REQUIRED_COLUMNS = {
    "output_file", "source_drone_file", "source_noise_file",
    "folder", "split", "label_binary",
}


def set_reproducibility(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def base_noise_source(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value)
    if not text:
        return None
    return text.split("#", 1)[0]


# ---------------------------------------------------------------------------
# Metadata audit
# ---------------------------------------------------------------------------
def audit_metadata(
    df: pd.DataFrame, audio_root: str,
    allow_leakage: bool, skip_file_check: bool,
) -> Dict[str, object]:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    split_values = set(df["split"].dropna().astype(str).unique())
    expected = {"train", "val", "test"}
    if split_values != expected:
        raise ValueError(
            f"Expected splits {sorted(expected)}, found {sorted(split_values)}"
        )

    df["label_binary"] = df["label_binary"].astype(int)

    source_split_maps = {}
    for source_col, normalize_fn in [
        ("source_drone_file",
         lambda v: None if pd.isna(v) or str(v) == "" else str(v)),
        ("source_noise_file", base_noise_source),
    ]:
        source_splits: Dict[str, set] = defaultdict(set)
        for _, row in df.iterrows():
            source = normalize_fn(row[source_col])
            if source:
                source_splits[source].add(str(row["split"]))
        leaks = {
            s: sorted(sp) for s, sp in source_splits.items()
            if len(sp) > 1
        }
        source_split_maps[source_col] = {
            "unique_sources": len(source_splits),
            "leaks": leaks,
        }

    missing_files = []
    if not skip_file_check:
        root = Path(audio_root)
        for rel in df["output_file"].astype(str):
            if not (root / rel).exists():
                missing_files.append(rel)
                if len(missing_files) >= 20:
                    break

    print("\n" + "=" * 70)
    print("Metadata audit")
    print("=" * 70)
    print(f"Rows: {len(df)}")
    print("\nBinary counts by split:")
    print(pd.crosstab(df["split"], df["label_binary"]).to_string())
    print("\nFolder counts by split:")
    print(pd.crosstab(df["split"], df["folder"]).to_string())
    print(
        "\nDrone sources:",
        source_split_maps["source_drone_file"]["unique_sources"],
        "| leaks:",
        len(source_split_maps["source_drone_file"]["leaks"]),
    )
    print(
        "Noise sources:",
        source_split_maps["source_noise_file"]["unique_sources"],
        "| leaks:",
        len(source_split_maps["source_noise_file"]["leaks"]),
    )

    if missing_files:
        print("\nMissing audio files:")
        for f in missing_files[:20]:
            print(f"  {f}")
        raise FileNotFoundError("Metadata references missing audio files")

    has_leakage = any(info["leaks"] for info in source_split_maps.values())
    if has_leakage:
        print("\nDetected source leakage:")
        for col, info in source_split_maps.items():
            if info["leaks"]:
                print(f"  {col}: {len(info['leaks'])} leaking sources")
        if not allow_leakage:
            raise ValueError("Source leakage detected.")

    print("Audit: OK" if not has_leakage else "Audit: leakage allowed")
    return source_split_maps


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def make_paths(df: pd.DataFrame, audio_root: str) -> np.ndarray:
    root = Path(audio_root)
    return np.array([
        normalize_path(str(root / rel))
        for rel in df["output_file"].astype(str)
    ])


def _load_wav(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(raw, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    audio = audio[:CLIP_SAMPLES]
    pad = CLIP_SAMPLES - tf.shape(audio)[0]
    audio = tf.pad(audio, [[0, pad]])
    audio = tf.ensure_shape(audio, [CLIP_SAMPLES])
    label = tf.cast(label, tf.float32)
    return audio, label


def _load_wav_weighted(
    path: tf.Tensor, label: tf.Tensor, weight: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    audio, label = _load_wav(path, label)
    return audio, label, tf.cast(weight, tf.float32)


def _augment(audio: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    # Random time shift
    shift = tf.random.uniform([], -SHIFT_MAX, SHIFT_MAX + 1, dtype=tf.int32)
    audio = tf.roll(audio, shift=shift, axis=0)

    # Random gain (stronger: ±6 dB)
    gain = tf.pow(10.0, tf.random.uniform([], -GAIN_DB, GAIN_DB) / 20.0)
    audio = audio * gain

    # Additive noise (stronger: SNR=20dB)
    rms = tf.sqrt(tf.reduce_mean(tf.square(audio)) + 1e-10)
    noise_std = rms / tf.pow(10.0, NOISE_SNR_DB / 20.0)
    audio = audio + tf.random.normal([CLIP_SAMPLES], stddev=noise_std)

    # Time masking: zero out a random segment (up to 10%)
    mask_len = tf.random.uniform(
        [], 0, tf.cast(tf.cast(CLIP_SAMPLES, tf.float32) * TIME_MASK_MAX, tf.int32) + 1,
        dtype=tf.int32,
    )
    mask_start = tf.random.uniform(
        [], 0, CLIP_SAMPLES - mask_len + 1, dtype=tf.int32
    )
    # Create mask
    indices = tf.range(CLIP_SAMPLES)
    mask = tf.cast(
        tf.logical_or(indices < mask_start, indices >= mask_start + mask_len),
        tf.float32,
    )
    audio = audio * mask

    audio = tf.clip_by_value(audio, -1.0, 1.0)
    return audio, label


def _augment_weighted(
    audio: tf.Tensor, label: tf.Tensor, weight: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    audio, label = _augment(audio, label)
    return audio, label, weight


def base_audio_dataset(
    paths: np.ndarray, labels: np.ndarray, cache: bool,
    sample_weights: Optional[np.ndarray] = None,
) -> tf.data.Dataset:
    labels = labels.astype(np.float32)
    if sample_weights is None:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(_load_wav, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices(
            (paths, labels, sample_weights.astype(np.float32))
        )
        ds = ds.map(_load_wav_weighted, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    return ds


def source_key_for_training(row: pd.Series) -> str:
    label = int(row["label_binary"])
    if label == 1:
        source = row.get("source_drone_file")
        if pd.isna(source) or str(source) == "":
            return "positive:unknown"
        return f"positive:{source}"
    source = base_noise_source(row.get("source_noise_file"))
    if not source:
        return "negative:unknown"
    return f"negative:{source}"


def source_balanced_weights(df: pd.DataFrame, alpha: float) -> np.ndarray:
    """Down-weight repeated clips from the same source, normalized per class."""
    if alpha <= 0.0:
        return np.ones(len(df), dtype=np.float32)

    clipped_alpha = min(alpha, 1.0)
    work = df.reset_index(drop=True).copy()
    work["_source_key"] = work.apply(source_key_for_training, axis=1)
    group_counts = work["_source_key"].map(work["_source_key"].value_counts())
    weights = np.power(group_counts.astype(float).to_numpy(), -clipped_alpha)

    labels = work["label_binary"].astype(int).to_numpy()
    for label in sorted(set(labels)):
        mask = labels == label
        class_mean = float(np.mean(weights[mask])) if np.any(mask) else 1.0
        if class_mean > 0:
            weights[mask] = weights[mask] / class_mean

    return weights.astype(np.float32)


def build_train_dataset(
    train_df: pd.DataFrame, audio_root: str, batch_size: int,
    seed: int, balanced: bool, augment: bool, cache_audio: bool,
    shuffle_buffer: int, source_balance_alpha: float,
) -> Tuple[tf.data.Dataset, int]:
    labels = train_df["label_binary"].astype(int).to_numpy()
    use_weights = source_balance_alpha > 0.0

    if balanced:
        pos_df = train_df[train_df["label_binary"].astype(int) == 1]
        neg_df = train_df[train_df["label_binary"].astype(int) == 0]
        if len(pos_df) == 0 or len(neg_df) == 0:
            raise ValueError("Balanced training requires both classes")

        pos_ds = base_audio_dataset(
            make_paths(pos_df, audio_root),
            np.ones(len(pos_df), dtype=np.float32),
            cache_audio,
            source_balanced_weights(pos_df, source_balance_alpha) if use_weights else None,
        )
        neg_ds = base_audio_dataset(
            make_paths(neg_df, audio_root),
            np.zeros(len(neg_df), dtype=np.float32),
            cache_audio,
            source_balanced_weights(neg_df, source_balance_alpha) if use_weights else None,
        )

        pos_ds = pos_ds.shuffle(
            min(shuffle_buffer, len(pos_df)), seed=seed,
            reshuffle_each_iteration=True,
        ).repeat()
        neg_ds = neg_ds.shuffle(
            min(shuffle_buffer, len(neg_df)), seed=seed + 1,
            reshuffle_each_iteration=True,
        ).repeat()

        ds = tf.data.Dataset.sample_from_datasets(
            [neg_ds, pos_ds], weights=[0.5, 0.5],
            seed=seed, stop_on_empty_dataset=False,
        )
        steps_per_epoch = max(1, (2 * min(len(pos_df), len(neg_df))) // batch_size)
    else:
        ds = base_audio_dataset(
            make_paths(train_df, audio_root), labels, cache_audio,
            source_balanced_weights(train_df, source_balance_alpha) if use_weights else None,
        )
        ds = ds.shuffle(
            min(shuffle_buffer, len(train_df)), seed=seed,
            reshuffle_each_iteration=True,
        ).repeat()
        steps_per_epoch = max(1, len(train_df) // batch_size)

    if augment:
        if use_weights:
            ds = ds.map(_augment_weighted, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds, steps_per_epoch


def build_eval_dataset(
    split_df: pd.DataFrame, audio_root: str,
    batch_size: int, cache_audio: bool,
) -> tf.data.Dataset:
    labels = split_df["label_binary"].astype(np.float32).to_numpy()
    ds = base_audio_dataset(make_paths(split_df, audio_root), labels, cache_audio)
    return ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# YAMNet Mean Pooling layer (simplified from v3)
# ---------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="DroneAudioV4")
class YAMNetMeanPool(tf.keras.layers.Layer):
    """YAMNet embedding extractor with mean-only pooling.

    Mean pooling is more robust than stats pooling (mean+std+max) because
    std and max capture source-specific artifacts (noise floor, peak transients)
    that don't generalize across recording conditions.
    """

    def __init__(
        self,
        yamnet_handle: str = "https://tfhub.dev/google/yamnet/1",
        **kwargs,
    ):
        kwargs.pop("trainable", None)
        super().__init__(trainable=False, **kwargs)
        self.yamnet_handle = yamnet_handle
        self.output_dim = 1024  # mean-only: 1024 dims

    def build(self, input_shape):
        import tensorflow_hub as hub
        self.yamnet_layer = hub.KerasLayer(
            self.yamnet_handle, trainable=False, name="yamnet_backbone",
        )
        super().build(input_shape)

    def call(self, wav_batch):
        def embed_one(waveform):
            _, embeddings, _ = self.yamnet_layer(waveform)
            embeddings = tf.cast(embeddings, tf.float32)
            embeddings = tf.stop_gradient(embeddings)
            # Mean-only pooling
            return tf.reduce_mean(embeddings, axis=0)

        return tf.map_fn(
            embed_one, wav_batch,
            fn_output_signature=tf.TensorSpec(
                shape=(self.output_dim,), dtype=tf.float32
            ),
        )

    def get_config(self):
        config = super().get_config()
        config.update({"yamnet_handle": self.yamnet_handle})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ---------------------------------------------------------------------------
# Model builder (simplified head)
# ---------------------------------------------------------------------------
def build_model(
    yamnet_handle: str, dropout: float, l2_weight: float,
) -> tf.keras.Model:
    """Build a simple classifier: YAMNet mean pool -> Dense(128) -> Dense(1).

    Single hidden layer is sufficient for fine-tuning a frozen feature extractor
    with ~5K-10K training samples. More layers just add overfitting risk.
    """
    regularizer = tf.keras.regularizers.l2(l2_weight)
    inputs = tf.keras.layers.Input(
        shape=(CLIP_SAMPLES,), dtype=tf.float32, name="waveform"
    )
    x = YAMNetMeanPool(yamnet_handle=yamnet_handle)(inputs)

    # Single hidden layer
    x = tf.keras.layers.Dense(128, kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="drone_score")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# ---------------------------------------------------------------------------
# Cosine annealing LR schedule
# ---------------------------------------------------------------------------
class CosineAnnealingSchedule(tf.keras.callbacks.Callback):
    """Cosine annealing learning rate from initial_lr to min_lr over total epochs."""

    def __init__(self, initial_lr: float, min_lr: float, total_epochs: int):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        progress = epoch / max(1, self.total_epochs - 1)
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )
        self.model.optimizer.learning_rate = lr
        print(f"  Cosine LR: {lr:.2e}")


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------
def compile_model(
    model: tf.keras.Model, learning_rate: float,
    weight_decay: float, label_smoothing: float,
) -> None:
    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
    except Exception:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            tf.keras.metrics.Recall(name="recall", thresholds=0.5),
        ],
        jit_compile=False,
    )


# ---------------------------------------------------------------------------
# LiteRT conversion (simplified from v3)
# ---------------------------------------------------------------------------
def convert_to_litert(
    model: tf.keras.Model, keras_path: Path, quantize: str = "none",
) -> Optional[Path]:
    litert_path = keras_path.with_suffix(".tflite")
    print("\n" + "=" * 70)
    print(f"LiteRT conversion ({quantize} precision)")
    print("=" * 70)

    try:
        @tf.function
        def run_model(inputs):
            return model(inputs)

        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
        )

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False

        if quantize == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantize == "dynamic_range":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()
        litert_path.write_bytes(tflite_model)
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"  Saved {size_mb:.1f} MB → {litert_path}")
        return litert_path

    except Exception as exc:
        print(f"  WARNING: LiteRT conversion failed — {exc}")
        return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_training_history(
    history: tf.keras.callbacks.History, output_dir: Path,
) -> None:
    metrics = ["loss", "accuracy", "roc_auc", "pr_auc", "precision", "recall"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, metric in zip(axes.flat, metrics):
        if metric not in history.history:
            continue
        ax.plot(history.history[metric], label="train")
        val_key = f"val_{metric}"
        if val_key in history.history:
            ax.plot(history.history[val_key], label="val")
        ax.set_title(metric)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_history4.png", dpi=150)
    plt.close(fig)


def plot_score_histogram(
    y_true: np.ndarray, y_scores: np.ndarray,
    threshold: float, output_dir: Path, split_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_scores[y_true == 0], bins=50, alpha=0.65, label="noise", density=True)
    ax.hist(y_scores[y_true == 1], bins=50, alpha=0.65, label="drone", density=True)
    ax.axvline(threshold, color="black", linestyle="--", label=f"t={threshold:.3f}")
    ax.axvline(0.5, color="orange", linestyle=":", label="0.5")
    ax.set_xlabel("Predicted drone score")
    ax.set_ylabel("Density")
    ax.set_title(f"{split_name} score distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"score_hist_{split_name}.png", dpi=150)
    plt.close(fig)


def plot_pr_curve(
    precision: np.ndarray, recall: np.ndarray,
    thresholds: np.ndarray, best_threshold: float,
    output_dir: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(recall, precision, linewidth=2)
    if len(thresholds):
        idx = int(np.argmin(np.abs(thresholds - best_threshold)))
        ax1.scatter(recall[idx], precision[idx], s=160, marker="*", color="red")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-recall curve")
    ax1.grid(alpha=0.3)

    if len(thresholds):
        p, r = precision[:-1], recall[:-1]
        f1 = 2 * p * r / (p + r + 1e-8)
        ax2.plot(thresholds, p, label="precision", linewidth=2)
        ax2.plot(thresholds, r, label="recall", linewidth=2)
        ax2.plot(thresholds, f1, label="f1", linewidth=2)
        ax2.axvline(best_threshold, linestyle="--", color="black", label="selected")
        ax2.axvline(0.5, linestyle=":", color="orange", label="0.5")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("Metrics vs threshold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "pr_curve4.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scoring and threshold selection
# ---------------------------------------------------------------------------
def collect_scores(model: tf.keras.Model, ds: tf.data.Dataset) -> np.ndarray:
    return model.predict(ds, verbose=1).reshape(-1)


def select_threshold(
    y_true: np.ndarray, y_scores: np.ndarray,
    strategy: str, min_precision: Optional[float],
    min_recall: Optional[float],
) -> Tuple[float, Dict[str, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    if len(thresholds) == 0:
        return (0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                (precision, recall, thresholds))

    p, r = precision[:-1], recall[:-1]
    f1 = 2 * p * r / (p + r + 1e-8)

    valid = np.ones_like(thresholds, dtype=bool)
    if min_precision is not None:
        valid &= p >= min_precision
    if min_recall is not None:
        valid &= r >= min_recall
    if not np.any(valid):
        # Relax: find the closest to min_precision
        if min_precision is not None:
            # Find threshold where precision is highest while keeping recall reasonable
            valid = p >= (min_precision * 0.9)  # 90% of target
            if not np.any(valid):
                valid = np.ones_like(thresholds, dtype=bool)

    candidate = np.where(valid)[0]
    if strategy == "balance_pr":
        best_idx = max(
            candidate,
            key=lambda i: (-abs(float(p[i] - r[i])), float(f1[i])),
        )
    else:
        best_idx = max(candidate, key=lambda i: float(f1[i]))

    threshold = float(thresholds[best_idx])
    metrics = {
        "precision": float(p[best_idx]),
        "recall": float(r[best_idx]),
        "f1": float(f1[best_idx]),
        "threshold": threshold,
    }
    return threshold, metrics, (precision, recall, thresholds)


def binary_metrics_at_threshold(
    y_true: np.ndarray, y_scores: np.ndarray, threshold: float,
) -> Dict[str, float]:
    y_pred = (y_scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "threshold": float(threshold),
    }


def evaluate_scores(
    y_true: np.ndarray, y_scores: np.ndarray,
    threshold: float, split_name: str,
    output_dir: Path, suffix: str,
) -> Dict[str, object]:
    y_pred = (y_scores >= threshold).astype(int)
    print("\n" + "=" * 70)
    print(f"{split_name} evaluation @ threshold={threshold:.4f}")
    print("=" * 70)
    print(
        classification_report(
            y_true, y_pred, target_names=["noise", "drone"], digits=4
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Noise", "Pred Drone"])
    ax.set_yticklabels(["True Noise", "True Drone"])
    ax.set_title(f"{split_name} confusion (t={threshold:.3f})")
    for row in range(2):
        for col in range(2):
            ax.text(
                col, row, str(cm[row, col]), ha="center", va="center",
                fontsize=15,
                color="white" if cm[row, col] > cm.max() / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_{split_name}_{suffix}.png", dpi=150)
    plt.close(fig)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics = {
        "threshold": float(threshold),
        "precision": float(prec), "recall": float(rec),
        "f1": float(f1), "accuracy": float(np.mean(y_true == y_pred)),
        "confusion_matrix": cm.tolist(),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


# ---------------------------------------------------------------------------
# Per-folder and per-source diagnostics
# ---------------------------------------------------------------------------
def write_folder_diagnostics(
    split_df: pd.DataFrame, y_scores: np.ndarray,
    threshold: float, output_dir: Path, split_name: str,
) -> None:
    rows = []
    y_pred = (y_scores >= threshold).astype(int)
    y_true = split_df["label_binary"].astype(int).to_numpy()

    for folder, indices in split_df.groupby("folder").groups.items():
        idx = np.array(list(indices), dtype=int)
        ft, fp, fs = y_true[idx], y_pred[idx], y_scores[idx]
        rows.append({
            "split": split_name, "folder": folder,
            "count": len(idx),
            "accuracy": float(np.mean(ft == fp)),
            "mean_score": float(np.mean(fs)),
            "p10": float(np.quantile(fs, 0.10)),
            "p50": float(np.quantile(fs, 0.50)),
            "p90": float(np.quantile(fs, 0.90)),
        })

    pd.DataFrame(rows).sort_values(["split", "folder"]).to_csv(
        output_dir / f"folder_diag_{split_name}.csv", index=False
    )


def write_source_diagnostics(
    split_df: pd.DataFrame, y_scores: np.ndarray,
    threshold: float, output_dir: Path, split_name: str,
) -> pd.DataFrame:
    scored = split_df.copy().reset_index(drop=True)
    scored["score"] = y_scores
    scored["pred"] = (y_scores >= threshold).astype(int)
    rows = []

    # Drone sources
    pos = scored[scored["label_binary"].astype(int) == 1]
    for src, grp in pos.groupby("source_drone_file", dropna=True):
        if not src:
            continue
        sc, pr = grp["score"].to_numpy(), grp["pred"].to_numpy()
        rows.append({
            "split": split_name, "type": "drone", "source": src,
            "count": len(grp), "accuracy": float(np.mean(pr == 1)),
            "error_count": int(np.sum(pr == 0)),
            "mean_score": float(np.mean(sc)),
        })

    # Noise sources
    neg = scored[scored["label_binary"].astype(int) == 0].copy()
    neg["base_src"] = neg["source_noise_file"].astype(str).str.split("#", n=1).str[0]
    for src, grp in neg.groupby("base_src", dropna=True):
        if not src or src == "nan":
            continue
        sc, pr = grp["score"].to_numpy(), grp["pred"].to_numpy()
        rows.append({
            "split": split_name, "type": "noise", "source": src,
            "count": len(grp), "accuracy": float(np.mean(pr == 0)),
            "error_count": int(np.sum(pr == 1)),
            "mean_score": float(np.mean(sc)),
        })

    diag = pd.DataFrame(rows)
    if not diag.empty:
        diag = diag.sort_values(["type", "accuracy"], ascending=[True, True])
    diag.to_csv(output_dir / f"source_diag_{split_name}.csv", index=False)
    return diag


def write_predictions(
    split_df: pd.DataFrame, y_scores: np.ndarray,
    thresholds: Dict[str, float], output_dir: Path, split_name: str,
) -> None:
    scored = split_df.copy().reset_index(drop=True)
    scored["score"] = y_scores
    labels = scored["label_binary"].astype(int).to_numpy()
    for name, t in thresholds.items():
        scored[f"pred_{name}"] = (y_scores >= t).astype(int)
        scored[f"error_{name}"] = scored[f"pred_{name}"].astype(int) != labels

    cols = [c for c in scored.columns if c not in {"drone_rms", "noise_rms", "sample_rate"}]
    scored[cols].to_csv(output_dir / f"predictions_{split_name}.csv", index=False)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train drone classifier v4")
    parser.add_argument("--metadata", default="output4/metadata.csv")
    parser.add_argument("--audio_root", default="output4")
    parser.add_argument("--output_dir", default="model4_artifacts")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--l2_weight", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_buffer", type=int, default=10_000)
    parser.add_argument("--no_balanced_train", action="store_true")
    parser.add_argument(
        "--source_balance_alpha", type=float, default=0.5,
        help="0=disabled, 1=full source balance, 0.5=moderate",
    )
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--cache_train_audio", action="store_true")
    parser.add_argument("--no_cache_eval", action="store_true")
    parser.add_argument("--allow_leakage", action="store_true")
    parser.add_argument("--skip_file_check", action="store_true")
    parser.add_argument(
        "--threshold_strategy",
        choices=["fixed_0_5", "max_f1", "balance_pr"],
        default="max_f1",
    )
    parser.add_argument("--min_precision", type=float, default=0.95)
    parser.add_argument("--min_recall", type=float, default=None)
    parser.add_argument(
        "--yamnet_handle",
        default="https://tfhub.dev/google/yamnet/1",
    )
    parser.add_argument(
        "--no_litert", action="store_true",
        help="Skip LiteRT conversion after training.",
    )
    parser.add_argument(
        "--litert_quantize",
        choices=["none", "float16", "dynamic_range"], default="none",
    )
    args = parser.parse_args()

    set_reproducibility(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = (
        Path(args.model_path) if args.model_path else output_dir / "best_model4.keras"
    )

    print("=" * 70)
    print("Drone Audio Classifier v4")
    print("=" * 70)
    print(f"Metadata:        {args.metadata}")
    print(f"Audio root:      {args.audio_root}")
    print(f"Output:          {output_dir}")
    print(f"Model:           {model_path}")
    print(f"Balanced:        {not args.no_balanced_train}")
    print(f"Source balance:  alpha={args.source_balance_alpha}")
    print(f"Augmentation:    {not args.no_augment}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Dropout:         {args.dropout}")
    print(f"LR:              {args.learning_rate} -> {args.min_lr} (cosine)")
    print(f"Threshold:       {args.threshold_strategy} (min_P={args.min_precision})")
    print(f"Monitor:         val_loss (early stopping)")

    df = pd.read_csv(args.metadata)
    df = df.reset_index(drop=True)
    audit_metadata(df, args.audio_root, args.allow_leakage, args.skip_file_check)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_ds, steps_per_epoch = build_train_dataset(
        train_df, args.audio_root, args.batch_size, args.seed,
        balanced=not args.no_balanced_train,
        augment=not args.no_augment,
        cache_audio=args.cache_train_audio,
        shuffle_buffer=args.shuffle_buffer,
        source_balance_alpha=args.source_balance_alpha,
    )
    val_ds = build_eval_dataset(
        val_df, args.audio_root, args.batch_size,
        cache_audio=not args.no_cache_eval,
    )
    test_ds = build_eval_dataset(
        test_df, args.audio_root, args.batch_size,
        cache_audio=not args.no_cache_eval,
    )

    print(f"\nTrain: {len(train_df)} rows | steps/epoch: {steps_per_epoch}")
    print(f"Val:   {len(val_df)} rows")
    print(f"Test:  {len(test_df)} rows")

    model = build_model(
        yamnet_handle=args.yamnet_handle,
        dropout=args.dropout,
        l2_weight=args.l2_weight,
    )
    compile_model(
        model, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        CosineAnnealingSchedule(
            initial_lr=args.learning_rate,
            min_lr=args.min_lr,
            total_epochs=args.epochs,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log4.csv")),
    ]

    print("\n" + "=" * 70)
    print("Training (frozen YAMNet backbone)")
    print("=" * 70)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
    )
    plot_training_history(history, output_dir)

    # LiteRT conversion
    litert_path: Optional[Path] = None
    if not args.no_litert:
        litert_path = convert_to_litert(
            model, model_path, quantize=args.litert_quantize
        )

    # --- Evaluation ---
    print("\n" + "=" * 70)
    print("Threshold selection & evaluation")
    print("=" * 70)
    y_val = val_df["label_binary"].astype(int).to_numpy()
    y_test = test_df["label_binary"].astype(int).to_numpy()
    val_scores = collect_scores(model, val_ds)
    test_scores = collect_scores(model, test_ds)

    # Score distribution stats
    print(f"\nVal score stats: drone mean={val_scores[y_val==1].mean():.4f} "
          f"noise mean={val_scores[y_val==0].mean():.4f}")
    print(f"Test score stats: drone mean={test_scores[y_test==1].mean():.4f} "
          f"noise mean={test_scores[y_test==0].mean():.4f}")

    strategy = args.threshold_strategy if args.threshold_strategy != "fixed_0_5" else "max_f1"
    candidate_t, candidate_m, pr_data = select_threshold(
        y_val, val_scores, strategy=strategy,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
    )
    precision, recall, thresholds = pr_data

    if args.threshold_strategy == "fixed_0_5":
        selected_t = 0.5
        selected_m = binary_metrics_at_threshold(y_val, val_scores, 0.5)
    else:
        selected_t = candidate_t
        selected_m = candidate_m

    plot_pr_curve(precision, recall, thresholds, selected_t, output_dir)

    print(f"\nSelected threshold: {selected_t:.4f} | "
          f"P={selected_m['precision']:.4f} R={selected_m['recall']:.4f} "
          f"F1={selected_m['f1']:.4f}")
    if args.threshold_strategy == "fixed_0_5":
        print(f"Candidate threshold: {candidate_t:.4f} | "
              f"P={candidate_m['precision']:.4f} R={candidate_m['recall']:.4f} "
              f"F1={candidate_m['f1']:.4f}")

    val_default = evaluate_scores(y_val, val_scores, 0.5, "val", output_dir, "default")
    val_selected = evaluate_scores(y_val, val_scores, selected_t, "val", output_dir, "selected")
    test_default = evaluate_scores(y_test, test_scores, 0.5, "test", output_dir, "default")
    test_selected = evaluate_scores(y_test, test_scores, selected_t, "test", output_dir, "selected")

    plot_score_histogram(y_val, val_scores, selected_t, output_dir, "val")
    plot_score_histogram(y_test, test_scores, selected_t, output_dir, "test")

    pred_thresholds = {"default": 0.5, "selected": selected_t}
    write_predictions(val_df, val_scores, pred_thresholds, output_dir, "val")
    write_predictions(test_df, test_scores, pred_thresholds, output_dir, "test")
    write_folder_diagnostics(val_df, val_scores, selected_t, output_dir, "val")
    write_folder_diagnostics(test_df, test_scores, selected_t, output_dir, "test")
    val_diag = write_source_diagnostics(val_df, val_scores, selected_t, output_dir, "val")
    test_diag = write_source_diagnostics(test_df, test_scores, selected_t, output_dir, "test")

    # Save results
    results = {
        "threshold": selected_t,
        "strategy": args.threshold_strategy,
        "min_precision": args.min_precision,
        "val_default": val_default,
        "val_selected": val_selected,
        "test_default": test_default,
        "test_selected": test_selected,
        "candidate_threshold": candidate_t,
        "candidate_metrics": candidate_m,
        "model_path": str(model_path),
        "litert_path": str(litert_path) if litert_path else None,
    }
    save_json(output_dir / "results4.json", results)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model:     {model_path}")
    if litert_path:
        print(f"LiteRT:    {litert_path}")
    print(f"Threshold: {selected_t:.4f}")
    print(f"Artifacts: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
