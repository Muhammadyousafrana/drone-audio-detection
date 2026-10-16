"""
Leakage-safe drone audio preprocessing pipeline v4.

Fixes over pipeline3.py (addressing root cause analysis):
  1. Zero overlap – no near-duplicate clips from the same source.
  2. Per-source clip cap – prevents long recordings from dominating.
  3. Short clip combining – groups sub-2s files by filename pattern
     and concatenates them before clipping.
  4. Automated noise screening – flags noise files with drone-like
     spectral content (strong low-frequency humming energy).
  5. Drone audibility check after SNR mixing – rejects mixes where
     the drone signal becomes inaudible.
  6. Dynamic balanced splits – works with any number of sources,
     enforces 1:1 class balance per split.
  7. Sub-segment silence detection – rejects clips with any silent
     sub-segment.
  8. 2 SNR difficulty levels (easy + hard) instead of 3, plus clean.

Usage:
    python pipeline4.py --output_dir output4
    python pipeline4.py --output_dir output4 --max_clips_per_source 50
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import iirfilter, sosfilt
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline4")

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".aac", ".wma",".opus"
}

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Two difficulty levels: easy (clearly audible) and hard (challenging but detectable)
SNR_RANGES = {
    "easy": (10.0, 20.0),
    "hard": (2.0, 8.0),
}

FOLDER_TO_BINARY = {
    "drone_clean": 1,
    "easy": 1,
    "hard": 1,
    "noise_clean": 0,
}

# ---------------------------------------------------------------------------
# Filename-pattern grouping for short drone clips
# ---------------------------------------------------------------------------
# Patterns to detect groups of short clips from the same drone model.
# Examples:
#   B_S2_D1_take1.wav, B_S2_D1_take2.wav  -> group "B_S2_D1"
#   mixed_15-bebop.wav, mixed_16-bebop.wav -> group "bebop"
#   DJI_Mini3_clip1.wav, DJI_Mini3_clip2.wav -> group "DJI_Mini3"

_GROUP_PATTERNS = [
    # Pattern: prefix like B_S2_D1 (letter_Sdigit_Ddigit)
    re.compile(r"^([A-Z]_S\d+_D\d+)", re.IGNORECASE),
    # Pattern: mixed_NN-dronename
    re.compile(r"^mixed_\d+-(.+?)(?:_\d+)?$", re.IGNORECASE),
    # Pattern: DroneName_ModelNumber (e.g. DJI_Mavic_Mini2)
    re.compile(r"^((?:DJI|Autel|Syma|Yuneec|Holystone|Hover|Swellpro|UDI|Hasakee|PhenoBee|David)_[A-Za-z0-9_]+?)_\d+", re.IGNORECASE),
]


def group_key_for_combining(filepath: Path) -> Optional[str]:
    """Extract a group key from filename for combining short clips.

    Returns None if no known pattern matches (file won't be combined).
    """
    stem = filepath.stem
    for pattern in _GROUP_PATTERNS:
        match = pattern.match(stem)
        if match:
            return match.group(1).lower().replace(" ", "_")
    return None


@dataclass
class AudioClip:
    audio: np.ndarray
    source: str
    rms: float


@dataclass
class PipelineConfig:
    drone_dir: str = "drone"
    noise_dir: str = "noise"
    output_dir: str = "output4"
    sample_rate: int = 16_000
    clip_duration: float = 2.0
    seed: int = 42
    target_rms: float = 0.1
    drone_min_rms: float = 0.01
    noise_min_rms: float = 0.002
    noise_max_rms: float = 0.3
    noise_peak_ratio: float = 3.0
    negative_ratio: float = 1.0
    max_clips_per_source: int = 30
    overwrite: bool = False
    apply_noise_eq: bool = True
    eq_freq: float = 500.0
    eq_boost_range: Tuple[float, float] = (0.0, 4.0)
    snr_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(SNR_RANGES)
    )
    # Sub-segment silence detection
    subsegment_duration: float = 0.5
    subsegment_min_rms: float = 0.002
    # Noise screening
    screen_noise: bool = True
    noise_drone_energy_threshold: float = 0.6
    # Drone audibility check after mixing
    min_drone_energy_ratio: float = 0.3
    # Minimum drone sources per split
    min_sources_per_split: int = 5

    @property
    def clip_samples(self) -> int:
        return int(self.sample_rate * self.clip_duration)

    @property
    def subsegment_samples(self) -> int:
        return int(self.sample_rate * self.subsegment_duration)


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------
def scan_audio_files(directory: str) -> List[Path]:
    root = Path(directory)
    if not root.exists():
        log.error("Directory not found: %s", directory)
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_audio(filepath: Path, sr: int) -> Optional[np.ndarray]:
    try:
        import librosa
        y, _ = librosa.load(str(filepath), sr=sr, mono=True)
        if y is not None and len(y) > 0:
            return y.astype(np.float32)
    except Exception:
        pass
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(filepath))
        audio = audio.set_frame_rate(sr).set_channels(1).set_sample_width(2)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        return samples / 32768.0
    except Exception as exc:
        log.warning("Failed to load %s: %s", filepath, exc)
        return None


def save_wav(filepath: Path, audio: np.ndarray, sr: int) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    sf.write(filepath, np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")


def source_id(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------
def compute_rms(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def normalize_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    rms = compute_rms(audio)
    if rms < 1e-10:
        return audio.astype(np.float32)
    return (audio * (target_rms / rms)).astype(np.float32)


def extract_clips_no_overlap(
    audio: np.ndarray, clip_samples: int
) -> List[np.ndarray]:
    """Extract non-overlapping clips (zero overlap)."""
    if len(audio) < clip_samples:
        return []
    clips = []
    for start in range(0, len(audio) - clip_samples + 1, clip_samples):
        clips.append(audio[start: start + clip_samples].copy())
    return clips


def has_silent_subsegment(
    clip: np.ndarray,
    subseg_samples: int,
    min_rms: float,
) -> bool:
    """Check if any sub-segment of the clip is below the RMS threshold."""
    for start in range(0, len(clip) - subseg_samples + 1, subseg_samples):
        seg = clip[start: start + subseg_samples]
        if compute_rms(seg) < min_rms:
            return True
    return False


def is_valid_noise_clip(
    clip: np.ndarray,
    min_rms: float,
    max_rms: float,
    peak_ratio: float,
) -> bool:
    rms = compute_rms(clip)
    if rms < min_rms or rms > max_rms:
        return False
    peak = float(np.max(np.abs(clip)))
    return peak <= peak_ratio * rms


# ---------------------------------------------------------------------------
# Noise screening: detect drone-like frequency content in noise files
# ---------------------------------------------------------------------------
def compute_drone_energy_ratio(audio: np.ndarray, sr: int) -> float:
    """Compute the ratio of energy in drone-typical frequency bands.

    Drones produce strong energy in 100-8000 Hz with characteristic
    harmonic peaks. This computes the ratio of energy in that band
    relative to the full spectrum. High ratios suggest drone contamination.
    """
    if len(audio) < sr:  # need at least 1 second
        return 0.0
    try:
        # Use a segment from the middle for more representative analysis
        mid = len(audio) // 2
        seg_len = min(len(audio), sr * 2)  # 2 seconds max
        start = max(0, mid - seg_len // 2)
        segment = audio[start: start + seg_len]

        # Compute FFT
        fft = np.fft.rfft(segment)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)

        # Total energy
        total_energy = np.sum(power)
        if total_energy < 1e-12:
            return 0.0

        # Drone band energy (100-8000 Hz, where drone harmonics live)
        drone_mask = (freqs >= 100) & (freqs <= 8000)
        drone_energy = np.sum(power[drone_mask])

        # Also check for strong harmonic peaks (characteristic of motors)
        # Drones have fundamental frequencies typically 50-200 Hz with harmonics
        low_mask = (freqs >= 50) & (freqs <= 300)
        if np.sum(low_mask) > 0:
            low_power = power[low_mask]
            low_freqs = freqs[low_mask]
            # Check if there's a dominant peak (motor fundamental)
            if len(low_power) > 0:
                peak_idx = np.argmax(low_power)
                peak_power = low_power[peak_idx]
                mean_power = np.mean(low_power)
                # Strong peak = likely motor/drone
                if mean_power > 0 and peak_power / mean_power > 10.0:
                    return min(1.0, drone_energy / total_energy + 0.2)

        return float(drone_energy / total_energy)
    except Exception:
        return 0.0


def screen_noise_files(
    files: List[Path],
    sr: int,
    threshold: float,
) -> Tuple[List[Path], List[Path]]:
    """Screen noise files for drone-like spectral content.

    Returns (clean_files, flagged_files).
    """
    clean = []
    flagged = []

    for path in tqdm(files, desc="Screening noise files", unit="file"):
        audio = load_audio(path, sr)
        if audio is None:
            continue
        ratio = compute_drone_energy_ratio(audio, sr)
        if ratio > threshold:
            flagged.append(path)
            log.info(
                "Noise file flagged (drone energy ratio=%.3f): %s",
                ratio, path.name
            )
        else:
            clean.append(path)

    return clean, flagged


# ---------------------------------------------------------------------------
# Short clip combining
# ---------------------------------------------------------------------------
def combine_short_clips(
    files: List[Path],
    sr: int,
    clip_samples: int,
    root: Path,
) -> Tuple[List[Tuple[np.ndarray, str]], List[Path]]:
    """Group short audio files by filename pattern and concatenate.

    Returns:
        combined: List of (audio_array, group_source_id) for groups that
                  reached >= clip_samples after concatenation.
        long_files: Files that are already >= clip_samples individually.
    """
    short_groups: Dict[str, List[Path]] = defaultdict(list)
    long_files: List[Path] = []
    ungrouped_short: List[Path] = []

    for path in files:
        audio = load_audio(path, sr)
        if audio is None:
            continue
        if len(audio) >= clip_samples:
            long_files.append(path)
        else:
            key = group_key_for_combining(path)
            if key is not None:
                short_groups[key].append(path)
            else:
                ungrouped_short.append(path)

    combined: List[Tuple[np.ndarray, str]] = []

    for group_key, group_files in short_groups.items():
        # Load and concatenate all files in the group
        segments = []
        for path in sorted(group_files):
            audio = load_audio(path, sr)
            if audio is not None and len(audio) > 0:
                segments.append(audio)

        if not segments:
            continue

        concatenated = np.concatenate(segments)
        if len(concatenated) >= clip_samples:
            group_sid = f"combined:{group_key}({len(segments)}files)"
            combined.append((concatenated, group_sid))
            log.info(
                "Combined %d short files into %.1fs for group '%s'",
                len(segments),
                len(concatenated) / sr,
                group_key,
            )
        else:
            log.warning(
                "Group '%s' (%d files, %.1fs total) too short even after combining",
                group_key,
                len(segments),
                len(concatenated) / sr,
            )

    # Try to use ungrouped short files if they're long enough individually
    for path in ungrouped_short:
        audio = load_audio(path, sr)
        if audio is not None and len(audio) >= clip_samples:
            long_files.append(path)
        elif audio is not None:
            log.debug(
                "Short ungrouped file skipped (%.1fs): %s",
                len(audio) / sr, path.name,
            )

    return combined, long_files


# ---------------------------------------------------------------------------
# Drone and noise processing
# ---------------------------------------------------------------------------
def process_drone_folder(config: PipelineConfig) -> List[AudioClip]:
    root = Path(config.drone_dir)
    files = scan_audio_files(config.drone_dir)
    log.info("Found %d drone source files", len(files))

    rng = random.Random(config.seed)

    # Phase 1: Handle short clips by combining
    combined_groups, long_files = combine_short_clips(
        files, config.sample_rate, config.clip_samples, root
    )
    log.info(
        "Short clip combining: %d groups combined, %d files already long enough",
        len(combined_groups), len(long_files),
    )

    # Phase 2: Extract clips from all sources
    all_clips_by_source: Dict[str, List[AudioClip]] = defaultdict(list)
    rejected_silent = 0
    rejected_subseg = 0

    # Process long files normally
    for path in tqdm(long_files, desc="Processing drone files", unit="file"):
        audio = load_audio(path, config.sample_rate)
        if audio is None:
            continue
        sid = source_id(path, root)
        windows = extract_clips_no_overlap(audio, config.clip_samples)
        for window in windows:
            rms = compute_rms(window)
            if rms < config.drone_min_rms:
                rejected_silent += 1
                continue
            if has_silent_subsegment(
                window, config.subsegment_samples, config.subsegment_min_rms
            ):
                rejected_subseg += 1
                continue
            normalized = normalize_rms(window, config.target_rms)
            all_clips_by_source[sid].append(
                AudioClip(normalized, sid, compute_rms(normalized))
            )

    # Process combined groups
    for concatenated_audio, group_sid in combined_groups:
        windows = extract_clips_no_overlap(concatenated_audio, config.clip_samples)
        for window in windows:
            rms = compute_rms(window)
            if rms < config.drone_min_rms:
                rejected_silent += 1
                continue
            if has_silent_subsegment(
                window, config.subsegment_samples, config.subsegment_min_rms
            ):
                rejected_subseg += 1
                continue
            normalized = normalize_rms(window, config.target_rms)
            all_clips_by_source[group_sid].append(
                AudioClip(normalized, group_sid, compute_rms(normalized))
            )

    # Phase 3: Cap clips per source
    clips: List[AudioClip] = []
    capped_sources = 0
    for sid, source_clips in all_clips_by_source.items():
        if len(source_clips) > config.max_clips_per_source:
            # Randomly sample to keep diversity
            sampled = rng.sample(source_clips, config.max_clips_per_source)
            clips.extend(sampled)
            capped_sources += 1
        else:
            clips.extend(source_clips)

    log.info(
        "Drone clips: %d valid from %d sources | %d silent rejected | "
        "%d subseg-silent rejected | %d sources capped at %d",
        len(clips),
        len(all_clips_by_source),
        rejected_silent,
        rejected_subseg,
        capped_sources,
        config.max_clips_per_source,
    )
    return clips


def process_noise_folder(config: PipelineConfig) -> Tuple[List[AudioClip], List[str]]:
    """Process noise files with optional drone-content screening.

    Returns (clips, flagged_filenames).
    """
    root = Path(config.noise_dir)
    files = scan_audio_files(config.noise_dir)
    log.info("Found %d noise source files", len(files))

    rng = random.Random(config.seed + 7)
    flagged_names: List[str] = []

    # Phase 1: Screen noise files for drone contamination
    if config.screen_noise:
        clean_files, flagged_files = screen_noise_files(
            files, config.sample_rate, config.noise_drone_energy_threshold
        )
        flagged_names = [f.name for f in flagged_files]
        log.info(
            "Noise screening: %d clean, %d flagged as drone-contaminated",
            len(clean_files), len(flagged_files),
        )
        files = clean_files
    else:
        log.info("Noise screening: disabled")

    # Phase 2: Extract clips
    all_clips_by_source: Dict[str, List[AudioClip]] = defaultdict(list)
    rejected_short = 0
    rejected_bad = 0

    for path in tqdm(files, desc="Processing noise files", unit="file"):
        audio = load_audio(path, config.sample_rate)
        if audio is None:
            continue
        windows = extract_clips_no_overlap(audio, config.clip_samples)
        if not windows:
            rejected_short += 1
            continue

        sid = source_id(path, root)
        for window in windows:
            if not is_valid_noise_clip(
                window,
                config.noise_min_rms,
                config.noise_max_rms,
                config.noise_peak_ratio,
            ):
                rejected_bad += 1
                continue
            # Also check for silent sub-segments in noise
            if has_silent_subsegment(
                window, config.subsegment_samples, config.subsegment_min_rms
            ):
                rejected_bad += 1
                continue
            normalized = normalize_rms(window, config.target_rms)
            all_clips_by_source[sid].append(
                AudioClip(normalized, sid, compute_rms(normalized))
            )

    # Phase 3: Cap clips per source (noise too)
    clips: List[AudioClip] = []
    for sid, source_clips in all_clips_by_source.items():
        if len(source_clips) > config.max_clips_per_source:
            sampled = rng.sample(source_clips, config.max_clips_per_source)
            clips.extend(sampled)
        else:
            clips.extend(source_clips)

    log.info(
        "Noise clips: %d usable from %d sources | %d too short | %d rejected",
        len(clips), len(all_clips_by_source), rejected_short, rejected_bad,
    )
    return clips, flagged_names


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_groups_by_weight(
    group_weights: Dict[str, int],
    ratios: Dict[str, float],
    seed: int,
    min_per_split: int,
) -> Dict[str, str]:
    """Assign source groups to splits, respecting weight ratios."""
    if not group_weights:
        return {}
    if len(group_weights) < len(ratios) * min_per_split:
        log.warning(
            "Only %d source groups for %d splits (want %d per split minimum). "
            "Proceeding with best-effort assignment.",
            len(group_weights), len(ratios), min_per_split,
        )

    rng = random.Random(seed)
    items = [(name, max(1, int(weight))) for name, weight in group_weights.items()]
    rng.shuffle(items)
    items.sort(key=lambda item: item[1], reverse=True)

    split_names = list(ratios)
    total_weight = float(sum(weight for _, weight in items))
    targets = {name: max(1e-9, total_weight * ratios[name]) for name in split_names}
    loads = {name: 0.0 for name in split_names}
    counts = {name: 0 for name in split_names}
    assignment: Dict[str, str] = {}

    for group_name, weight in items:
        def score(split_name: str) -> Tuple[float, float]:
            projected = dict(loads)
            projected[split_name] += weight
            total_error = sum(
                abs(projected[name] - targets[name]) / targets[name]
                for name in split_names
            )
            return total_error, projected[split_name] / targets[split_name]

        split = min(split_names, key=score)
        assignment[group_name] = split
        loads[split] += weight
        counts[split] += 1

    # Report split allocation
    for split_name in split_names:
        log.info(
            "  Split '%s': %d sources, %d weight (target: %.0f)",
            split_name, counts[split_name], int(loads[split_name]),
            targets[split_name],
        )

    return assignment


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def apply_low_shelf_eq(
    audio: np.ndarray, sr: int, cutoff: float, boost_db: float
) -> np.ndarray:
    try:
        sos = iirfilter(
            N=2, Wn=cutoff, btype="lowpass", ftype="butter", fs=sr, output="sos"
        )
        low_band = sosfilt(sos, audio).astype(np.float32)
        gain = 10 ** (boost_db / 20.0) - 1.0
        return (audio + gain * low_band).astype(np.float32)
    except Exception:
        return audio


def apply_noise_variation(
    noise: np.ndarray, config: PipelineConfig, rng: np.random.Generator,
) -> np.ndarray:
    if len(noise) > 1:
        noise = np.roll(noise, int(rng.integers(0, len(noise))))
    gain_db = float(rng.uniform(-3.0, 3.0))
    noise = (noise * (10 ** (gain_db / 20.0))).astype(np.float32)
    if config.apply_noise_eq:
        boost_db = float(rng.uniform(*config.eq_boost_range))
        noise = apply_low_shelf_eq(
            noise, config.sample_rate, config.eq_freq, boost_db
        )
    return normalize_rms(noise, config.target_rms)


def mix_at_snr(
    drone: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> Tuple[np.ndarray, float, float]:
    drone_rms = compute_rms(drone)
    noise_rms = compute_rms(noise)
    if noise_rms < 1e-10:
        return drone.copy(), drone_rms, 0.0

    desired_noise_rms = drone_rms / (10 ** (snr_db / 20.0))
    scaled_noise = noise * (desired_noise_rms / noise_rms)
    mixed = drone + scaled_noise

    peak = float(np.max(np.abs(mixed)))
    if peak > 0.98:
        mixed = mixed / peak * 0.98

    return mixed.astype(np.float32), drone_rms, compute_rms(scaled_noise)


def check_drone_audibility(
    mixed: np.ndarray,
    drone: np.ndarray,
    min_energy_ratio: float,
) -> bool:
    """Verify drone signal is still audible in the mix.

    Checks that the drone's energy is at least `min_energy_ratio` of
    the mixed signal energy.
    """
    drone_energy = np.sum(np.square(drone))
    mixed_energy = np.sum(np.square(mixed))
    if mixed_energy < 1e-12:
        return False
    ratio = drone_energy / mixed_energy
    return ratio >= min_energy_ratio


# ---------------------------------------------------------------------------
# Negative sample selection
# ---------------------------------------------------------------------------
def choose_noise_for_negatives(
    noise_clips: List[AudioClip],
    target_count: int,
    rng: np.random.Generator,
) -> List[Tuple[AudioClip, bool]]:
    if target_count <= 0:
        return []
    if not noise_clips:
        raise ValueError("Cannot build negatives: split has no noise clips")

    if len(noise_clips) >= target_count:
        indices = rng.choice(len(noise_clips), size=target_count, replace=False)
        return [(noise_clips[int(index)], False) for index in indices]

    chosen = [(clip, False) for clip in noise_clips]
    extra_count = target_count - len(noise_clips)
    extra_indices = rng.integers(0, len(noise_clips), size=extra_count)
    chosen.extend((noise_clips[int(index)], True) for index in extra_indices)
    return chosen


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def add_metadata_row(
    rows: List[Dict[str, object]],
    output_file: str,
    folder: str,
    split: str,
    sample_rate: int,
    source_drone_file: str = "",
    source_noise_file: str = "",
    snr_db: str = "",
    drone_rms: str = "",
    noise_rms: str = "",
) -> None:
    rows.append({
        "output_file": output_file,
        "source_drone_file": source_drone_file,
        "source_noise_file": source_noise_file,
        "snr_db": snr_db,
        "difficulty": "clean" if folder in {"drone_clean", "noise_clean"} else folder,
        "drone_rms": drone_rms,
        "noise_rms": noise_rms,
        "sample_rate": sample_rate,
        "folder": folder,
        "split": split,
        "label_binary": FOLDER_TO_BINARY[folder],
    })


def audit_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    split_folder_counts = Counter((row["split"], row["folder"]) for row in rows)
    split_label_counts = Counter(
        (row["split"], str(row["label_binary"])) for row in rows
    )

    drone_source_splits: Dict[str, set] = defaultdict(set)
    noise_source_splits: Dict[str, set] = defaultdict(set)

    for row in rows:
        split = str(row["split"])
        drone_source = str(row.get("source_drone_file") or "")
        noise_source = str(row.get("source_noise_file") or "")
        if drone_source:
            drone_source_splits[drone_source].add(split)
        if noise_source:
            base = noise_source.split("#", 1)[0]
            noise_source_splits[base].add(split)

    drone_leaks = {
        s: sorted(splits) for s, splits in drone_source_splits.items()
        if len(splits) > 1
    }
    noise_leaks = {
        s: sorted(splits) for s, splits in noise_source_splits.items()
        if len(splits) > 1
    }

    # Per-split source counts
    split_drone_sources: Dict[str, int] = defaultdict(int)
    split_noise_sources: Dict[str, int] = defaultdict(int)
    seen_drone: Dict[str, set] = defaultdict(set)
    seen_noise: Dict[str, set] = defaultdict(set)
    for row in rows:
        split = str(row["split"])
        ds = str(row.get("source_drone_file") or "")
        ns = str(row.get("source_noise_file") or "")
        if ds:
            seen_drone[split].add(ds)
        if ns:
            seen_noise[split].add(ns.split("#", 1)[0])
    for split in seen_drone:
        split_drone_sources[split] = len(seen_drone[split])
    for split in seen_noise:
        split_noise_sources[split] = len(seen_noise[split])

    return {
        "total_rows": len(rows),
        "split_folder_counts": {
            f"{split}/{folder}": count
            for (split, folder), count in sorted(split_folder_counts.items())
        },
        "split_label_counts": {
            f"{split}/label_{label}": count
            for (split, label), count in sorted(split_label_counts.items())
        },
        "split_drone_sources": dict(split_drone_sources),
        "split_noise_sources": dict(split_noise_sources),
        "unique_drone_sources": len(drone_source_splits),
        "unique_noise_sources": len(noise_source_splits),
        "drone_source_leaks": drone_leaks,
        "noise_source_leaks": noise_leaks,
    }


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------
def ensure_output_dirs(config: PipelineConfig) -> Dict[str, Path]:
    root = Path(config.output_dir).resolve()
    folders = ["drone_clean", "noise_clean", *config.snr_ranges.keys()]

    if root.exists() and any(root.iterdir()):
        if not config.overwrite:
            raise SystemExit(
                f"Output directory is not empty: {root}. "
                "Use --overwrite or choose a new --output_dir."
            )
        for child in root.iterdir():
            if child.is_dir() and child.name in folders:
                shutil.rmtree(child)
            elif child.name in {"metadata.csv", "split_summary.json", "flagged_noise.json"}:
                child.unlink()

    root.mkdir(parents=True, exist_ok=True)
    dirs = {folder: root / folder for folder in folders}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(config: PipelineConfig) -> None:
    rng = np.random.default_rng(config.seed)

    log.info("=" * 70)
    log.info("Pipeline v4 — Leakage-safe drone preprocessing")
    log.info("=" * 70)
    log.info("Drone dir:  %s", config.drone_dir)
    log.info("Noise dir:  %s", config.noise_dir)
    log.info("Output dir: %s", config.output_dir)
    log.info(
        "Clip: %.2fs @ %d Hz | zero overlap",
        config.clip_duration, config.sample_rate,
    )
    log.info("Max clips per source: %d", config.max_clips_per_source)
    log.info("SNR ranges: %s", config.snr_ranges)
    log.info("Noise screening: %s", config.screen_noise)

    drone_clips = process_drone_folder(config)
    noise_clips, flagged_noise = process_noise_folder(config)

    if not drone_clips:
        raise SystemExit("No valid drone clips extracted")
    if not noise_clips:
        raise SystemExit("No usable noise clips extracted")

    # --- Split sources ---
    difficulties = list(config.snr_ranges.keys())
    variants_per_drone = 1 + len(difficulties)  # clean + N difficulties

    drone_weights = {
        source: count * variants_per_drone
        for source, count in Counter(c.source for c in drone_clips).items()
    }
    noise_weights = Counter(c.source for c in noise_clips)

    log.info("")
    log.info("Splitting %d drone sources:", len(drone_weights))
    drone_split_map = split_groups_by_weight(
        drone_weights, SPLIT_RATIOS, config.seed, config.min_sources_per_split
    )

    log.info("Splitting %d noise sources:", len(noise_weights))
    noise_split_map = split_groups_by_weight(
        dict(noise_weights), SPLIT_RATIOS, config.seed + 17,
        config.min_sources_per_split,
    )

    drone_by_split: Dict[str, List[AudioClip]] = {s: [] for s in SPLIT_RATIOS}
    noise_by_split: Dict[str, List[AudioClip]] = {s: [] for s in SPLIT_RATIOS}

    for clip in drone_clips:
        drone_by_split[drone_split_map[clip.source]].append(clip)
    for clip in noise_clips:
        noise_by_split[noise_split_map[clip.source]].append(clip)

    for split in SPLIT_RATIOS:
        if not drone_by_split[split]:
            raise SystemExit(f"Split '{split}' has no drone clips")
        if not noise_by_split[split]:
            raise SystemExit(f"Split '{split}' has no noise clips")

    # --- Write clips ---
    dirs = ensure_output_dirs(config)
    output_root = Path(config.output_dir)
    metadata_rows: List[Dict[str, object]] = []
    file_counters: Counter[str] = Counter()

    log.info("")
    log.info("Writing split-safe clips")

    for split in SPLIT_RATIOS:
        split_drone = drone_by_split[split]
        split_noise = noise_by_split[split]

        positive_count = len(split_drone) * variants_per_drone
        target_noise_count = int(round(positive_count * config.negative_ratio))

        log.info(
            "%s: %d drone clips -> %d positives | "
            "%d noise clips -> %d negatives",
            split, len(split_drone), positive_count,
            len(split_noise), target_noise_count,
        )

        # Write drone_clean
        for clip in tqdm(split_drone, desc=f"{split}: drone_clean", unit="clip"):
            file_counters["drone_clean"] += 1
            fname = f"{split}_drone_{file_counters['drone_clean']:06d}.wav"
            rel_path = f"drone_clean/{fname}"
            save_wav(dirs["drone_clean"] / fname, clip.audio, config.sample_rate)
            add_metadata_row(
                metadata_rows, rel_path, "drone_clean", split,
                config.sample_rate, source_drone_file=clip.source,
                drone_rms=f"{clip.rms:.6f}",
            )

        # Write noise_clean (balanced to match positive count)
        selected_noise = choose_noise_for_negatives(
            split_noise, target_noise_count, rng
        )
        for idx, (clip, augmented) in enumerate(
            tqdm(selected_noise, desc=f"{split}: noise_clean", unit="clip")
        ):
            file_counters["noise_clean"] += 1
            fname = f"{split}_noise_{file_counters['noise_clean']:06d}.wav"
            rel_path = f"noise_clean/{fname}"
            audio = clip.audio
            source = clip.source
            if augmented:
                audio = apply_noise_variation(audio.copy(), config, rng)
                source = f"{clip.source}#aug{idx}"
            save_wav(dirs["noise_clean"] / fname, audio, config.sample_rate)
            add_metadata_row(
                metadata_rows, rel_path, "noise_clean", split,
                config.sample_rate, source_noise_file=source,
                noise_rms=f"{compute_rms(audio):.6f}",
            )

        # Write mixed positives
        rejected_inaudible = 0
        for drone_clip in tqdm(
            split_drone, desc=f"{split}: mixed positives", unit="clip"
        ):
            for difficulty in difficulties:
                noise_clip = split_noise[int(rng.integers(0, len(split_noise)))]
                noise_audio = apply_noise_variation(
                    noise_clip.audio.copy(), config, rng
                )
                snr_low, snr_high = config.snr_ranges[difficulty]
                snr_db = float(rng.uniform(snr_low, snr_high))
                mixed, d_rms, n_rms = mix_at_snr(
                    drone_clip.audio, noise_audio, snr_db
                )

                # Audibility check
                if not check_drone_audibility(
                    mixed, drone_clip.audio, config.min_drone_energy_ratio
                ):
                    rejected_inaudible += 1
                    # Retry with a higher SNR
                    snr_db = float(rng.uniform(snr_high, snr_high + 5.0))
                    mixed, d_rms, n_rms = mix_at_snr(
                        drone_clip.audio, noise_audio, snr_db
                    )

                file_counters[difficulty] += 1
                fname = (
                    f"{split}_drone_{file_counters[difficulty]:06d}"
                    f"_{difficulty}.wav"
                )
                rel_path = f"{difficulty}/{fname}"
                save_wav(dirs[difficulty] / fname, mixed, config.sample_rate)
                add_metadata_row(
                    metadata_rows, rel_path, difficulty, split,
                    config.sample_rate,
                    source_drone_file=drone_clip.source,
                    source_noise_file=noise_clip.source,
                    snr_db=f"{snr_db:.2f}",
                    drone_rms=f"{d_rms:.6f}",
                    noise_rms=f"{n_rms:.6f}",
                )

        if rejected_inaudible:
            log.info(
                "  %s: %d mixes retried due to inaudible drone",
                split, rejected_inaudible,
            )

    # --- Write metadata ---
    fieldnames = [
        "output_file", "source_drone_file", "source_noise_file",
        "snr_db", "difficulty", "drone_rms", "noise_rms",
        "sample_rate", "folder", "split", "label_binary",
    ]

    metadata_path = output_root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    summary = audit_rows(metadata_rows)
    summary_path = output_root / "split_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # Save flagged noise files
    if flagged_noise:
        flagged_path = output_root / "flagged_noise.json"
        with flagged_path.open("w", encoding="utf-8") as handle:
            json.dump({"flagged_noise_files": flagged_noise}, handle, indent=2)
        log.info("Flagged noise files saved: %s", flagged_path)

    if summary["drone_source_leaks"] or summary["noise_source_leaks"]:
        log.error("Source leakage detected!")
        log.error("Drone leaks: %d", len(summary["drone_source_leaks"]))
        log.error("Noise leaks: %d", len(summary["noise_source_leaks"]))
        raise SystemExit(1)

    log.info("")
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info("Metadata: %s", metadata_path)
    log.info("Summary:  %s", summary_path)
    log.info("Rows:     %d", len(metadata_rows))
    log.info("No source overlap across splits ✓")
    for key, count in sorted(summary["split_label_counts"].items()):
        log.info("  %s: %d", key, count)
    log.info("Drone sources per split: %s", summary["split_drone_sources"])
    log.info("Noise sources per split: %s", summary["split_noise_sources"])
    log.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Pipeline v4: leakage-safe drone audio preprocessing"
    )
    parser.add_argument("--drone_dir", default="drone")
    parser.add_argument("--noise_dir", default="noise")
    parser.add_argument("--output_dir", default="output4")
    parser.add_argument("--sample_rate", type=int, default=16_000)
    parser.add_argument("--clip_duration", type=float, default=2.0)
    parser.add_argument("--target_rms", type=float, default=0.1)
    parser.add_argument("--drone_min_rms", type=float, default=0.01)
    parser.add_argument("--noise_min_rms", type=float, default=0.002)
    parser.add_argument("--noise_max_rms", type=float, default=0.3)
    parser.add_argument("--noise_peak_ratio", type=float, default=3.0)
    parser.add_argument("--negative_ratio", type=float, default=1.0)
    parser.add_argument("--max_clips_per_source", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_noise_eq", action="store_true")
    parser.add_argument("--no_noise_screen", action="store_true")
    parser.add_argument(
        "--noise_drone_threshold", type=float, default=0.6,
        help="Drone-energy ratio threshold for noise screening (0-1).",
    )
    parser.add_argument(
        "--min_drone_energy_ratio", type=float, default=0.3,
        help="Minimum drone energy ratio in mixed clips for audibility check.",
    )
    parser.add_argument(
        "--min_sources_per_split", type=int, default=5,
        help="Minimum number of drone sources per split.",
    )
    parser.add_argument(
        "--overwrite", dest="overwrite", action="store_true",
        help="Overwrite the output directory.",
    )
    parser.add_argument(
        "--no_overwrite", dest="overwrite", action="store_false",
    )
    parser.set_defaults(overwrite=True)
    args = parser.parse_args()

    return PipelineConfig(
        drone_dir=args.drone_dir,
        noise_dir=args.noise_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        clip_duration=args.clip_duration,
        target_rms=args.target_rms,
        drone_min_rms=args.drone_min_rms,
        noise_min_rms=args.noise_min_rms,
        noise_max_rms=args.noise_max_rms,
        noise_peak_ratio=args.noise_peak_ratio,
        negative_ratio=args.negative_ratio,
        max_clips_per_source=args.max_clips_per_source,
        overwrite=args.overwrite,
        apply_noise_eq=not args.no_noise_eq,
        seed=args.seed,
        screen_noise=not args.no_noise_screen,
        noise_drone_energy_threshold=args.noise_drone_threshold,
        min_drone_energy_ratio=args.min_drone_energy_ratio,
        min_sources_per_split=args.min_sources_per_split,
    )


if __name__ == "__main__":
    run_pipeline(parse_args())
