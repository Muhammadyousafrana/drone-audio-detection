# Drone Audio Detection

<p align="center">
  <img src="assets/drone-audio-detection-banner.svg" alt="Drone Audio Detection — machine learning for drone sound recognition" width="100%">
</p>

<p align="center">
  Source-aware audio preparation and a TensorFlow classifier for detecting drone sound in noisy recordings.
</p>

## Overview

Drone Audio Detection prepares short, labeled audio clips and trains a binary classifier to distinguish drone recordings from background noise. The repository includes the data preparation pipeline, training script, a processed sample dataset, and exported model and evaluation artifacts.

The training model uses a frozen [YAMNet](https://tfhub.dev/google/yamnet/1) audio embedding backbone with mean pooling and a compact Keras classification head. The training script checks source-level split separation to help prevent clips from the same original recording leaking across train, validation, and test sets.

## What's included

- `pipeline4.py` — reads source recordings, creates fixed-length clips, normalizes audio, combines drone and noise recordings, assigns labels and splits, and writes a metadata manifest.
- `train_model4.py` — loads the manifest, trains and evaluates the binary classifier, selects a decision threshold, and exports Keras and TensorFlow Lite models with reports and plots.
- `output4/` — bundled processed dataset with 3,888 WAV clips and `metadata.csv`.
- `model4_artifacts/` — trained model exports, predictions, metrics, plots, and training logs.

Raw source recordings in `drone/` and `noise/` and the local `.venv/` are intentionally excluded from the repository. Supply source audio in those folders to regenerate the processed dataset. Check the source datasets' licenses and redistribution terms before using them.

## Dataset

The included dataset contains **3,888 mono WAV clips**, each two seconds long at 16 kHz. The current manifest has these folders:

| Folder | Clips | Meaning |
|---|---:|---|
| `drone_clean` | 648 | Drone audio without added background noise |
| `easy` | 648 | Drone audio mixed with background noise at an easier signal-to-noise level |
| `hard` | 648 | Drone audio mixed with background noise at a harder signal-to-noise level |
| `noise_clean` | 1,944 | Background noise without drone audio |

The current split counts are 2,682 training, 606 validation, and 600 test clips. Splits are grouped by source recordings rather than individual clips, so a source recording is kept within one split.

`output4/metadata.csv` is the manifest and records each clip's relative path, source drone and noise files, SNR, difficulty, sample rate, folder, split, and binary label (`1` for drone-containing clips, `0` for clean noise).

## Quick start

Use a Python version and TensorFlow build compatible with your platform. Install the libraries used by the scripts:

```bash
python -m pip install numpy pandas scipy soundfile tqdm tensorflow tensorflow-hub scikit-learn matplotlib
```

Train using the bundled dataset and default settings:

```bash
python train_model4.py
```

The default training configuration uses 40 epochs, a batch size of 32, and writes models, metrics, predictions, and plots to `model4_artifacts/`. The first run may download the YAMNet model from TensorFlow Hub.

To use different manifest, audio, or output locations:

```bash
python train_model4.py \
  --metadata output4/metadata.csv \
  --audio_root output4 \
  --output_dir model4_artifacts
```

## Rebuild the processed dataset

Place source audio files under `drone/` and `noise/`, then run:

```bash
python pipeline4.py
```

The default output location is `output4/`. Common options include:

```bash
python pipeline4.py --drone_dir drone --noise_dir noise --output_dir output4
python pipeline4.py --clip_duration 2.0 --sample_rate 16000 --max_clips_per_source 30 --seed 42
```

See `python pipeline4.py --help` and `python train_model4.py --help` for the full option lists. Rebuilding replaces or regenerates files in the selected output directory; back up any local data there that you need to keep.

## Training outputs

The checked-in `model4_artifacts/` directory includes:

- `best_model4.keras` — Keras model.
- `best_model4.tflite` — TensorFlow Lite export.
- `results4.json` — summarized evaluation results and selected threshold.
- `predictions_*.csv` — per-clip predictions for validation and test splits.
- CSV logs, diagnostic reports, and plots for training and evaluation.

## Project layout

```text
.
├── assets/
│   └── drone-audio-detection-banner.svg
├── output4/
│   ├── drone_clean/
│   ├── easy/
│   ├── hard/
│   ├── noise_clean/
│   └── metadata.csv
├── model4_artifacts/
├── pipeline4.py
├── train_model4.py
└── README.md
```

## Notes

- The dataset and artifacts are provided as the current project outputs; regenerate them when changing source audio or pipeline settings.
- Raw source recordings are not included. Their ownership, licenses, and attribution requirements remain with their respective providers.
- Model quality depends on the source recordings and recording conditions represented in the data. Review `results4.json` and the diagnostic plots before using the model in a new environment.
