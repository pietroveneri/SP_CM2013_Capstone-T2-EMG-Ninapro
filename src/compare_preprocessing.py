# File: src/compare_preprocessing.py
# This script compares the distributions of features extracted from EMG signals
# after different preprocessing options. It generates boxplots for each feature type
# across all subjects, gestures, and channels, allowing for visual comparison of the
# effects of preprocessing on feature distributions.

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from development_folds import load_development_records
from emg_ninapro import EMGNinaproTrack
from preprocessing import filter_windows
from features import FEATURE_SETS, extract_window_features, expanded_feature_names


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "NinaPro_Mat"
OUTPUT_DIR = ROOT / "outputs" / "preprocessing_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

loader = EMGNinaproTrack()
records = load_development_records(loader, DATA_DIR)

options = ("none", "median-savgol")

# Summary of feature distributions across all subjects, gestures and channels.
feature_types = FEATURE_SETS["full_60"]

names = expanded_feature_names()

# Store one median per subject, gesture and channel.
summary = {
    feature: {option: [] for option in options}
    for feature in feature_types
}

for record in records:
    print(f"Processing {record.group}...", flush=True)

    # Extract features once per preprocessing option.
    matrices = {}
    for name in options:
        processed = filter_windows(record, name)
        matrices[name] = extract_window_features(processed)[0]

    labels = np.asarray(record.labels)
    gestures = np.unique(labels)
    n_channels = record.epochs["emg"].shape[-1]

    for gesture in gestures:
        mask = labels == gesture

        for channel in range(1, n_channels + 1):
            for feature in feature_types:
                column = names.index(
                    f"ch{channel:02d}_{feature}"
                )

                for option_name, X in matrices.items():
                    median_value = np.median(X[mask, column])
                    summary[feature][option_name].append(median_value)

# Plot all subject–gesture–channel medians.
fig, axes = plt.subplots(2, 3, figsize=(12, 7))

for ax, feature in zip(axes.ravel(), feature_types):
    values = [
        summary[feature][option_name]
        for option_name in options
    ]

    ax.boxplot(values, showfliers=False)
    ax.set_xticks(
        range(1, len(options) + 1),
        list(options),
    )
    ax.set_title(feature)
    ax.grid(axis="y", alpha=0.2)

fig.suptitle(
    "Feature distributions across subjects, gestures and channels\n"
    "Each value is a subject–gesture–channel median"
)
fig.tight_layout()

path = OUTPUT_DIR / "feature_summary_all_subjects.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {path}")