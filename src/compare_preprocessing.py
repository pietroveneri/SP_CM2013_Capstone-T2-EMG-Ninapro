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

from emg_ninapro import EMGNinaproTrack
from filter_option02 import FilterOption02
from filter_option03 import FilterOption03
from features import extract_window_features, expanded_feature_names


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs" / "preprocessing_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

loader = EMGNinaproTrack()
records = loader.load(DATA_DIR, exercise="E1")

if not records:
    raise RuntimeError(f"No recordings found in {DATA_DIR}")

loader.assert_no_holdout(records)


gestures = [
    "g01", "g02", "g03", "g04", "g05", "g06", "g07", "g08", "g09", "g10", "g11", "g12", "g13"
]
# index 0, "Thumb abduction"
# index 1, "Index flexion"
# index 2, "Middle flexion"
# index 3, "Ring flexion"
# index 4, "Little flexion"
# index 5, "Thumb adduction"
# index 6, "Index extension"
# index 7, "Middle extension"
# index 8, "Ring extension"
# index 9, "Little extension"
# index 10, "Thumb flexion"
# index 11, "Index abduction"
# index 12, "Middle abduction"
# index 13, "Ring abduction"



options = {
    "Original": None,
    "02": FilterOption02(),
    "03": FilterOption03(),
}

# -------------------------------------------------
# # Per subject, gesture and channel, store the median feature value for each preprocessing option.
# GESTURE = "g01"  
# CHANNEL = 1

# # Per subject, gesture and channel, store the median feature value for each preprocessing option.
# feature_types = [
#     "MAV", "WL", "VAR", "relative_MAV",
#     "mean_frequency", "median_frequency",
# ]
# names = expanded_feature_names()

# for record in records:
#     mask = np.asarray(record.labels) == GESTURE
#     if not mask.any():
#         continue

#     matrices = {}
#     for name, option in options.items():
#         processed = (
#             record if option is None
#             else option.preprocess(record)
#         )
#         matrices[name] = extract_window_features(processed)[0]

#     fig, axes = plt.subplots(2, 3, figsize=(12, 7))

#     for ax, feature in zip(axes.ravel(), feature_types):
#         column = names.index(f"ch{CHANNEL:02d}_{feature}")

#         ax.boxplot(
#             [X[mask, column] for X in matrices.values()],
#             showfliers=False,
#         )
#         ax.set_xticks(range(1, 5), options.keys())
#         ax.set_title(feature)
#         ax.grid(axis="y", alpha=0.2)

#     fig.suptitle(
#         f"{record.group} — {GESTURE}, channel {CHANNEL}"
#     )
#     fig.tight_layout()
#     fig.savefig(
#         OUTPUT_DIR / f"{record.group}_{GESTURE}_ch{CHANNEL:02d}.png",
#         dpi=150,
#         bbox_inches="tight",
#     )
#     plt.close(fig)

# -------------------------------------------------

# Summary of feature distributions across all subjects, gestures and channels.
feature_types = [
    "MAV", "WL", "VAR", "relative_MAV",
    "mean_frequency", "median_frequency",
]

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
    for name, option in options.items():
        processed = (
            record if option is None
            else option.preprocess(record)
        )
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
        list(options.keys()),
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