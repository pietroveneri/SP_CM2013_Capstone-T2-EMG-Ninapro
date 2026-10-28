
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]

from emg_ninapro import EMGNinaproTrack

# Original baseline configuration
cfg = {
    "seed": 0,
    "select": "none",
    "imbalance": "balanced",
    "cv_max_splits": 5,
    "loso_max_groups": 12,
}

track = EMGNinaproTrack()
records = track.load(ROOT / "NinaPro_Mat", exercise="E1")
track.assert_no_holdout(records)

# Build the original 50-feature matrix
X, y, subjects, _ = track._build(records, cfg)

assert {r.group for r in records} == {
    f"S{i}" for i in range(1, 23)
}
assert X.shape[1] == 50
assert np.isfinite(X).all()

# Recover the archived new-subject fold assignments
fold_path = (
    ROOT / "docs" / "Baseline"
    / "baseline_20261022_holdout_separated"
    / "folds.json"
)
folds = json.loads(fold_path.read_text())
folds = sorted(
    (f for f in folds if f["mode"] == "new_subject"),
    key=lambda f: f["fold"],
)
assert len(folds) == 5

# Feature order from emg_ninapro._win_features()
families = ["MAV", "RMS", "WL", "VAR", "rFFT_centroid"]
names = [
    f"ch{ch:02d}_{feature}"
    for ch in range(1, 11)
    for feature in families
]

importances = []

for fold in folds:
    train_groups = set(fold["train_groups"])
    test_groups = set(fold["test_groups"])

    assert train_groups.isdisjoint(test_groups)

    train = np.flatnonzero(np.isin(subjects, list(train_groups)))
    test = np.flatnonzero(np.isin(subjects, list(test_groups)))

    assert len(train) == fold["train_n"]
    assert len(test) == fold["test_n"]

    # Refit the exact baseline pipeline on training data only
    selector = track.select_features(X[train], y[train], cfg)
    model = clone(track._baseline(cfg))
    model.fit(selector.transform(X[train]), y[train])

    importance = model.named_steps["clf"].feature_importances_
    importances.append(importance)

scores = np.vstack(importances)

df = pd.DataFrame({
    "feature": names,
    "importance": scores.mean(axis=0),
    "fold_sd": scores.std(axis=0, ddof=1),
    "channel": np.repeat(np.arange(1, 11), 5),
    "family": families * 10,
})

print("\nTOP 20 INDIVIDUAL FEATURES")
print(df.sort_values("importance", ascending=False)
        [["feature", "importance", "fold_sd"]]
        .head(20).to_string(index=False))

print("\nIMPORTANCE BY FEATURE FAMILY")
print(df.groupby("family")["importance"]
        .sum().sort_values(ascending=False).to_string())

print("\nIMPORTANCE BY EMG CHANNEL")
print(df.groupby("channel")["importance"]
        .sum().sort_values(ascending=False).to_string())

print("\nTotal importance:", df["importance"].sum())
print("Folds analyzed:", len(folds))
