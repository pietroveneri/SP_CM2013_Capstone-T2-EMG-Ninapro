# File: src/compare_feature_sets.py
# This script compares the performance of different feature sets on the NinaPro E1 dataset.
# It evaluates the original 70-feature set, 
    # a reduced 60-feature set (excluding RMS), 
    # a time-domain only 40-feature set, 
    # and an amplitude-domain only 20-feature set.

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from emg_ninapro import EMGNinaproTrack
from features import extract_window_features, expanded_feature_names


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "feature_selection"
OUTPUT.mkdir(parents=True, exist_ok=True)

CFG = {
    "seed": 0,
    "select": "none",
    "imbalance": "balanced",
    "cv_max_splits": 5,
    "loso_max_groups": 12,
}
# full_70: all features, including RMS.
# full_60: all features except RMS.
# time_40: only time-domain features.
# amplitude_20: only amplitude-domain features.

FEATURE_SETS = {
    "full_70": {"RMS", "MAV", "WL", "VAR", "relative_MAV",
                 "mean_frequency", "median_frequency"},
    "full_60": {
        "MAV", "WL", "VAR", "relative_MAV",
        "mean_frequency", "median_frequency",
    },
    "time_40": {"MAV", "WL", "VAR", "relative_MAV"},
    "amplitude_20": {"MAV", "relative_MAV"},
}

track = EMGNinaproTrack()

print("Loading development recordings...", flush=True)
records = track.load(ROOT / "NinaPro_Mat", exercise="E1")
track.assert_no_holdout(records)

assert {r.group for r in records} == {
    f"S{i}" for i in range(1, 23)
}

# Explicit expanded extraction; no preprocessing.
Xs, ys, groups = [], [], []

for record in records:
    X_record, y_record, subject = extract_window_features(record)
    Xs.append(X_record)
    ys.append(y_record)
    groups.append(np.repeat(subject, len(y_record)))

X = np.vstack(Xs)
y = np.concatenate(ys)
subjects = np.concatenate(groups)

names = np.asarray(expanded_feature_names())
families = np.asarray([name.split("_", 1)[1] for name in names])

assert X.shape[1] == len(names) == 70
assert np.isfinite(X).all()

fold_path = (
    ROOT / "docs" / "Baseline"
    / "baseline_20261022_holdout_separated"
    / "folds.json"
)
folds = json.loads(fold_path.read_text())
folds = sorted(
    [f for f in folds if f["mode"] == "new_subject"],
    key=lambda f: f["fold"],
)
assert len(folds) == 5

fold_results = []
subject_results = []
importance_results = []
test_counts = np.zeros(len(y), dtype=int)

for fold in folds:
    fold_id = fold["fold"]
    train_groups = set(fold["train_groups"])
    test_groups = set(fold["test_groups"])

    assert train_groups.isdisjoint(test_groups)
    assert train_groups | test_groups == set(subjects)

    train = np.flatnonzero(np.isin(subjects, list(train_groups)))
    test = np.flatnonzero(np.isin(subjects, list(test_groups)))

    assert len(train) == fold["train_n"]
    assert len(test) == fold["test_n"]
    test_counts[test] += 1

    for set_name, retained in FEATURE_SETS.items():
        columns = np.flatnonzero(np.isin(families, list(retained)))
        X_train = X[np.ix_(train, columns)]
        X_test = X[np.ix_(test, columns)]

        print(
            f"Fold {fold_id}: {set_name}, fitting...",
            flush=True,
        )

        # Same baseline classifier and settings for every feature set.
        model = clone(track._baseline(CFG))
        model.fit(X_train, y[train])
        predictions = model.predict(X_test)

        score = f1_score(y[test], predictions, average="macro")
        fold_results.append({
            "fold": fold_id,
            "feature_set": set_name,
            "n_features": len(columns),
            "macro_f1": score,
        })
        print(f"  Macro-F1: {score:.4f}", flush=True)

        for subject in np.unique(subjects[test]):
            mask = subjects[test] == subject
            subject_results.append({
                "fold": fold_id,
                "subject": subject,
                "feature_set": set_name,
                "macro_f1": f1_score(
                    y[test][mask],
                    predictions[mask],
                    average="macro",
                ),
            })

        # Rank the full sets only; reduced sets are assessed by refitting.
        if set_name in ["full_70", "full_60"]:
            print("  Permutation importance...", flush=True)
            result = permutation_importance(
                model,
                X_test,
                y[test],
                scoring="f1_macro",
                n_repeats=5,
                random_state=CFG["seed"],
                n_jobs=1,
            )

            for i, column in enumerate(columns):
                importance_results.append({
                    "fold": fold_id,
                    "feature_set": set_name,
                    "feature": names[column],
                    "importance": result.importances_mean[i],
                    "shuffle_sd": result.importances_std[i],
                })

assert np.all(test_counts == 1)

fold_df = pd.DataFrame(fold_results)
subject_df = pd.DataFrame(subject_results)
importance_df = pd.DataFrame(importance_results)

ranking = (
    importance_df.groupby(["feature_set", "feature"])["importance"]
    .agg(mean_importance="mean", fold_sd="std")
    .sort_values(["feature_set", "mean_importance"], ascending=[True, False])
)

# Equal weight for each subject.
summary = (
    subject_df.groupby("feature_set")["macro_f1"]
    .agg(mean_subject_f1="mean", subject_sd="std")
)

paired = subject_df.pivot(
    index="subject",
    columns="feature_set",
    values="macro_f1",
)

for candidate in ("full_60", "time_40", "amplitude_20"):
    difference = paired[candidate] - paired["full_70"]
    summary.loc[candidate, "mean_difference_vs_full"] = difference.mean()
    summary.loc[candidate, "subjects_better_than_full"] = (difference > 0).sum()

fold_df.to_csv(OUTPUT / "fold_scores.csv", index=False)
subject_df.to_csv(OUTPUT / "subject_scores.csv", index=False)
importance_df.to_csv(OUTPUT / "importance_by_fold.csv", index=False)
ranking.to_csv(OUTPUT / "importance_ranking.csv")
summary.to_csv(OUTPUT / "feature_set_summary.csv")

print("\nFEATURE SET COMPARISON")
print(summary.round(4).to_string())

for set_name in ("full_70", "full_60"):
    print(f"\nTOP 20 FEATURES: {set_name}")
    print(ranking.loc[set_name].head(20).round(5).to_string())

print(f"\nSaved results to {OUTPUT}")
