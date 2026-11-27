"""Verify the ensemble submission path against iteration 31 on S1-S22."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from development_folds import BASELINE, load_development_records
from ensemble_track import FilteredEnsembleTrack
from features import FeatureTrack
from preprocessing import filter_windows


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "ensemble_submission_check"


def aligned_probabilities(classifier, features, labels, train, test, classes):
    model = make_pipeline(StandardScaler(), classifier)
    model.fit(features[train], labels[train])
    if set(model.classes_) != set(classes):
        raise ValueError("Training fold does not contain all expected gesture classes.")
    columns = pd.Index(model.classes_).get_indexer(classes)
    return model.predict_proba(features[test])[:, columns]


def compute_reference(records, folds):
    """Reproduce iteration 31 (9f7c992) independently of the submission track."""
    filtered = [filter_windows(rec, "median-savgol") for rec in records]
    rf_track = FeatureTrack("time_40")
    lda_track = FeatureTrack("baseline")
    rf_features, labels, subjects, repetitions = rf_track._build(filtered)
    lda_features, lda_labels, lda_subjects, lda_repetitions = lda_track._build(filtered)
    np.testing.assert_array_equal(lda_labels, labels)
    np.testing.assert_array_equal(lda_subjects, subjects)
    np.testing.assert_array_equal(lda_repetitions, repetitions)
    if rf_features.shape != (len(labels), 40) or lda_features.shape != (len(labels), 50):
        raise ValueError("Expected 40 RF and 50 LDA features per development window.")

    classes = np.asarray(rf_track.meta.classes)
    epochs = np.concatenate([np.arange(len(rec.labels)) for rec in records])
    windows = pd.MultiIndex.from_arrays([subjects, epochs], names=["record", "epoch"])
    predictions = []
    for fold in folds:
        train = np.isin(subjects, fold["train_groups"])
        test = np.isin(subjects, fold["test_groups"])
        if np.any(train & test) or not np.all(train | test):
            raise ValueError(f"Invalid subject partition in fold {fold['fold']}.")
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        rf_probs = aligned_probabilities(rf, rf_features, labels, train, test, classes)
        lda_probs = aligned_probabilities(lda, lda_features, labels, train, test, classes)
        winners = np.argmax((rf_probs + lda_probs) / 2, axis=1)
        predictions.append(pd.DataFrame(
            {"fold": fold["fold"], "label": classes[winners]}, index=windows[test]
        ))

    return pd.concat(predictions, verify_integrity=True)


def main():
    track = FilteredEnsembleTrack()
    records = load_development_records(track, ROOT / "NinaPro_Mat")

    archive = json.loads((BASELINE / "folds.json").read_text(encoding="utf-8"))
    folds = sorted(
        (f for f in archive if f["mode"] == "new_subject"),
        key=lambda f: f["fold"],
    )
    if [f["fold"] for f in folds] != list(range(5)):
        raise ValueError("Expected development folds 0 through 4.")

    reference = compute_reference(records, folds)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    total = 0

    for f in folds:
        fold = f["fold"]
        train_recs = [r for r in records if r.group in f["train_groups"]]
        test_recs = [r for r in records if r.group in f["test_groups"]]

        # Pass the loaded recordings directly. The track applies the filters.
        model = track.train_baseline(train_recs=train_recs)
        path = OUTPUT / f"dev_fold_{fold}.csv"
        track.write_submission(test_recs=test_recs, path=path, model=model)

        actual = pd.read_csv(path).set_index(
            ["record", "epoch"], verify_integrity=True
        )["label"]
        expected = reference.loc[reference["fold"] == fold, "label"]

        # Detect missing windows, extra windows and different predicted labels.
        pd.testing.assert_series_equal(
            actual.sort_index(),
            expected.sort_index(),
            check_dtype=False,
            obj=f"Fold {fold} predictions",
        )

        total += len(actual)
        print(f"Fold {fold}: all {len(actual):,} predictions match.")

    if total != len(reference) or total != 60_182:
        raise ValueError(f"Expected 60,182 development windows, verified {total:,}.")

    print(f"PASS: all {total:,} predictions match across the five folds.")


if __name__ == "__main__":
    main()
