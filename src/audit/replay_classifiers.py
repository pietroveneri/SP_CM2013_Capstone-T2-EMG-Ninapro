"""Replay development classifiers and save indexed out-of-fold predictions.

Run with the baseline Python environment. --jobs parallelizes independent folds.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from development_folds import BASELINE, development_folds
from features import expanded_feature_names
from knn.run_knn import CFG as KNN_CFG, KNNTrack
from lda_regularized.run_lda_regularized import CFG as LDA_CFG, LDATrack
from random_forest.run_rf import CFG as RF_CFG, RFTrack
from svm_rbf.run_svm_rbf import CFG as SVM_CFG, SVMTrack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=SRC.parent / "NinaPro_Mat")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    # Preserve the original training row order even when path sorting differs by OS.
    reference = pd.read_csv(BASELINE / "new_subject/predictions.csv").sort_values("source_window")
    subject_order = list(reference.subject.unique())
    track = RFTrack("full_feature")
    records = track.load(args.data, exercise="E1")
    track.assert_no_holdout(records)
    if len(records) != 22 or {r.group for r in records} != set(subject_order):
        raise ValueError("Expected one development recording per subject")
    records.sort(key=lambda r: subject_order.index(r.group))
    expanded, labels, subjects, repetitions = track._build(records, RF_CFG)
    baseline = RFTrack("baseline")._build(records, RF_CFG)[0]
    for actual, expected in ((subjects, reference.subject), (repetitions, reference.repetition),
                             (labels, reference.y_true), (np.arange(len(labels)), reference.source_window)):
        np.testing.assert_array_equal(actual, expected)
    modes = development_folds(subjects, repetitions)
    sizes = {"full_feature": 70, "time": 40, "amplitude": 20, "baseline": 50, "full_60": 60}
    families = np.array([n.split("_", 1)[1] for n in expanded_feature_names()])

    def run_mode(model_track, cfg, X, mode, path, pool):
        def predict(split):
            train, test = split
            selector = model_track.select_features(X[train], labels[train], cfg)
            model = model_track._baseline(cfg)
            model.fit(selector.transform(X[train]), labels[train])
            return test, model.predict(selector.transform(X[test]))

        predictions = np.empty_like(labels)
        folds = np.full(len(labels), -1)
        for fold, (test, predicted) in enumerate(pool.map(predict, modes[mode])):
            folds[test] = fold
            predictions[test] = predicted
        frame = pd.DataFrame({"source_window": np.arange(len(labels)), "subject": subjects,
                              "repetition": repetitions, "fold": folds,
                              "y_true": labels, "y_pred": predictions})
        frame.to_csv(path / f"{mode}.csv.gz", index=False,
                     compression={"method": "gzip", "mtime": 0})
        print(f"Saved {path.name}/{mode}", flush=True)

    configs = (("rf", RFTrack, RF_CFG), ("svm", SVMTrack, SVM_CFG),
               ("lda", LDATrack, LDA_CFG), ("knn", KNNTrack, KNN_CFG))
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for name, track_class, cfg in configs:
            domains = ["full_feature", "time", "amplitude"]
            if name == "rf":
                domains.append("full_60")
            else:
                domains.insert(0, "baseline")
            for domain in domains:
                model_track = track_class("full_feature" if domain == "full_60" else domain)
                if domain == "full_60":
                    X = expanded[:, families != "RMS"]
                elif domain == "baseline":
                    X = baseline
                else:
                    X = expanded[:, model_track.columns]
                if X.shape != (len(reference), sizes[domain]):
                    raise ValueError(f"{domain}: unexpected feature dimensions {X.shape}")
                path = args.output / f"{name}_{domain}"
                path.mkdir()
                print(f"Fitting {path.name}", flush=True)
                for mode in (["new_subject"] if domain == "full_60" else modes):
                    run_mode(model_track, cfg, X, mode, path, pool)


if __name__ == "__main__":
    main()
