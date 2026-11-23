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

from classifiers import MODELS
from development_folds import BASELINE, development_folds, load_development_records
from features import feature_columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=SRC.parent / "NinaPro_Mat")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    # Preserve the original training row order even when path sorting differs by OS.
    reference = pd.read_csv(BASELINE / "new_subject/predictions.csv").sort_values("source_window")
    track = MODELS["rf"]("full_70")
    records = load_development_records(track, args.data)
    expanded, labels, subjects, repetitions = track._build(records, track.cfg)
    baseline_track = MODELS["rf"]("baseline")
    baseline = baseline_track._build(records, baseline_track.cfg)[0]
    for actual, expected in ((subjects, reference.subject), (repetitions, reference.repetition),
                             (labels, reference.y_true), (np.arange(len(labels)), reference.source_window)):
        np.testing.assert_array_equal(actual, expected)
    modes = development_folds(subjects, repetitions)
    sizes = {"full_feature": 70, "time": 40, "amplitude": 20, "baseline": 50, "full_60": 60}
    archive_sets = {"full_feature": "full_70", "time": "time_40", "amplitude": "amplitude_20",
                    "baseline": "baseline", "full_60": "full_60"}

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

    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for name, track_class in MODELS.items():
            domains = ["full_feature", "time", "amplitude"]
            if name == "rf":
                domains.append("full_60")
            else:
                domains.insert(0, "baseline")
            for domain in domains:
                feature_set = archive_sets[domain]
                model_track = track_class(feature_set)
                X = baseline if domain == "baseline" else expanded[:, feature_columns(feature_set)]
                if X.shape != (len(reference), sizes[domain]):
                    raise ValueError(f"{domain}: unexpected feature dimensions {X.shape}")
                path = args.output / f"{name}_{domain}"
                path.mkdir()
                print(f"Fitting {path.name}", flush=True)
                for mode in (["new_subject"] if domain == "full_60" else modes):
                    run_mode(model_track, model_track.cfg, X, mode, path, pool)


if __name__ == "__main__":
    main()
