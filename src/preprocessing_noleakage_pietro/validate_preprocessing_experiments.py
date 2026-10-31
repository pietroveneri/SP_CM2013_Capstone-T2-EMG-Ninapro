"""Verify and index a completed fold-safe preprocessing experiment archive."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from fold_safe_preprocessing import FoldSafePreprocessingTrack


ROOT = Path(__file__).resolve().parents[2]
SUBJECTS = {f"S{i}" for i in range(1, 23)}
MODES = ("within_subject", "new_subject")


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    allowed = (ROOT / "docs" / "Preprocessing").resolve()
    if allowed not in archive.parents:
        raise ValueError("Archive must be under docs/Preprocessing")
    run = json.loads((archive / "run.json").read_text(encoding="utf-8"))
    if run["status"] not in ("complete", "partial"):
        raise ValueError("The run has not completed")
    summary = json.loads((archive / "summary.json").read_text(encoding="utf-8"))
    options = list(summary)
    if not options or not set(options).issubset(set(run["options"].values())):
        raise ValueError("Saved options do not match run provenance")
    if run["status"] == "complete" and set(options) != set(run["options"].values()):
        raise ValueError("A completed run must contain every requested option")
    for relpath, expected in {**run["data_sha256"], **run["source_sha256"]}.items():
        if digest(ROOT / relpath) != expected:
            raise ValueError(f"Input or source changed since execution: {relpath}")

    track = FoldSafePreprocessingTrack(option=2)
    recs = track.load(ROOT / "NinaPro_Mat", exercise="E1")
    track.assert_no_holdout(recs)
    if len(recs) != 22 or {r.group for r in recs} != SUBJECTS:
        raise ValueError("Unexpected cohort")
    labels = np.concatenate([r.labels for r in recs])
    groups = np.concatenate([np.repeat(r.group, len(r.labels)) for r in recs])
    reps = np.concatenate([r.meta["rep"] for r in recs])
    sample = np.arange(len(labels))

    def within_folds():
        for subject in sorted(SUBJECTS):
            ids = np.flatnonzero(groups == subject)
            for train, test in GroupKFold(n_splits=5).split(ids, labels[ids], reps[ids]):
                yield ids[train], ids[test]

    folds = {
        "within_subject": list(within_folds()),
        "new_subject": list(GroupKFold(n_splits=5).split(sample, labels, groups)),
    }
    fold_records = []
    for mode, splits in folds.items():
        test_ids = np.concatenate([test for _, test in splits])
        if not np.array_equal(np.sort(test_ids), sample):
            raise ValueError(f"{mode}: not exactly one test prediction per source window")
        for fold, (train, test) in enumerate(splits):
            if np.intersect1d(train, test).size:
                raise ValueError(f"{mode}: train/test windows overlap")
            train_subjects, test_subjects = set(groups[train]), set(groups[test])
            if mode == "new_subject" and train_subjects & test_subjects:
                raise ValueError("Subject leakage")
            train_reps, test_reps = set(reps[train]), set(reps[test])
            if mode == "within_subject" and (len(train_subjects | test_subjects) != 1 or train_reps & test_reps):
                raise ValueError("Repetition leakage")
            fold_records.append({"mode": mode, "fold": fold,
                                 "train_n": len(train), "test_n": len(test),
                                 "train_subjects": sorted(train_subjects), "test_subjects": sorted(test_subjects),
                                 "train_repetitions": sorted(map(int, train_reps)) if mode == "within_subject" else None,
                                 "test_repetitions": sorted(map(int, test_reps)) if mode == "within_subject" else None})

        for option in options:
            path = archive / option / mode / "predictions.csv"
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != len(test_ids):
                raise ValueError(f"{option}/{mode}: incorrect number of predictions")
            fold_ids = np.concatenate([np.repeat(i, len(test)) for i, (_, test) in enumerate(splits)])
            indexed_path = path.with_name("indexed_predictions.csv")
            if indexed_path.exists():
                raise FileExistsError(indexed_path)
            with indexed_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["source_window", "subject", "repetition", "fold", "y_true", "y_pred"])
                for row, index, fold in zip(rows, test_ids, fold_ids, strict=True):
                    if (row["subject"] != groups[index] or int(row["fold"]) != fold
                            or row["y_true"] != labels[index]):
                        raise ValueError(f"{option}/{mode}: predictions do not match reconstructed folds")
                    writer.writerow([index, row["subject"], int(reps[index]), fold, row["y_true"], row["y_pred"]])

    (archive / "folds.json").write_text(json.dumps(fold_records, indent=2) + "\n", encoding="utf-8")
    validation = {"source_and_data_hashes_match": True,
                  "loader_excludes_reserved_subjects_by_default": True,
                  "holdout_reads_instrumented": False,
                  "one_test_prediction_per_window": True, "test_predictions_match_fold_assignment": True,
                  "train_test_split_disjoint": True, "windows_per_mode": len(sample)}
    (archive / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
