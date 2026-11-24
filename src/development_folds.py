"""Map archived DB1 development folds to the caller's window order."""
import json
from pathlib import Path

import numpy as np


BASELINE = Path(__file__).resolve().parents[1] / "docs/Baseline/baseline_20261022_holdout_separated"


def load_development_records(track, data):
    """Keep the archived training row order across operating systems and cache layouts."""
    records = track.load(data, exercise="E1")
    track.assert_no_holdout(records)
    if len(records) != 22 or {r.group for r in records} != {f"S{i}" for i in range(1, 23)}:
        raise ValueError("Expected one E1 recording for each subject S1-S22 in " + str(data))
    # The original Windows run sorted MAT filenames, e.g. S10_A1_E1.mat before S1_A1_E1.mat.
    return sorted(records, key=lambda r: f"{r.group}_A1_E1.mat")


def development_folds(subjects, repetitions):
    subjects, repetitions = np.asarray(subjects), np.asarray(repetitions)
    expected = {f"S{i}" for i in range(1, 23)}
    if set(subjects) != expected or subjects.shape != repetitions.shape:
        raise ValueError("Expected aligned subject and repetition arrays for S1–S22")
    archive = json.loads((BASELINE / "folds.json").read_text())
    modes = {}
    for mode, count in (("new_subject", 5), ("within_subject", 110)):
        rows = sorted((f for f in archive if f["mode"] == mode), key=lambda f: f["fold"])
        if [f["fold"] for f in rows] != list(range(count)):
            raise ValueError(f"{mode}: missing or repeated fold")
        splits = []
        coverage = np.zeros(len(subjects), dtype=int)
        for f in rows:
            train = np.isin(subjects, f["train_groups"])
            test = np.isin(subjects, f["test_groups"])
            if mode == "within_subject":
                train &= np.isin(repetitions, f["train_repetitions"])
                test &= np.isin(repetitions, f["test_repetitions"])
            if (np.any(train & test) or train.sum() != f["train_n"]
                    or test.sum() != f["test_n"]):
                raise ValueError(f"{mode} fold {f['fold']}: invalid split")
            coverage += test
            splits.append((np.flatnonzero(train), np.flatnonzero(test)))
        if not np.all(coverage == 1):
            raise ValueError(f"{mode}: expected one test prediction per window")
        modes[mode] = splits
    return modes
