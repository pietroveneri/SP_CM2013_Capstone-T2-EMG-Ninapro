"""Check saved preprocessing predictions against their reports and baseline folds."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING = ROOT / "docs" / "Preprocessing"
ARCHIVES = (
    PREPROCESSING / "fold_safe_20261028" / "zscore",
    PREPROCESSING / "smooth_20261028" / "median_savgol_zscore",
)
FOLDS = ROOT / "docs" / "Baseline" / "baseline_20261022_holdout_separated" / "folds.json"


def verify():
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    fold_map = {(row["mode"], row["fold"]): row for row in folds}
    reference = None
    for archive in ARCHIVES:
        for mode in ("within_subject", "new_subject"):
            path = archive / mode
            predictions = pd.read_csv(path / "predictions.csv")
            indexed = pd.read_csv(path / "indexed_predictions.csv")
            pd.testing.assert_frame_equal(predictions, indexed[predictions.columns])
            assert len(indexed) == 60182
            assert sorted(indexed["source_window"]) == list(range(len(indexed)))
            assert set(indexed["subject"]) == {f"S{i}" for i in range(1, 23)}
            assert set(indexed["fold"]) == {row["fold"] for row in folds if row["mode"] == mode}

            identity = indexed.sort_values("source_window")[["source_window", "subject", "repetition", "y_true"]].reset_index(drop=True)
            if reference is None:
                reference = identity
            else:
                pd.testing.assert_frame_equal(identity, reference)

            per_subject = json.loads((path / "per_subject.json").read_text(encoding="utf-8"))
            per_fold = json.loads((path / "per_fold.json").read_text(encoding="utf-8"))
            for rows, saved, column, name in (
                (indexed, per_subject, "subject", "group"),
                (indexed, per_fold, "fold", "fold"),
            ):
                groups = dict(tuple(rows.groupby(column)))
                assert len(groups) == len(saved)
                for item in saved:
                    group = groups[item[name]]
                    score = f1_score(group["y_true"], group["y_pred"], average="macro")
                    assert len(group) == item["n"]
                    np.testing.assert_allclose(score, item["macro_f1"], rtol=0, atol=1e-12)

            for fold, group in indexed.groupby("fold"):
                split = fold_map[(mode, int(fold))]
                assert len(group) == split["test_n"]
                assert set(group["subject"]) == set(split["test_groups"])
                if mode == "within_subject":
                    assert set(group["repetition"]) == set(split["test_repetitions"])
                    assert not set(group["repetition"]) & set(split["train_repetitions"])
                else:
                    assert not set(group["subject"]) & set(split["train_groups"])

            scores = [item["macro_f1"] for item in per_subject]
            print(f"{archive.name} {mode}: {np.mean(scores):.3f} ± {np.std(scores, ddof=1):.3f}")
            if mode == "new_subject":
                scores = [item["macro_f1"] for item in per_fold]
                print(f"{archive.name} new_subject folds: {np.mean(scores):.3f} ± {np.std(scores, ddof=1):.3f}")


if __name__ == "__main__":
    verify()
