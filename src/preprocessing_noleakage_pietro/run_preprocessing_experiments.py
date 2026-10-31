"""Retired runner for historical fold-fitted normalization experiments.

The shared reporting helpers remain in use by the no-normalization runner.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy
import sklearn

from fold_safe_preprocessing import FoldSafePreprocessingTrack


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "NinaPro_Mat"
SUBJECTS = {f"S{i}" for i in range(1, 23)}
CFG = {"seed": 0, "select": "none", "imbalance": "balanced",
       "cv_max_splits": 5, "loso_max_groups": 12}
OPTIONS = {2: "zscore", 5: "peak", 3: "median_savgol_zscore"}
METRICS = ("macro_f1", "cohens_kappa", "balanced_accuracy", "accuracy")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [plain(v) for v in obj]
    return obj


def write_json(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def spread(rows):
    return {
        metric: {"mean": float(np.mean(values)),
                 "sample_sd": float(np.std(values, ddof=1)),
                 "min": float(np.min(values)), "max": float(np.max(values)),
                 "n": len(values)}
        for metric in METRICS
        for values in [[float(row[metric]) for row in rows]]
    }


def save_result(track, result, mode, output, expected_counts):
    y_true = np.asarray(result["y_true"])
    y_pred = np.asarray(result["y_pred"])
    groups = np.asarray(result["y_group"])
    folds = np.asarray(result["y_fold"])
    count = sum(expected_counts.values())
    if not (len(y_true) == len(y_pred) == len(groups) == len(folds) == count):
        raise ValueError(f"{mode}: inconsistent prediction lengths")
    if set(groups.tolist()) != SUBJECTS:
        raise ValueError(f"{mode}: wrong development subjects")
    if any(np.count_nonzero(groups == subject) != n for subject, n in expected_counts.items()):
        raise ValueError(f"{mode}: incorrect number of predictions per subject")
    expected_folds = 110 if mode == "within_subject" else 5
    if len(np.unique(folds)) != expected_folds:
        raise ValueError(f"{mode}: expected {expected_folds} folds")
    output.mkdir(parents=True)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject", "fold", "y_true", "y_pred"])
        writer.writerows(zip(groups, folds, y_true, y_pred, strict=True))
    write_json(output / "per_subject.json", result["per_group"])
    write_json(output / "per_fold.json", result["per_fold"])
    panel = track.report(result, show=False)
    (output / "report.md").write_text(panel["text"] + "\n", encoding="utf-8")
    return {"split_unit": result["split_unit"], "n_predictions": count,
            "n_subjects": len(SUBJECTS), "n_folds": expected_folds,
            "pooled": panel["pooled"], "per_subject_spread": spread(result["per_group"]),
            "per_fold_spread": spread(result["per_fold"])}


def main():
    raise RuntimeError(
        "Retired: this runner describes signal normalization, which is no longer "
        "part of the current preprocessing. See docs/FOLD_SAFE_PREPROCESSING.md."
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    allowed = (ROOT / "docs" / "Preprocessing").resolve()
    if output == allowed or allowed not in output.parents:
        raise ValueError("Output must be a new directory under docs/Preprocessing")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    # Hash exactly the 22 permitted MAT files, without scanning/reading hold-out.
    paths = [DATA / f"s{i}" / f"S{i}_A1_E1.mat" for i in range(1, 23)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing development MAT files: {missing}")
    data_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    track = FoldSafePreprocessingTrack(option=2, guard=0.05)
    records = track.load(DATA, exercise="E1")
    track.assert_no_holdout(records)
    if len(records) != 22 or {r.group for r in records} != SUBJECTS:
        raise ValueError("Expected exactly one recording per development subject S1-S22")
    counts = {r.group: len(r.labels) for r in records}

    output.mkdir(parents=True)
    sources = [Path(__file__), ROOT / "src" / "preprocessing_noleakage_pietro" / "fold_safe_preprocessing.py",
               ROOT / "src" / "emg_ninapro.py", ROOT / "src" / "adapter.py",
               ROOT / "src" / "report.py", ROOT / "src" / "bsp" / "metrics.py"]
    provenance = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                  "config": CFG, "options": OPTIONS, "peak_guard": 0.05,
                  "data_sha256": data_hashes,
                  "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
                  "versions": {"python": sys.version, "platform": platform.platform(),
                               "numpy": np.__version__, "scipy": scipy.__version__,
                               "scikit_learn": sklearn.__version__},
                  "subjects": sorted(SUBJECTS), "counts": counts,
                  "holdout_scored": False}
    write_json(output / "run.json", provenance)
    summaries = {}
    for option, name in OPTIONS.items():
        print(f"Evaluating option {option}: {name}", flush=True)
        track = FoldSafePreprocessingTrack(option=option, guard=0.05)
        results = track.evaluate_modes(records, cfg=dict(CFG))
        summaries[name] = {
            mode: save_result(track, results[mode], mode, output / name / mode, counts)
            for mode in ("within_subject", "new_subject")
        }
        write_json(output / "summary.json", summaries)
    provenance.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "run.json", provenance)


if __name__ == "__main__":
    main()
