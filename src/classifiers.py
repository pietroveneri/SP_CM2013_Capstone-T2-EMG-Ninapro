"""Run one classical classifier on DB1 E1 development subjects S1-S22."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np
import scipy
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from adapter import metric_spread
from classifier_evaluation import configure_imbalance, evaluate_modes_progress
from development_folds import BASELINE, load_development_records
from features import FEATURE_SETS, FeatureTrack
from preprocessing import PREPROCESSING
from report import plot_classifier_results


ROOT = Path(__file__).resolve().parents[1]
CV_CONFIG = {"seed": 0, "select": "none", "cv_max_splits": 5, "loso_max_groups": 12}


class ClassifierTrack(FeatureTrack):
    """Defaults shared by the classical tracks, including the supplied RF baseline."""
    imbalance = "balanced"
    MODEL_CFG_KEYS = frozenset()

    def __init__(self, feature_set, cfg=None):
        super().__init__(feature_set, {**CV_CONFIG, "imbalance": self.imbalance, **(cfg or {})})

    @classmethod
    def supported_cfg_keys(cls):
        return super().supported_cfg_keys() | cls.MODEL_CFG_KEYS

    def feature_cfg_keys(self):
        return super().feature_cfg_keys() - self.MODEL_CFG_KEYS


class LDATrack(ClassifierTrack):
    imbalance = "none"
    MODEL_CFG_KEYS = frozenset({"shrinkage"})

    def baseline(self, cfg=None):
        cfg = self._cfg(cfg)
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=cfg.get("shrinkage", "auto"))
        return configure_imbalance(make_pipeline(StandardScaler(), model), cfg)


class KNNTrack(ClassifierTrack):
    imbalance = "none"
    MODEL_CFG_KEYS = frozenset({"n_neighbors"})

    def baseline(self, cfg=None):
        cfg = self._cfg(cfg)
        model = KNeighborsClassifier(n_neighbors=cfg.get("n_neighbors", 5))
        return configure_imbalance(make_pipeline(StandardScaler(), model), cfg)


class SVMTrack(ClassifierTrack):
    MODEL_CFG_KEYS = frozenset({"C", "gamma"})

    def baseline(self, cfg=None):
        cfg = self._cfg(cfg)
        mode = cfg["imbalance"]
        if mode not in ("none", "balanced"):
            raise ValueError(f"SVM supports imbalance='none' or 'balanced', got {mode!r}")
        model = SVC(kernel="rbf", C=cfg.get("C", 1.0), gamma=cfg.get("gamma", "scale"),
                    class_weight="balanced" if mode == "balanced" else None,
                    random_state=cfg["seed"])
        return make_pipeline(StandardScaler(), model)


# ClassifierTrack inherits the course RF from EMGNinaproTrack.
MODELS = {"rf": ClassifierTrack, "lda": LDATrack, "knn": KNNTrack, "svm": SVMTrack}


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_report(track, result, mode, output):
    output.mkdir()
    columns = ("source_window", "y_group", "repetition", "y_fold", "y_true", "y_pred")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source_window", "subject", "repetition", "fold", "y_true", "y_pred"))
        writer.writerows(zip(*(result[k] for k in columns), strict=True))
    title = "New-subject · No calibration" if mode == "new_subject" else "Within-subject"
    panel = track.report(result, title=title, show=False)
    (output / "report.md").write_text(panel["text"] + "\n", encoding="utf-8")
    summary = {"split_unit": result["split_unit"], "n_predictions": len(result["y_pred"]),
               "n_subjects": len(result["per_group"]), "n_folds": len(result["per_fold"]),
               "pooled": panel["pooled"]}
    for unit, key in (("subject", "per_group"), ("fold", "per_fold")):
        rows = result[key]
        write_json(output / f"per_{unit}.json", rows)
        spreads = {}
        for metric in ("macro_f1", "cohens_kappa", "balanced_accuracy", "accuracy"):
            spread = metric_spread(rows, metric)
            spread["sample_sd"] = spread.pop("std")
            spreads[metric] = spread
        summary[f"per_{unit}_spread"] = spreads
        f1 = spreads["macro_f1"]
        print(f"{title}: macro-F1 {f1['mean']:.3f} ± {f1['sample_sd']:.3f} across {f1['n']} {unit}s")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--features", choices=("baseline", *FEATURE_SETS), required=True)
    parser.add_argument("--preprocessing", choices=PREPROCESSING, default="none")
    parser.add_argument("--output-dir", type=Path, help="Fresh directory (default: outputs/<model>_<features>_<preprocessing>)")
    args = parser.parse_args()
    name = f"{args.model}_{args.features}_{args.preprocessing}"
    output = args.output_dir or ROOT / "outputs" / name
    if output.exists():
        parser.error(f"{output} already exists. Choose a fresh --output-dir.")
    track = MODELS[args.model](args.features)
    data = ROOT / "NinaPro_Mat"
    records = load_development_records(track, data)
    paths = sorted(p for i in range(1, 23) for p in data.rglob(f"S{i}_A1_E1.mat"))
    sources = [Path(__file__), *(ROOT / "src" / p for p in (
        "classifier_evaluation.py", "preprocessing.py", "features.py", "development_folds.py",
        "adapter.py", "emg_ninapro.py", "report.py", "bsp/metrics.py", "bsp/notebook_checks.py"))]
    metadata = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "features": args.features, "preprocessing": args.preprocessing,
        "new_subject_setting": "No calibration", "config": track.cfg,
        "subjects": [r.group for r in records], "counts": {r.group: len(r.labels) for r in records},
        "data_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths},
        "source_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sources},
        "folds_sha256": file_hash(BASELINE / "folds.json"),
        "versions": {"python": platform.python_version(), "platform": platform.platform(),
                     "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
    }
    if shutil.which("git"):
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
        metadata["git_head"] = git.stdout.strip() if git.returncode == 0 else None
    output.mkdir(parents=True)
    write_json(output / "run.json", metadata)
    print(f"Running {name} on {len(records)} development subjects...", flush=True)
    results = evaluate_modes_progress(track, records, track.cfg, preprocessing=args.preprocessing, archived_folds=True)
    summary = {mode: save_report(track, result, mode, output / mode) for mode, result in results.items()}
    write_json(output / "summary.json", summary)
    plot_classifier_results(results, name, output)
    metadata.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "run.json", metadata)
    print(f"Saved predictions, reports and plots to {output}")


if __name__ == "__main__":
    main()
