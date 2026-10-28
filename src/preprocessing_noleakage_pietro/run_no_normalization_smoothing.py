"""Evaluate median/Savitzky-Golay preprocessing without signal normalization.

Example: python src/run_no_normalization_smoothing.py --output-dir docs/Preprocessing/new-run
The destination must be new; only S1-S22 are used and no data are downloaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy
import sklearn

from fold_safe_preprocessing import FoldSafePreprocessingTrack
from run_preprocessing_experiments import CFG, DATA, ROOT, SUBJECTS, save_result, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir.resolve()
    allowed = (ROOT / "docs" / "Preprocessing").resolve()
    if output == allowed or allowed not in output.parents:
        raise ValueError("Output must be a new directory under docs/Preprocessing")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    paths = [DATA / f"s{i}" / f"S{i}_A1_E1.mat" for i in range(1, 23)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing development MAT files: {missing}")
    data_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    track = FoldSafePreprocessingTrack(option=3)
    records = track.load(DATA, exercise="E1")
    track.assert_no_holdout(records)
    if len(records) != 22 or {r.group for r in records} != SUBJECTS:
        raise ValueError("Expected exactly one recording per development subject S1-S22")
    counts = {r.group: len(r.labels) for r in records}
    output.mkdir(parents=True)
    sources = [Path(__file__), ROOT / "src" / "run_preprocessing_experiments.py",
               ROOT / "src" / "fold_safe_preprocessing.py", ROOT / "src" / "emg_ninapro.py",
               ROOT / "src" / "adapter.py", ROOT / "src" / "report.py",
               ROOT / "src" / "bsp" / "metrics.py"]
    name = "median_savgol_no_normalization"
    provenance = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": CFG, "options": {3: name},
        "signal_preprocessing": "median filter length 3, Savitzky-Golay length 5 order 2; no signal normalization",
        "feature_scaler": "StandardScaler fitted by the classifier pipeline on each training fold",
        "data_sha256": data_hashes,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
        "versions": {"python": sys.version, "platform": platform.platform(),
                     "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "subjects": sorted(SUBJECTS), "counts": counts, "holdout_scored": False,
    }
    write_json(output / "run.json", provenance)
    print(f"Evaluating {name}", flush=True)
    results = track.evaluate_modes(records, cfg=dict(CFG))
    summaries = {name: {
        mode: save_result(track, results[mode], mode, output / name / mode, counts)
        for mode in ("within_subject", "new_subject")
    }}
    write_json(output / "summary.json", summaries)
    provenance.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "run.json", provenance)


if __name__ == "__main__":
    main()
