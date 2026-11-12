"""Shared runner for paired subject macro-F1 comparisons against the saved baseline."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from emg_ninapro import FS, N_CH, STEP, WIN


ROOT = Path(__file__).resolve().parents[2]
SUBJECTS = tuple(f"S{i}" for i in range(1, 23))
MODES = ("within_subject", "new_subject")
CFG = {
    "seed": 0,
    "select": "none",
    "imbalance": "balanced",
    "cv_max_splits": 5,
    "loso_max_groups": 12,
}


def _index_scores(rows, source):
    scores = {row["group"]: row for row in rows}
    if len(scores) != len(rows) or set(scores) != set(SUBJECTS):
        raise ValueError(f"{source}: expected exactly one score for each of S1-S22.")
    if any(not math.isfinite(row["macro_f1"]) for row in rows):
        raise ValueError(f"{source}: a subject has an undefined macro-F1.")
    return scores


def _load_baseline(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config["requested"] != CFG:
        raise ValueError("Baseline configuration differs from the ablation configuration.")
    signal = config["fixed_signal"]
    expected = {"fs": FS, "window_samples": WIN, "step_samples": STEP, "channels": N_CH}
    if any(signal[key] != value for key, value in expected.items()):
        raise ValueError("Baseline sampling rate, window, step or channel count differs.")

    baseline = {}
    for mode in MODES:
        rows = json.loads((directory / mode / "per_group.json").read_text(encoding="utf-8"))
        scores = _index_scores(rows, f"baseline/{mode}")
        predictions = {subject: [] for subject in SUBJECTS}
        with (directory / mode / "predictions.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                predictions[row["subject"]].append((row["y_true"], int(row["fold"])))
        for subject in SUBJECTS:
            if len(predictions[subject]) != scores[subject]["n"]:
                raise ValueError(f"Baseline {mode}/{subject}: score and prediction counts differ.")
        baseline[mode] = (scores, predictions)
    return baseline


def _compare(result, baseline, mode):
    baseline_scores, baseline_predictions = baseline
    variant_scores = _index_scores(result["per_group"], f"variant/{mode}")
    variant_predictions = {subject: [] for subject in SUBJECTS}
    for subject, label, fold in zip(
        result["y_group"], result["y_true"], result["y_fold"], strict=True
    ):
        variant_predictions[str(subject)].append((str(label), int(fold)))

    rows = []
    for subject in SUBJECTS:
        # Compare each subject separately; the order of subjects in the files may differ.
        # Within a subject, labels and fold membership must agree window by window.
        if variant_predictions[subject] != baseline_predictions[subject]:
            raise ValueError(f"{mode}/{subject}: baseline and variant labels or folds differ.")
        base = baseline_scores[subject]
        variant = variant_scores[subject]
        if variant["n"] != base["n"]:
            raise ValueError(f"{mode}/{subject}: baseline and variant sample counts differ.")
        rows.append({
            "subject": subject,
            "n": int(base["n"]),
            "macro_f1_baseline": float(base["macro_f1"]),
            "macro_f1_variant": float(variant["macro_f1"]),
            "delta_f1": float(variant["macro_f1"] - base["macro_f1"]),
        })
    return rows


def run_comparison(track, variant_name):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir", type=Path,
        default=ROOT / "docs" / "Baseline" / "baseline_20261022_holdout_separated",
        help="Baseline archive containing config.json and both modes' scores/predictions.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "docs" / "Baseline" / f"ablation_{variant_name}_vs_baseline",
        help="Destination for the two subject CSVs and comparison.md.",
    )
    args = parser.parse_args()
    baseline = _load_baseline(args.baseline_dir)

    track.download(ROOT / "NinaPro_Mat", subset=range(1, 23))
    records = track.load(ROOT / "NinaPro_Mat", exercise="E1")
    track.assert_no_holdout(records)
    if {rec.group for rec in records} != set(SUBJECTS):
        raise ValueError("Expected recordings for all 22 development subjects, S1-S22.")

    print(f"Evaluating {variant_name}; reference baseline: {args.baseline_dir}")
    results = track.evaluate_modes(records, cfg=dict(CFG))
    # Validate both comparisons before writing any results.
    comparisons = {mode: _compare(results[mode], baseline[mode], mode) for mode in MODES}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = [
        f"# Subject macro-F1: {variant_name} vs baseline", "",
        f"Baseline archive: `{args.baseline_dir.resolve()}`", "",
        f"Configuration: `{json.dumps(CFG, sort_keys=True)}`", "",
        "Exercise E1, S1-S22, 12 gestures. Macro-F1 is computed from each subject's "
        "out-of-fold predictions across all its test windows.", "",
        "within_subject: 5-fold GroupKFold on repetitions per subject. "
        "new_subject: 5-fold GroupKFold on subjects.", "",
        "delta_f1 = macro_f1_variant - macro_f1_baseline, on the 0-1 F1 scale. "
        "Positive values indicate an improvement after removing the feature.", "",
    ]
    for mode, rows in comparisons.items():
        destination = args.output_dir / f"delta_f1_{mode}.csv"
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        deltas = [row["delta_f1"] for row in rows]
        section = [
            f"## {mode}", "",
            f"Mean subject delta: {mean(deltas):+.6f}. "
            f"Improved: {sum(d > 0 for d in deltas)}; "
            f"worsened: {sum(d < 0 for d in deltas)}; "
            f"unchanged: {sum(d == 0 for d in deltas)}.", "",
            "| Subject | Windows | Baseline macro-F1 | Variant macro-F1 | Delta F1 |",
            "|---|---:|---:|---:|---:|",
        ]
        section.extend(
            f"| {r['subject']} | {r['n']} | {r['macro_f1_baseline']:.6f} | "
            f"{r['macro_f1_variant']:.6f} | {r['delta_f1']:+.6f} |"
            for r in rows
        )
        report.extend(section + [""])
        print(f"\n{variant_name}: {mode}, {results[mode]['summary']}")
        track.report(results[mode])
        print("\n".join(section))
        print(f"Saved: {destination}")

    (args.output_dir / "comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")

