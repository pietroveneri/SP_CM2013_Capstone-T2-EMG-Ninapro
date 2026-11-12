# File: src/compare_options.py
# This script compares the performance of two different preprocessing options (02 and 03) on the NinaPro E1 dataset.

from pathlib import Path

from sklearn.metrics import f1_score

from filter_option02 import FilterOption02
from filter_option03 import FilterOption03
from features import extract_window_features


# Both options use exactly the same expanded features.
class Option02Expanded(FilterOption02):
    def extract_features(self, rec, cfg=None):
        return extract_window_features(rec)


class Option03Expanded(FilterOption03):
    def extract_features(self, rec, cfg=None):
        return extract_window_features(rec)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "NinaPro_Mat"  # Change if needed.

options = {
    "02": Option02Expanded(),
    "03": Option03Expanded(),
}

# Load the same development recordings once.
loader = options["02"]
print(f"Loading recordings from {DATA_DIR}...", flush=True)
records = loader.load(DATA_DIR, exercise="E1")
print(f"Loaded {len(records)} recordings.", flush=True)
loader.assert_no_holdout(records)

if not records:
    raise RuntimeError(f"No recordings found in {DATA_DIR}")

# Identical evaluation settings for both recipes.
evaluation_cfg = {
    "seed": 0,
    "imbalance": "balanced",
    "select": "none",
    "cv_max_splits": 5,
    "loso_max_groups": 12,
}

scores = {}

for option_name, track in options.items():
    print(f"\nEvaluating option {option_name}...")

    reports = track.evaluate_modes(records, cfg=evaluation_cfg)
    scores[option_name] = {}

    for mode, report in reports.items():
        score = f1_score(
            report["y_true"],
            report["y_pred"],
            average="macro",
        )
        scores[option_name][mode] = score

        print(f"{mode}:")
        print(f"  Pooled macro-F1: {score:.4f}")
        print(f"  Cohen's kappa:   {report['cohens_kappa']:.4f}")
        print(f"  Subject spread:  {report['summary']}")

print("\nDifference in pooled macro-F1: option 02 minus 03")
for mode in scores["02"]:
    difference = scores["02"][mode] - scores["03"][mode]
    print(f"  {mode}: {difference:+.4f}")
