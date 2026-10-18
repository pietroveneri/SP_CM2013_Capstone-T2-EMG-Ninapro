from pathlib import Path
from experiments import ExpandedFeatureTrack

cfg = {
    "seed": 0,
    "select": "none",
    "imbalance": "balanced",
    "cv_max_splits": 5,
    "loso_max_groups": 12,
}

root = Path(__file__).resolve().parents[1]

# Uses the feature extraction defined in ExpandedFeatureTrack.
track = ExpandedFeatureTrack()
records = track.load(root / "data" / "raw", exercise="E1")

if not records:
    raise RuntimeError("No recordings found. Check the data path.")

# Check feature extraction before running the full evaluation.
X, y, subject = track.extract_features(records[0])

print("Subject:", subject)
print("Feature matrix:", X.shape)
print("Labels:", y.shape)

results = track.evaluate_modes(records, cfg=cfg)

for mode, result in results.items():
    print(mode, result["summary"])
    print("Pooled macro-F1:", result["macro_f1"])
    track.report(result)