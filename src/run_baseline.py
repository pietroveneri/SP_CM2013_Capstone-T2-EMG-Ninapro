import sys
from pathlib import Path

cfg = {
    "seed": 0,
    "select": "none",
    "imbalance": "balanced",
    "cv_max_splits": 5,
    "loso_max_groups": 12
}
# E1, WIN = 20, STEP = 15, 50 features and RF @ 200 trees.

root = Path(__file__).resolve().parents[1]

from emg_ninapro import EMGNinaproTrack

track = EMGNinaproTrack()
records = track.load(root / "NinaPro_Mat", exercise="E1")

results = track.evaluate_modes(records, cfg={"seed": 0})

for mode, result in results.items():
    print(mode, result["summary"])
    # print("Pooled macro-F1:", result["macro_f1"])
    track.report(result)