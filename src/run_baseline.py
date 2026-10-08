import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]

from emg_ninapro import EMGNinaproTrack

track = EMGNinaproTrack()
records = track.load(root / "NinaPro_Mat", exercise="E1")

results = track.evaluate_modes(records, cfg={"seed": 0})

for mode, result in results.items():
    print(mode, result["summary"])
    # print("Pooled macro-F1:", result["macro_f1"])
    track.report(result)