import sys
sys.path.insert(0, "")

from emg_ninapro import EMGNinaproTrack

track = EMGNinaproTrack()
records = track.load("NinaPro_Mat", exercise="E1")
records = [r for r in records
           if r.group in {"S1", "S2", "S3", "S4", "S5"}]

results = track.evaluate_modes(records, cfg={"seed": 0})

for mode, result in results.items():
    print(mode, result["summary"])
    print("Pooled macro-F1:", result["macro_f1"])
    track.report(result)