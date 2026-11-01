import sys
from pathlib import Path
from time import perf_counter

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import ExpandedFeatureTrack
from emg_ninapro import EMGNinaproTrack
from features import expanded_feature_names
from classifier_evaluation import evaluate_modes_progress
from knn.knn_plot_results import save_plots


ROOT = Path(__file__).resolve().parents[2]
CFG = {"seed": 0, "select": "none", "imbalance": "none",
       "cv_max_splits": 5, "loso_max_groups": 12}
DOMAINS = {
    "baseline": None,
    "full_feature": {"MAV", "WL", "VAR", "relative_MAV", "mean_frequency", "median_frequency", "RMS"},
    "time": {"MAV", "WL", "VAR", "relative_MAV"},
    "amplitude": {"MAV", "relative_MAV"},
}


class KNNTrack(ExpandedFeatureTrack):
    def __init__(self, domain):
        super().__init__()
        self.columns = None if DOMAINS[domain] is None else [
            i for i, name in enumerate(expanded_feature_names())
            if name.split("_", 1)[1] in DOMAINS[domain]]

    def extract_features(self, rec, cfg=None):
        if self.columns is None:
            return EMGNinaproTrack.extract_features(self, rec, cfg)
        X, y, group = super().extract_features(rec, cfg)
        return X[:, self.columns], y, group

    def baseline(self, cfg=None):
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))


def main():
    track = KNNTrack("full_feature")
    records = track.load(ROOT / "NinaPro_Mat", exercise="E1")
    track.assert_no_holdout(records)
    if len(records) != 22 or {r.group for r in records} != {f"S{i}" for i in range(1, 23)}:
        raise ValueError("Expected one development recording for each subject S1-S22")
    output = Path(__file__).resolve().parent / "plots"
    output.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        track = KNNTrack(domain)
        start = perf_counter()
        results = evaluate_modes_progress(track, records, cfg=CFG)
        elapsed = perf_counter() - start
        for mode, result in results.items():
            track.report(result, title=f"{domain} - {mode} (both modes: {elapsed:.2f} s)")
        save_plots(results, domain, output)


if __name__ == "__main__":
    main()
