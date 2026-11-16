import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import ExpandedFeatureTrack
from emg_ninapro import EMGNinaproTrack
from features import expanded_feature_names
from classifier_evaluation import evaluate_modes_progress
from random_forest.rf_plot_results import save_plots
from datetime import datetime


ROOT = Path(__file__).resolve().parents[2]
CFG = {"seed": 0, "select": "none", "imbalance": "balanced",
       "cv_max_splits": 5, "loso_max_groups": 12}
DOMAINS = {
    "baseline": None,
    "baseline_remove_abs": None,
    "full_feature": {"MAV", "WL", "VAR", "relative_MAV", "mean_frequency", "median_frequency", "RMS"},
    "time": {"MAV", "WL", "VAR", "relative_MAV"},
    "amplitude": {"MAV", "relative_MAV"},
}


class RFTrack(ExpandedFeatureTrack):
    def __init__(self, domain):
        super().__init__()

        self.domain = domain  # Store the domain for later use 13/11 TJ

        self.columns = None if DOMAINS[domain] is None else [
            i for i, name in enumerate(expanded_feature_names())
            if name.split("_", 1)[1] in DOMAINS[domain]]

    # def extract_features(self, rec, cfg=None): # Original iteration of extract_features method, commented out for reference
    #     if self.columns is None:
    #         return EMGNinaproTrack.extract_features(self, rec, cfg)
    #     X, y, group = super().extract_features(rec, cfg)
    #     return X[:, self.columns], y, group

    def extract_features(self, rec, cfg=None):
        if self.domain in ("baseline", "baseline_remove_abs"):
            X, y, group = EMGNinaproTrack.extract_features(self, rec, cfg)

            if self.domain == "baseline_remove_abs":
                # Baseline order per channel:
                # MAV, RMS, WL, VAR, rFFT centroid.
                assert X.shape[1] == 50

                columns = [
                    channel * 5 + offset
                    for channel in range(10)
                    for offset in (2, 3, 4)
                ]
                X = X[:, columns]
                assert X.shape[1] == 30

            return X, y, group

        X, y, group = super().extract_features(rec, cfg)
        return X[:, self.columns], y, group


def main():
    track = RFTrack("full_feature")
    records = track.load(ROOT / "NinaPro_Mat", exercise="E1")
    track.assert_no_holdout(records)
    if len(records) != 22 or {r.group for r in records} != {f"S{i}" for i in range(1, 23)}:
        raise ValueError("Expected one development recording for each subject S1-S22")
    # output = Path(__file__).resolve().parent / "plots" # Original iteration of extract_features method, commented out for reference
    # output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Generate a unique run ID based on the current date and time
    output = (
        Path(__file__).resolve().parent
        / "plots"
        / f"remove_mav_rms_{run_id}"
    )
    output.mkdir(parents=True, exist_ok=False)
    print(f"Saving plots to: {output}", flush=True)
    # for domain in DOMAINS: # Original iteration of extract_features method, commented out for reference
    for domain in ("baseline", "baseline_remove_abs"):
        track = RFTrack(domain)
        start = perf_counter()
        results = evaluate_modes_progress(track, records, cfg=CFG)
        elapsed = perf_counter() - start
        for mode, result in results.items():
            track.report(result, title=f"{domain} - {mode} (both modes: {elapsed:.2f} s)")
        save_plots(results, domain, output)


if __name__ == "__main__":
    main()
