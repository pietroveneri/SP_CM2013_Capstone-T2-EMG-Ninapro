from emg_ninapro import EMGNinaproTrack
from features import extract_window_features


class ExpandedFeatureTrack(EMGNinaproTrack):
    def extract_features(self, rec, cfg=None):
        X = extract_window_features(
            rec.epochs["emg"],
            fs=rec.fs
        )

        return X, rec.labels, rec.group