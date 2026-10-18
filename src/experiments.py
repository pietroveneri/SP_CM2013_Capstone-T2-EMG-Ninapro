from emg_ninapro import EMGNinaproTrack
from features import extract_window_features


class ExpandedFeatureTrack(EMGNinaproTrack):
    def extract_features(self, rec, cfg=None):
        return extract_window_features(rec, cfg=cfg)
