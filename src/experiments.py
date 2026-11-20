from emg_ninapro import EMGNinaproTrack
from features import extract_window_features


class ExpandedFeatureTrack(EMGNinaproTrack):
    MODEL_CFG_KEYS = frozenset()

    @classmethod
    def supported_cfg_keys(cls):
        return super().supported_cfg_keys() | cls.MODEL_CFG_KEYS

    def feature_cfg_keys(self):
        return super().feature_cfg_keys() - self.MODEL_CFG_KEYS

    def extract_features(self, rec, cfg=None):
        return extract_window_features(rec, cfg=cfg)
