# MLP figure provenance

The PNGs in `mlp_20`, `mlp_40` and `mlp_60` here were generated on
2026-11-16 from copies of the four CSVs in the corresponding `src/MLP/mlp_*`
directory using the corrected `src/MLP/plot_envelope_cnn.py`. Confusion and
ROC use both prediction CSVs; macro-F1 uses both subject-score CSVs. They are
out-of-fold development results, not a hold-out evaluation.

The historical `cnn_confusion.png`, `cnn_macro_f1.png` and `cnn_roc_auc.png`
in each of the three source directories were compared with these regenerated
images: all nine differ pixel-for-pixel. Rendering and library differences can
cause differences; this comparison does not establish that an original image
is wrong or identify its historical input. Their provenance remains unverified.
Use the regenerated images here when citing the corresponding CSVs.

Generation used Python 3.13.2, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2
and Matplotlib 3.10.9. SHA-256 hashes of the exact input files follow:

| Set | CSV | SHA-256 |
|---|---|---|
| 20 | within_subject_predictions.csv | `05df0917c6d170a56aacfa3a5db259af5fb40cc713ddf88d2209c4146d116dd7` |
| 20 | new_subject_predictions.csv | `29d34f5b33abf43d3157296f81965f1c15501481120f79843a2d1b21287a1850` |
| 20 | within_subject_subject_scores.csv | `73a4b64e672265a0f684a3fd084104c9a19a186ba554049301931b27ee38bc7f` |
| 20 | new_subject_subject_scores.csv | `bedfe7551011aa783d17800e9b8edf6e362f78865f6006b2894b9216d06447ce` |
| 40 | within_subject_predictions.csv | `29cbb401b267c586a46657b9ccea2e7c65648ae1858907473cd5acb5602cefb8` |
| 40 | new_subject_predictions.csv | `1b64a24b595b00d9faa2362a51b3a6ac33f6830b1797d3b0a0a66e83599658b5` |
| 40 | within_subject_subject_scores.csv | `cf262a015a594e4c80ae2cea86ff4a9c7b0df127f2e5454741b10e10bcfce389` |
| 40 | new_subject_subject_scores.csv | `9e65ded4194370b6ca9a8817fd128898f48520d02de08dc3b2d99436cc7e224a` |
| 60 | within_subject_predictions.csv | `0ba368869f3ad014ff2d23319be2200230ae31325840a7e38f654f1270ed06ee` |
| 60 | new_subject_predictions.csv | `9257484f5b768bcfa55f2916e228a22b1770ce6f1c93c44bb7be42528c0fbea1` |
| 60 | within_subject_subject_scores.csv | `5953b645705c9c43e86b378490b2d799c287c75a1c3ddd5500515f03d5d7f6d5` |
| 60 | new_subject_subject_scores.csv | `dc76ef1b7aa49ce227bc98763fab3ddda1461efcf0ea4cae29e7989236883d3a` |
