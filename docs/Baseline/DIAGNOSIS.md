# Preliminary baseline diagnosis — Ninapro DB1, E1

**Scope.** The saved out-of-fold baseline covers 27 subjects, 12 gestures and 75,197 windows. Mean subject macro-F1 is **0.798 ± 0.046 within-subject** (five repetition-grouped folds per subject) versus **0.205 ± 0.073 for new subjects** (five subject-grouped folds, not LOSO); SD is across subjects. This large transfer gap is the main finding. 

**Difficult subjects.** S2, S27 and S7 have the lowest new-subject macro-F1, although their within-subject results remain much stronger (table). S7's fall from 0.864 to 0.103 particularly suggests a subject-specific representation that transfers poorly. S25 provides a relatively successful comparison. Within-subject, the hardest cases are S14 (0.706; g09 → g12 in 19.8% of true-g09 windows) and S18 (0.708; g11 → g10 in 20.8%). Thus, difficulties transferring between subjects coexist with repetition-level errors.

| Subject | Macro-F1 within / new | g09 MAV median [Q1, Q3] | g09 RMS median [Q1, Q3] |
|---|---:|---:|---:|
| S2 | 0.775 / 0.082 | 0.421 [0.290, 0.569] | 0.424 [0.296, 0.574] |
| S27 | 0.768 / 0.085 | 0.077 [0.042, 0.117] | 0.080 [0.043, 0.118] |
| S7 | 0.864 / 0.103 | 0.169 [0.137, 0.209] | 0.171 [0.138, 0.212] |
| S25 | 0.855 / 0.380 | 0.401 [0.244, 0.656] | 0.403 [0.247, 0.664] |

MAV/RMS summaries are window distributions of the **mean across ten channels**, in native stored units; brackets contain the interquartile range. Windows are correlated observations.

**Main confusions.** Pooled within-subject errors concentrate on g09 → g11 (11.4%) and g11 → g09 (10.4%). For new subjects, the largest errors are g10 → g12 (18.4%), g09 → g12 (17.0%), g12 → g10 (16.2%) and g06 → g04 (16.0%). New-subject recall is especially low for g09 (7.3%) and g07 (7.9%). Subject-level errors are stronger: S2 sends 69.4% of g09 to g12, while S27 sends 83.6% of g10 to g12. Percentages use true-class row denominators, not all errors.

**Same-gesture amplitude and channel patterns.** Across the 12 gestures, the largest/smallest subject-median amplitude ratios range from **6.7–22.1× for MAV** and **6.5–20.9× for RMS**. However, S2 and S25 have similar g09 amplitudes despite very different transfer scores. Their spatial patterns differ: after dividing each window's channel MAVs by their sum, the largest mean share for g09 is channel 1 in S2 (30%), channel 7 in S7 (34%), channel 1 in S27 (49%) and channel 8 in S25 (33%). These observations support amplitude and channel-pattern differences; they do not identify their physical cause.

