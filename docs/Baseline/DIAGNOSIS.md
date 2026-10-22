# Baseline diagnosis — Ninapro DB1, E1 — 2026-10-22

**Scope.** S1–S22 only: 22 subjects, 12 gestures and 60,182 OOF windows per mode. Subject macro-F1 is **0.794 ± 0.048 within-subject** versus **0.177 ± 0.055 new-subject** (sample SD across subjects). New-subject uses five subject folds, not LOSO; its across-fold macro-F1 is **0.190 ± 0.016**.

**Correction.** The previous diagnosis included S23–S27 and had no separate hold-out test. Its amplitude and channel-pattern comparisons are withdrawn. No hold-out files were read or scored for this replacement. Earlier exposure remains documented in RESULTS.md.

**within_subject.** Lowest subject macro-F1: S14 (0.706), S18 (0.708), S16 (0.739). Largest pooled off-diagonal rates: g09 → g11 (10.0%), g11 → g09 (8.4%), g09 → g10 (6.4%), g11 → g10 (6.3%).


**new_subject.** Lowest subject macro-F1: S2 (0.064), S7 (0.092), S9 (0.122). Largest pooled off-diagonal rates: g06 → g04 (20.5%), g09 → g10 (16.2%), g11 → g02 (15.4%), g10 → g11 (15.0%).

Rates use true-class row denominators. The within/new-subject gap motivates testing subject-robust features on development folds; these errors alone do not identify a physical cause. No improvement was applied. See [BASELINE.md](BASELINE.md) and the [run archive](baseline_20261022_holdout_separated/README.md).
