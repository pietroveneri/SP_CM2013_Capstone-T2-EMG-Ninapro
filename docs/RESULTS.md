# Results log — `EMG NinaPro`, `T2`

## Hold-out correction — 2026-10-21

**All results recorded before this correction that used all 27 subjects are invalid under the current course hold-out protocol.** The baseline included S23–S27 in training and cross-validation before their hold-out designation was published. These results and artifacts are retained only for historical transparency; they must not be used as valid development scores or as the registered expected hold-out score.

Development now uses **S1–S22 only (22 subjects)**; **S23–S27 are reserved for final new-subject evaluation**. The loader excludes these subjects before reading their files, including in existing local downloads; evaluation and final training reject hold-out groups. Rerun the baseline and affected experiments on the permitted subjects before reporting replacement results. No replacement baseline was run for this correction.

The historical statistics below describe the old 27-subject run. Their spread is **across subjects**, not the **across-fold** mean ± SD required in `EXPECTED_SCORE.md`. Register new-subject macro-F1, subject split unit, fold count and the frozen-model commit before final hold-out access. The earlier exposure to S23–S27 is disclosed here and is not undone by filtering them now.

**Track:** `<emg_ninapro >` · **Subjects:** 27 · **Classes:** g01-g12; rest excluded · **Windows:** 75,197 per evaluation mode 

**Evaluation modes:** `<within-subject + new-subject>` · **Split unit:** `<repetition (within-subject); subject (new-subject)>` ·  **CV:** 5-fold GroupKFold on repetitions within each subject; 5 fold GroupKFold on subjects for new-subject
**Primary metric:** `<macro-F1>` · **Secondary metrics:** Cohen's κ, balanced accuracy, accuracy **Spread:** mean, sample SD and min-max across 27 subjects.
 
**Detailed baseline results:** [BASELINE.md](Baseline/BASELINE.md)

```python
cfg = {
    "seed": 0, 
    "select": "none",
    "imbalance": "balanced",
    "cv_max_splits": 5,
    "loso_max_groups": 12
}
```

## Iteration log

Paste the metric straight from the harness — `rep["summary"]` prints the required shape.

| # | Date | What changed & why (one line) | Primary metric **with spread** | Better than previous? | If not — why it was kept | Commit |
|---|---|---|---|---|---|---|
| 1 | 2026-10-08 | supplied baseline, unchanged | within_subject mean macro_f1 0.798 (sd 0.049, range 0.697-0.879 across 27 subjects, spread is per subject) - new_subject mean macro_f1 0.204 (mean macro_f1 0.205 (sd 0.073, range 0.082-0.380 across 27 subjects), spread is per subject) | — | (baseline) | `379f1e6` |
| 2 |  |  |  | yes / no |  |  |
| 3 |  |  |  | yes / no |  |  |
| 4 |  |  |  | yes / no |  |  |

### Baseline provenance

- Results registered in commit: `379f1e6`.
- Verification run from commit `0b85ef5`.
- Original execution environment: [requirements-lock.txt](../requirements-lock.txt)
- Detailed metric panel and confusion matrices: [BASELINE.md](Baseline/BASELINE.md).
- Per-window predictions and complete per-subject results: [BASELINE 10/10/2026](Baseline/baseline_20261010_verified).

*"Better" means better under the same honest harness — same split unit, same evaluation mode,
same seed. A change that lowers the metric can still be the right call (simpler, faster, more
robust across subjects, removes a leak). Say so in the column instead of quietly reverting it:
"kept — κ fell 0.02 but the worst-subject κ rose from 0.18 to 0.31" is a stronger result than a
silent higher mean.*

| approach | what you do | what it costs |
|---|---|---|
| **an explicit development set** | hold out a few groups *up front* as a DEV set; compare every design option on DEV only; run the chosen pipeline **once** on the untouched evaluation folds and report that | fewer groups serving each purpose — painful on these cohorts — and a noisier DEV estimate |
| **a comparison budget** | decide in advance how many configurations you will compare (a handful, not a sweep), write them into this file **before** you run them, and report the count | you may miss a better option, and you have to resist re-opening the budget once you have seen the numbers — which is the whole discipline |

## Decision log — the choices behind the numbers

Baseline's detailed decision log: [BASELINE.md](Baseline/BASELINE.md)

| Pipeline module | Option chosen | Alternative(s) considered | Why this one (one sentence) | Iteration | Revised later? |
|---|---|---|---|---|---|
| 1. Data loading | DB1 E1; all 27 subjects; restimulus/rerepetition; rest excluded | - | Use the prescribed dataset scope and the supplied correct labels | - |
| 2. Preprocessing |  |  |  |  |  |
| 3. Feature extraction |  |  |  |  |  |
| 4. Feature selection |  |  |  |  |  |
| 5. Classification, incl. `imbalance` |  |  |  |  |  |
| 6. Inference |  |  |  |  |  |
| 7. Reporting |  |  |  |  |  |

## Revisions — the ones that went **backwards** (Criterion 9's actual evidence)

Adding iterations forward is a to-do list. What this section wants is the place a number
downstream sent you back **up** the pipeline. One row is enough; two is a good project.
The track instructions have a symptom → stage table to
diagnose from.

| # | The downstream result that triggered it | Which earlier decision it indicted | What you changed | What happened to the metric |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |

## Who did what

One line per person per iteration — the honest-disclosure requirement of §16.7, and the evidence
`INDIVIDUAL_ASSESSMENT.md` asks for. The seven modules do **not** have to map one-to-one onto
people: one person may own several modules, two people may share one, and ownership may rotate
between iterations. Record what actually happened.

| Iteration | Who | Modules / tasks owned | Reviewed by |
|---|---|---|---|
| 1 | Pietro  | Ran the supplied baseline and recorded its results |  |



## Final numbers 

| | Value | Under what split |
|---|---|---|
| Primary metric, development (mean + spread) |  |  |
| Secondary metrics (macro-F1 / balanced accuracy / per-class recall) |  |  |
| Held-out set — **scored once, never tuned against** |  |  |
| Supplied baseline, same harness |  |  |
| Yardstick from the dataset card (human ceiling / benchmark / chance) |  |  |
| **Configurations compared** before settling (a number) |  |  |
| How option comparison was kept separate from final reporting |  |  |

