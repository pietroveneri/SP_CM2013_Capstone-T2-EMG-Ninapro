# Results log — `EMG NinaPro`, `T2`

**Track:** `<emg_ninapro >` · **Subjects:** 27 · **Classes:** g01-g12; rest excluded · **Windows:** 75,197 per evaluation mode 

**Evaluation modes:** `<new-subject | within-subject + new-subject>` · **Split unit:** `<repetition (within-subject); subject (new-subject)>` ·  **CV:** 5-fold GroupKFold on repetitions within each subject; 5 fold GroupKFold on subjects for new-subject
**Primary metric:** `<Cohen's κ | macro-F1>` · **Secondary metrics:** Cohen's κ, balanced accuracy, accuracy **Spdread:** mean, sample SD and min-max across 27 subjects.
 
**Detailed baseline results:** [BASELINE.md](BASELINE.md)

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
| 1 | 2026-10-08 | supplied baseline, unchanged | within_subject mean macro_f1 0.798 (sd 0.049, range 0.697-0.879, spread is per subject) across 27 subjects - new_subject mean macro_f1 0.204 (sd 0.074, range 0.080-0.381, spread is per subject) across 27 subjects | — | (baseline) | `4c2fb7a` |
| 2 |  |  |  | yes / no |  |  |
| 3 |  |  |  | yes / no |  |  |
| 4 |  |  |  | yes / no |  |  |

### Baseline provenance

- Results registered in commit: `4c2fb7a`.
- Exact commit executed for the original run: `4c2fb7a`.
- Original execution environment: [requirements-lock.txt](requirements-lock.txt).
- Detailed metric panel and confusion matrices: [BASELINE.md](BASELINE.md).
- Per-window predictions and complete per-subject results: not yet archived.

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

Rows above say *what happened*; this says *what you chose and why*, which is what §16.4 asks you
to make traceable and what the report's defence is built from. There is no single correct
pipeline here — the scaffold deliberately ships options, not answers. One line per decision;
add rows as the pipeline grows, and note the alternative you rejected.

| Pipeline module | Option chosen | Alternative(s) considered | Why this one (one sentence) | Iteration | Revised later? |
|---|---|---|---|---|---|
| 1. Data loading | DB1 E1; all 27 subjects; restimulus/repetition; rest excluded | - | Use the prescribed dataset scope and the supplied correct labels | - |
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

