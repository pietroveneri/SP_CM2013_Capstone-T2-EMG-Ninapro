# CM2013 Capstone — EMG Gesture Recognition with Ninapro DB1

Team project for CM2013 Biomedical Signal Processing & Data Analytics.

The project investigates hand-gesture classification using the Ninapro DB1 surface-EMG dataset, with particular attention to the difference between within-subject and new-subject generalization.

## Current baseline

The supplied pipeline has been evaluated on all 27 subjects using `cfg={"seed": 0}`.

| Evaluation mode | Macro-F1 |
| --- | --- |
| Within-subject | 0.798 ± 0.049 |
| New-subject | 0.204 ± 0.074 |

Full experiment history is recorded in `docs/RESULTS.md`.

## Repository structure

```text
src/
├── adapter.py          # supplied evaluation/pipeline framework
├── emg_ninapro.py      # EMG track integration
├── report.py           # supplied reporting utilities
├── run_baseline.py     # supplied baseline runner
├── preprocessing.py    # team preprocessing
├── features.py         # team feature engineering
├── models.py           # team ML models
└── experiments.py      # experiment configurations/runners

docs/                   # track documentation and RESULTS.md
Notes/                  # internal notes
Reference/              # literature/background material
data/                   # local working data
```

Each experiment or isolated change is developed on a short-lived branch:
```
git switch main
git pull
git switch -c experiment/<name>
```
After completing the experiment:
```
git add .
git commit -m "Add per-channel normalization"
git push -u origin experiment/normalization
```
Then open a Pull Request into main.

## Experiment
For every experimental iteration:
1. Start from the current accepted pipeline.
2. Change one clearly defined component when possible.
3. Keep the prescribed group-aware validation.
4. Evaluate both within-subject and new-subject performance.
5. Report the metric together with its spread.
6. Update docs/RESULTS.md.
7. Record negative or rejected experiments as well as successful ones.

Poi: 

```bash
git add README.md data/README.md .gitignore
git commit -m "Set up repository structure and workflow"
git push
```