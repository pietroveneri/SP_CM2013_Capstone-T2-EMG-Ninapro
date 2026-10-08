# Data

This directory is intended for local working data used by the project.

The project uses the Ninapro DB1 dataset, primarily Exercise E1.

Dataset files should not be committed to Git. The repository contains the code required to load and, where applicable, download the data reproducibly.

The current EMG track expects Ninapro DB1 `.mat` files and uses:

- 27 intact subjects
- Exercise E1
- 12 finger movements
- 10 surface-EMG channels
- 100 Hz sampling rate

See `src/emg_ninapro.py` for the dataset loader and track implementation.