"""Run the fold-safe median/Savitzky-Golay z-score recipe independently."""
import run_preprocessing_experiments as runner

runner.OPTIONS = {3: "median_savgol_zscore"}
runner.main()
