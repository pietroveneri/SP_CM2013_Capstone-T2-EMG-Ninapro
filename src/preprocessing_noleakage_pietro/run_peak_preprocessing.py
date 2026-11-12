"""Run the fold-safe peak recipe independently."""
import run_preprocessing_experiments as runner

runner.OPTIONS = {5: "peak"}
runner.main()
