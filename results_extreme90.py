from pathlib import Path

import results_90times


ROOT = Path(__file__).resolve().parent
results_90times.OUT_DIR = ROOT / "outputs" / "comparisons" / "selected_test" / "results_extreme90"
results_90times.PRED_DIR = results_90times.OUT_DIR / "predictions"
results_90times.PALETTE_MODE = "uniform"  # "uniform" ou "adaptive"
results_90times.RESULTS_LABEL = "90 cas extremes"


if __name__ == "__main__":
    results_90times.main()
