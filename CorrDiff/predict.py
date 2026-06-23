import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from corrdiff_core import predict_corrdiff


if __name__ == "__main__":
    predict_corrdiff(
        Path(__file__).resolve().parent,
        "CorrDiff",
        "corrdiff",
        use_mnt=False,
        backprojection=False,
        bilateral=True,
    )
