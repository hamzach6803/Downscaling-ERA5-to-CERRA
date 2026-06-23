import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from corrdiff_core import train_corrdiff


if __name__ == "__main__":
    train_corrdiff(Path(__file__).resolve().parent, "CorrDiff", "corrdiff", use_mnt=False)
