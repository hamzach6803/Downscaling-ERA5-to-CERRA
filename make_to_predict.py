import argparse
from pathlib import Path

from utils import (
    DEFAULT_CERRA,
    DEFAULT_ERA5,
    DEFAULT_VARIABLE,
    TRAIN_RATIO,
    default_to_predict_path,
    make_to_predict_file,
)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a small NetCDF file containing only the prediction date."
    )
    parser.add_argument("--era5", type=Path, default=DEFAULT_ERA5)
    parser.add_argument("--cerra", type=Path, default=DEFAULT_CERRA)
    parser.add_argument("--variable", default=DEFAULT_VARIABLE)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--date", default=None, help="Date a extraire, ex: 2025-01-03 ou 2025-01-03T12:00")
    parser.add_argument("--time-index", type=int, default=None, help="Index dans le split test si pas de date")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or default_to_predict_path(args.variable)
    make_to_predict_file(
        era5_path=args.era5,
        cerra_path=args.cerra,
        variable=args.variable,
        train_ratio=args.train_ratio,
        date=args.date,
        time_index=args.time_index,
        output_path=output,
    )


if __name__ == "__main__":
    main()
