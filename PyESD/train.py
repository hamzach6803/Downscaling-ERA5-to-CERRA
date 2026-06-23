import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils import (
    add_common_args,
    add_training_memory_args,
    compute_train_mean_std,
    cleanup_memory,
    fit_pixelwise_linear_regression_streaming,
    load_aligned_training_data,
    save_hyperparams,
    split_index,
    variable_tag,
)


MODEL_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Train PyESD pixel-wise linear downscaling."))
    add_training_memory_args(parser, max_samples=True)
    args = parser.parse_args()

    era5_interp, cerra = load_aligned_training_data(args.era5, args.cerra, args.variable)
    n_train = split_index(len(era5_interp), args.train_ratio)
    if args.max_train_samples is not None:
        n_train = min(n_train, int(args.max_train_samples))

    print(f"Training samples: {n_train}")
    print(f"Chunk size: {args.chunk_size}")
    print("Computing normalization stats by chunks...")
    x_mu, x_std = compute_train_mean_std(era5_interp, n_train, args.chunk_size)
    y_mu, y_std = compute_train_mean_std(cerra, n_train, args.chunk_size)
    print("Fitting pixel-wise linear regression by chunks...")
    slope, intercept = fit_pixelwise_linear_regression_streaming(
        era5_interp,
        cerra,
        n_train,
        x_mu,
        x_std,
        y_mu,
        y_std,
        args.chunk_size,
    )
    del era5_interp, cerra
    cleanup_memory()

    tag = variable_tag(args.variable)
    path = CHECKPOINT_DIR / f"pyesd_linear_{tag}.npz"
    np.savez_compressed(
        path,
        slope=slope,
        intercept=intercept,
        x_mu=x_mu,
        x_std=x_std,
        y_mu=y_mu,
        y_std=y_std,
        train_ratio=args.train_ratio,
        variable=args.variable,
    )
    print(f"PyESD saved: {path}")
    save_hyperparams(
        MODEL_DIR,
        "PyESD",
        {
            "model_type": "pixel-wise linear regression",
            "variable": args.variable,
            "era5": args.era5,
            "cerra": args.cerra,
            "train_ratio": args.train_ratio,
            "train_samples": n_train,
            "max_train_samples": args.max_train_samples,
            "chunk_size": args.chunk_size,
            "checkpoint": path,
            "x_mu": x_mu,
            "x_std": x_std,
            "y_mu": y_mu,
            "y_std": y_std,
        },
    )
    del slope, intercept
    cleanup_memory()


if __name__ == "__main__":
    main()
