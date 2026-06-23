# CorrDiff

## Idea

`CorrDiff` is a corrective diffusion model. It first builds a deterministic forecast with pixel-wise linear regression, then learns a score-based diffusion model for the residual:

```text
baseline = slope * ERA5 + intercept
residual = CERRA - baseline
prediction = baseline + sampled_residual
```

The diffusion part uses a VE-SDE style score model trained on noisy residuals.

## Code Flow

`train.py`:

1. Loads and normalizes ERA5/CERRA.
2. Fits a pixel-wise linear regression baseline.
3. Computes train baselines.
4. Computes residuals.
5. Trains a score U-Net on residual patches.
6. Saves the score model and linear regression coefficients.

`predict.py`:

1. Loads the score model and linear regression coefficients.
2. Selects a date, test index, or random test map.
3. Computes linear regression baseline.
4. Samples a residual with Langevin-style score sampling.
5. Adds residual to baseline and saves outputs.

## Outputs

```text
checkpoints/corrdiff_score_<variable>.pth
checkpoints/corrdiff_loss_<variable>.png
predictions/corrdiff_pred_<variable>.nc
predictions/corrdiff_<variable>_000.png
```

## Main Settings

Important arguments: `--sigma-min`, `--sigma-max`, `--sample-steps`, `--patch-size`, and `--channels`.
