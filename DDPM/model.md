# DDPM

## Idea

`DDPM` is a conditional denoising diffusion model with a cosine noise schedule. The model learns to predict Gaussian noise added to the CERRA field, conditioned on the ERA5 field.

During prediction, the model starts from random noise and iteratively removes predicted noise.

## Code Flow

`model.py`:

- Defines the cosine schedule.
- Defines `add_cosine_noise`.
- Defines `CosineDDPM`, a small conditional U-Net.

`train.py`:

1. Loads and normalizes ERA5/CERRA.
2. Upsamples ERA5 to the CERRA shape.
3. Adds cosine-scheduled noise to CERRA.
4. Trains the U-Net to predict the noise.
5. Saves checkpoint and loss curve.

`predict.py`:

1. Loads checkpoint.
2. Selects date, test index, or random test map.
3. Starts from Gaussian noise.
4. Iteratively denoises using the trained model.
5. Denormalizes and saves metrics, NetCDF, and plot.

## Outputs

```text
checkpoints/ddpm_cosine_best_<variable>.pth
checkpoints/ddpm_loss_<variable>.png
predictions/ddpm_cosine_pred_<variable>.nc
predictions/ddpm_cosine_<variable>_000.png
```

## Main Settings

Use `--steps` for the number of diffusion steps. More steps are slower but usually smoother.
