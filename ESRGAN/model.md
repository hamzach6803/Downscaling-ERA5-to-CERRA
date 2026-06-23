# ESRGAN

## Idea

`ESRGAN` is a GAN-based downscaling model. It uses a generator with RRDB-style residual dense blocks and a discriminator trained to distinguish real CERRA patches from generated patches.

The generator loss combines:

```text
L_G = L1(fake, real) + adv_weight * adversarial_loss
```

Training uses random spatial patches to avoid CUDA out-of-memory errors.

## Code Flow

`model.py`:

- Defines dense residual blocks.
- Defines the ESRGAN generator.
- Defines the discriminator.

`train.py`:

1. Loads and normalizes ERA5/CERRA.
2. Builds random patch loaders.
3. Alternates discriminator and generator updates.
4. Uses AMP automatically on CUDA unless `--no-amp` is passed.
5. Saves the best generator checkpoint.

`predict.py`:

1. Loads the generator.
2. Selects a date, test index, or random test map.
3. Predicts the full CERRA grid.
4. Saves metrics, NetCDF, and plot.

## Outputs

```text
checkpoints/esrgan_generator_<variable>.pth
checkpoints/esrgan_loss_<variable>.png
predictions/esrgan_pred_<variable>.nc
predictions/esrgan_<variable>_000.png
```

## Main Settings

For 4 GB GPUs, use:

```bash
python ESRGAN/train.py --device cuda --batch-size 1 --patch-size 32 --channels 16 --rrdb-blocks 1
```
