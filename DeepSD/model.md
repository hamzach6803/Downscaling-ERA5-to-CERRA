# DeepSD

## Idea

`DeepSD` is a convolutional neural network for deterministic super-resolution. It receives an ERA5 field interpolated on the CERRA grid and learns to produce the CERRA target.

The network uses convolution layers and an upsampling block. The output is trained with mean squared error.

## Code Flow

`model.py`:

- Defines the `DeepSD` CNN.
- Uses convolution layers, ReLU activations, and bilinear upsampling.

`train.py`:

1. Loads ERA5 and CERRA.
2. Normalizes data using the train split.
3. Creates train and test loaders.
4. Trains the CNN with MSE loss.
5. Saves the best checkpoint by test loss.

`predict.py`:

1. Loads the trained checkpoint.
2. Selects a date, test index, or random test map.
3. Runs the CNN.
4. Denormalizes prediction, computes metrics, saves NetCDF and PNG plot.

## Outputs

```text
checkpoints/deepsd_best_<variable>.pth
checkpoints/deepsd_loss_<variable>.png
predictions/deepsd_pred_<variable>.nc
predictions/deepsd_<variable>_000.png
```

## Main Settings

Use `--epochs`, `--batch-size`, and `--device cuda` for training. For small GPUs, reduce `--batch-size`.
