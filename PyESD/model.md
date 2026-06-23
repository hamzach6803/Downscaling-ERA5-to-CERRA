# PyESD

## Idea

`PyESD` is a simple pixel-wise deterministic downscaling model. ERA5 is first interpolated to the CERRA grid. Then, for each high-resolution pixel, the model learns a linear relation:

```text
CERRA_norm = slope * ERA5_interp_norm + intercept
```

This is fast and interpretable, but it cannot generate fine stochastic structures.

## Code Flow

`train.py`:

1. Loads ERA5 and CERRA with `load_and_preprocess`.
2. Normalizes both datasets using only the 80 percent train split.
3. Computes pixel-wise slope and intercept on the train period.
4. Saves `pyesd_linear_<variable>.npz`.

`predict.py`:

1. Loads the checkpoint.
2. Selects a date, test index, or random test map.
3. Applies the pixel-wise linear equation.
4. Denormalizes, evaluates metrics, saves NetCDF and PNG plot.

## Outputs

```text
checkpoints/pyesd_linear_<variable>.npz
predictions/pyesd_pred_<variable>.nc
predictions/pyesd_<variable>_000.png
```

## Main Settings

Change the weather variable and data paths in `../utils.py`. See `config.yaml` for the expected command and output names.
