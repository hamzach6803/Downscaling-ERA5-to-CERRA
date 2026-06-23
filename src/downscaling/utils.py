import argparse
import gc
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
TO_PREDICT_DIR = ROOT_DIR / "to_predict"

# Change ces variables ici au lieu de les passer en argument.
COLOR = "coolwarm"  # fallback colormap
METEO_VARIABLE = "t2m"
ERA5_DATA_PATH = DATA_DIR / "era5_2021_2025.nc"
CERRA_DATA_PATH = DATA_DIR / "cerra_2021_2025_t2m_wgs84.nc"
MNT_PATH = DATA_DIR / "mnt.nc"
MNT_VARIABLE_CANDIDATES = [
    "elevation",
    "orog",
    "z",
    "mnt",
    "dem",
    "altitude",
    "__xarray_dataarray_variable__",
]
ALLOW_MNT_COORD_FALLBACK = True

DEFAULT_ERA5 = ERA5_DATA_PATH
DEFAULT_CERRA = CERRA_DATA_PATH
DEFAULT_VARIABLE = METEO_VARIABLE
TRAIN_RATIO = 0.8
BATCH_SIZE = 1
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cleanup_memory(device=None):
    gc.collect()
    if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
        torch.cuda.empty_cache()

def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--era5", type=Path, default=DEFAULT_ERA5)
    parser.add_argument("--cerra", type=Path, default=DEFAULT_CERRA)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=str(DEVICE), choices=["cpu", "cuda"])
    parser.set_defaults(variable=METEO_VARIABLE)
    return parser

def add_training_memory_args(parser: argparse.ArgumentParser, max_samples=True) -> argparse.ArgumentParser:
    if max_samples:
        parser.add_argument(
            "--max-train-samples",
            type=int,
            default=None,
            help="Limit training timesteps to reduce RAM usage.",
        )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Timesteps processed at once for low-RAM training.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 on low-RAM machines.",
    )
    return parser

def variable_tag(variable=METEO_VARIABLE):
    return str(variable).replace("/", "_").replace("\\", "_").replace(" ", "_")

def variable_kind(variable=METEO_VARIABLE):
    var = str(variable).lower()
    if var in {"t2m", "temperature", "temp"}:
        return "temperature"
    if var in {"ws10", "wind", "wind_speed", "w10", "si10"}:
        return "wind"
    if var in {"tp", "precip", "precipitation", "pluie", "rain"}:
        return "precipitation"
    return "default"

def plot_cmap(variable=METEO_VARIABLE):
    kind = variable_kind(variable)
    if kind == "temperature":
        return "coolwarm"
    if kind == "wind":
        return "viridis"
    if kind == "precipitation":
        return "Blues"
    return COLOR

def plot_unit(variable=METEO_VARIABLE):
    kind = variable_kind(variable)
    if kind == "temperature":
        return "degC"
    if kind == "wind":
        return "m/s"
    if kind == "precipitation":
        return "mm"
    return str(variable)

def values_for_plot(data, variable=METEO_VARIABLE):
    arr = np.asarray(data, dtype=np.float32)
    if variable_kind(variable) == "temperature":
        finite = arr[np.isfinite(arr)]
        if finite.size and float(np.nanmedian(finite)) > 100.0:
            return arr - 273.15
    return arr

def plot_limits(data, variable=METEO_VARIABLE, lower=2, upper=98):
    arr = np.asarray(data, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    kind = variable_kind(variable)
    if kind in {"wind", "precipitation"}:
        vmax = float(np.nanpercentile(arr, upper))
        return 0.0, vmax if vmax > 0 else float(np.nanmax(arr))
    vmin = float(np.nanpercentile(arr, lower))
    vmax = float(np.nanpercentile(arr, upper))
    if vmin == vmax:
        pad = max(abs(vmin) * 0.05, 1.0)
        return vmin - pad, vmax + pad
    return vmin, vmax

def save_hyperparams(model_dir, model_name, hyperparams, filename="hyperparsm.md"):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / filename

    rows = []
    for key, value in hyperparams.items():
        if isinstance(value, Path):
            value = str(value)
        rows.append(f"| `{key}` | `{value}` |")

    content = "\n".join(
        [
            f"# Hyperparameters - {model_name}",
            "",
            f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
            "",
            "| Parameter | Value |",
            "| --- | --- |",
            *rows,
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")

    print(f"\n{model_name} hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    print(f"Hyperparameters saved: {path}")
    return path

def add_predict_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_common_args(parser)
    parser.add_argument("--date", default=None, help="Date a tracer, ex: 2025-01-03 ou 2025-01-03T12:00")
    parser.add_argument("--time-index", type=int, default=None, help="Index dans le split test si pas de date")
    parser.add_argument(
        "--to-predict",
        type=Path,
        default=None,
        help="Petit fichier NetCDF deja prepare avec make_to_predict.py.",
    )
    return parser

def load_and_preprocess(era5_path=DEFAULT_ERA5, cerra_path=DEFAULT_CERRA, variable=DEFAULT_VARIABLE):
    era5_ds = xr.open_dataset(era5_path, decode_timedelta=False).sortby("time")
    cerra_ds = xr.open_dataset(cerra_path, decode_timedelta=False).sortby("time")
    if variable not in era5_ds or variable not in cerra_ds:
        raise KeyError(
            f"Variable '{variable}' introuvable. ERA5={list(era5_ds.data_vars)}, CERRA={list(cerra_ds.data_vars)}"
        )

    era5 = era5_ds[variable]
    cerra = cerra_ds[variable]
    common_times = np.intersect1d(era5.time.values, cerra.time.values)
    if len(common_times) == 0:
        raise ValueError("Aucun timestep commun entre ERA5 et CERRA.")
    era5 = era5.sel(time=common_times)
    cerra = cerra.sel(time=common_times)
    era5_interp = era5.interp(latitude=cerra.latitude, longitude=cerra.longitude, method="linear")

    def clean(da):
        da = da.interpolate_na(dim="time", fill_value="extrapolate")
        return da.fillna(float(da.mean()))

    era5 = clean(era5).astype(np.float32)
    era5_interp = clean(era5_interp).astype(np.float32)
    cerra = clean(cerra).astype(np.float32)
    print(f"ERA5 original : {era5.shape}")
    print(f"ERA5 interp   : {era5_interp.shape}")
    print(f"CERRA target  : {cerra.shape}")
    return era5, era5_interp, cerra

def load_aligned_training_data(era5_path=DEFAULT_ERA5, cerra_path=DEFAULT_CERRA, variable=DEFAULT_VARIABLE):
    era5_ds = xr.open_dataset(era5_path, decode_timedelta=False).sortby("time")
    cerra_ds = xr.open_dataset(cerra_path, decode_timedelta=False).sortby("time")
    if variable not in era5_ds or variable not in cerra_ds:
        raise KeyError(
            f"Variable '{variable}' introuvable. ERA5={list(era5_ds.data_vars)}, CERRA={list(cerra_ds.data_vars)}"
        )

    era5 = era5_ds[variable]
    cerra = cerra_ds[variable]
    common_times = np.intersect1d(era5.time.values, cerra.time.values)
    if len(common_times) == 0:
        raise ValueError("Aucun timestep commun entre ERA5 et CERRA.")
    era5 = era5.sel(time=common_times)
    cerra = cerra.sel(time=common_times).astype(np.float32)
    era5_interp = era5.interp(latitude=cerra.latitude, longitude=cerra.longitude, method="linear").astype(np.float32)
    print(f"ERA5 interp training: {era5_interp.shape}")
    print(f"CERRA training      : {cerra.shape}")
    return era5_interp, cerra

def default_to_predict_path(variable=DEFAULT_VARIABLE):
    return TO_PREDICT_DIR / f"to_predict_{variable_tag(variable)}.nc"

def clean_prediction_slice(da):
    mean = da.mean(skipna=True)
    return da.fillna(mean).astype(np.float32)

def make_to_predict_file(
    era5_path=DEFAULT_ERA5,
    cerra_path=DEFAULT_CERRA,
    variable=DEFAULT_VARIABLE,
    train_ratio=TRAIN_RATIO,
    date=None,
    time_index=None,
    output_path=None,
):
    output_path = Path(output_path) if output_path is not None else default_to_predict_path(variable)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    era5_ds = xr.open_dataset(era5_path, decode_timedelta=False).sortby("time")
    cerra_ds = xr.open_dataset(cerra_path, decode_timedelta=False).sortby("time")
    try:
        if variable not in era5_ds or variable not in cerra_ds:
            raise KeyError(
                f"Variable '{variable}' introuvable. ERA5={list(era5_ds.data_vars)}, CERRA={list(cerra_ds.data_vars)}"
            )

        era5 = era5_ds[variable]
        cerra = cerra_ds[variable]
        common_times = np.intersect1d(era5.time.values, cerra.time.values)
        if len(common_times) == 0:
            raise ValueError("Aucun timestep commun entre ERA5 et CERRA.")

        if date:
            target = np.datetime64(date)
            idx = int(np.argmin(np.abs(common_times - target)))
            local_idx = None
        else:
            n_train = split_index(len(common_times), train_ratio)
            n_test = len(common_times) - n_train
            if n_test <= 0:
                raise ValueError("Le split test est vide. Diminue TRAIN_RATIO.")
            local_idx = np.random.randint(0, n_test) if time_index is None else int(time_index)
            local_idx = max(0, min(local_idx, n_test - 1))
            idx = n_train + local_idx

        selected_time = common_times[idx]
        print(f"Preparing to_predict date: {selected_time} (global index {idx})")
        era5_raw = clean_prediction_slice(era5.sel(time=[selected_time])).load()
        cerra_sel = clean_prediction_slice(cerra.sel(time=[selected_time])).load()
        era5_interp = clean_prediction_slice(
            era5_raw.interp(latitude=cerra_sel.latitude, longitude=cerra_sel.longitude, method="linear")
        ).load()

        ds = xr.Dataset(
            {
                "era5_raw": era5_raw.rename({"latitude": "era5_latitude", "longitude": "era5_longitude"}),
                "era5_interp": era5_interp,
                "cerra": cerra_sel,
            },
            attrs={
                "variable": variable,
                "selected_time": str(selected_time),
                "global_index": idx,
                "test_index": "" if local_idx is None else local_idx,
                "train_ratio": float(train_ratio),
                "source": "prepared prediction slice",
            },
        )
        ds.to_netcdf(output_path)
        ds.close()
    finally:
        era5_ds.close()
        cerra_ds.close()
    print(f"to_predict saved: {output_path}")
    return output_path

def load_to_predict_file(path, variable=DEFAULT_VARIABLE):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier to_predict introuvable: {path}. Cree-le avec: python make_to_predict.py --date YYYY-MM-DD"
        )
    ds = xr.open_dataset(path, decode_timedelta=False)
    ds.load()
    ds.close()
    saved_variable = ds.attrs.get("variable")
    if saved_variable and saved_variable != variable:
        raise ValueError(f"Le fichier {path} contient variable={saved_variable}, mais predict demande {variable}.")
    era5_raw = ds["era5_raw"].rename({"era5_latitude": "latitude", "era5_longitude": "longitude"})
    return era5_raw, ds["era5_interp"], ds["cerra"]

def load_prediction_inputs(args, train_ratio=TRAIN_RATIO):
    path = args.to_predict if args.to_predict is not None else default_to_predict_path(args.variable)
    if Path(path).exists():
        print(f"Using prepared to_predict file: {path}")
        return load_to_predict_file(path, args.variable)

    print(
        f"WARNING: {path} introuvable. Ancien mode utilise (charge toute la periode). "
        "Pour economiser la RAM, lance d'abord: python make_to_predict.py --date YYYY-MM-DD"
    )
    era5_raw, era5_interp, cerra = load_and_preprocess(args.era5, args.cerra, args.variable)
    return select_prediction_period(era5_raw, era5_interp, cerra, train_ratio, args.date, args.time_index)

def split_index(n_items, train_ratio=TRAIN_RATIO):
    return int(n_items * train_ratio)

def normalize_from_train(da, train_ratio=TRAIN_RATIO):
    n_train = split_index(len(da), train_ratio)
    train = da.isel(time=slice(0, n_train))
    mu = float(train.mean())
    std = float(train.std())
    std = std if std > 1e-8 else 1.0
    return (da - mu) / std, mu, std

def compute_train_mean_std(da, n_train, chunk_size=16):
    total = 0.0
    total_sq = 0.0
    count = 0
    for start in range(0, n_train, chunk_size):
        end = min(start + chunk_size, n_train)
        chunk = np.asarray(da.isel({da.dims[0]: slice(start, end)}).values, dtype=np.float32)
        valid = np.isfinite(chunk)
        if valid.any():
            values = chunk[valid].astype(np.float64, copy=False)
            total += float(values.sum(dtype=np.float64))
            total_sq += float((values * values).sum(dtype=np.float64))
            count += int(values.size)
        del chunk, valid
        cleanup_memory()
    if count == 0:
        raise ValueError("Aucune valeur finie pour calculer mean/std.")
    mean = total / count
    var = max(total_sq / count - mean * mean, 0.0)
    std = float(np.sqrt(var))
    std = std if std > 1e-8 else 1.0
    return float(mean), float(std)

def fit_pixelwise_linear_regression_streaming(x_da, y_da, n_train, x_mu, x_std, y_mu, y_std, chunk_size=16):
    spatial_shape = y_da.shape[-2:]
    sum_x = np.zeros(spatial_shape, dtype=np.float64)
    sum_y = np.zeros(spatial_shape, dtype=np.float64)
    sum_xy = np.zeros(spatial_shape, dtype=np.float64)
    sum_x2 = np.zeros(spatial_shape, dtype=np.float64)

    for start in range(0, n_train, chunk_size):
        end = min(start + chunk_size, n_train)
        x_chunk = np.asarray(x_da.isel({x_da.dims[0]: slice(start, end)}).values, dtype=np.float32)
        y_chunk = np.asarray(y_da.isel({y_da.dims[0]: slice(start, end)}).values, dtype=np.float32)
        x_chunk = np.where(np.isfinite(x_chunk), x_chunk, x_mu)
        y_chunk = np.where(np.isfinite(y_chunk), y_chunk, y_mu)
        x_chunk = (x_chunk - x_mu) / x_std
        y_chunk = (y_chunk - y_mu) / y_std
        sum_x += x_chunk.sum(axis=0, dtype=np.float64)
        sum_y += y_chunk.sum(axis=0, dtype=np.float64)
        sum_xy += (x_chunk * y_chunk).sum(axis=0, dtype=np.float64)
        sum_x2 += (x_chunk * x_chunk).sum(axis=0, dtype=np.float64)
        del x_chunk, y_chunk
        cleanup_memory()

    x_mean = sum_x / n_train
    y_mean = sum_y / n_train
    cov = (sum_xy / n_train) - (x_mean * y_mean)
    var = (sum_x2 / n_train) - (x_mean * x_mean)
    slope = cov / np.maximum(var, 1e-8)
    intercept = y_mean - slope * x_mean
    return slope.astype(np.float32), intercept.astype(np.float32)

def as_float32_array(data):
    return np.asarray(data, dtype=np.float32)

class GridDataset(Dataset):
    def __init__(self, x, y):
        self.x = as_float32_array(x)
        self.y = as_float32_array(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx][None]), torch.from_numpy(self.y[idx][None])

def make_loaders(
    x,
    y,
    train_ratio=TRAIN_RATIO,
    batch_size=BATCH_SIZE,
    max_train_samples=None,
    num_workers=0,
):
    n_train = split_index(len(x), train_ratio)
    if max_train_samples is not None:
        n_train = min(n_train, int(max_train_samples))
    pin_memory = torch.cuda.is_available()
    train = DataLoader(
        GridDataset(x[:n_train], y[:n_train]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test = DataLoader(
        GridDataset(x[split_index(len(x), train_ratio):], y[split_index(len(y), train_ratio):]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train, test

def predict_full(model, x, target_shape, mean, std, device=DEVICE, batch_size=BATCH_SIZE):
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    h, w = target_shape[-2], target_shape[-1]
    out = []
    with torch.no_grad():
        for i in range(0, len(xt), batch_size):
            xb = xt[i:i + batch_size].to(device)
            pred = model(xb)
            if pred.shape[-2:] != (h, w):
                pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
            out.append(pred.cpu().numpy())
    return np.concatenate(out, axis=0).squeeze(1) * std + mean

def evaluate(pred, true, name):
    flat_true = true.reshape(-1)
    flat_pred = pred.reshape(-1)
    valid = np.isfinite(flat_true) & np.isfinite(flat_pred)
    flat_true = flat_true[valid]
    flat_pred = flat_pred[valid]
    if len(flat_true) == 0:
        raise ValueError(f"Aucune valeur finie pour evaluer {name}.")
    mse = float(np.mean((flat_true - flat_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(flat_pred - flat_true)))
    bias = float(np.mean(flat_pred - flat_true))
    ss_res = float(np.sum((flat_true - flat_pred) ** 2))
    ss_tot = float(np.sum((flat_true - flat_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    print(f"{name}: RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} Bias={bias:.4f}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Bias": bias}

def save_prediction_nc(pred, template_da, path, model_name, variable, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(
        pred,
        dims=template_da.dims,
        coords=template_da.coords,
        name=f"{variable}_prediction",
        attrs={"model": model_name, "source": "ERA5 to CERRA downscaling"},
    )
    ds = da.to_dataset()
    ds.attrs.update(metrics)
    ds.to_netcdf(path)
    print(f"Prediction NetCDF saved: {path}")

def frame_index_from_date(times, date=None, time_index=None):
    if date:
        target = np.datetime64(date)
        values = np.asarray(times.values)
        return int(np.argmin(np.abs(values - target)))
    return 0 if time_index is None else int(time_index)

def select_prediction_period(era5_raw, era5_interp, cerra, train_ratio=TRAIN_RATIO, date=None, time_index=None):
    """Use one chosen date, one chosen test index, or one random test map by default."""
    if date:
        idx = frame_index_from_date(cerra.time, date)
        label = str(cerra.time.values[idx])
        print(f"Selected date: {label} (global index {idx})")
        return (
            era5_raw.isel(time=slice(idx, idx + 1)),
            era5_interp.isel(time=slice(idx, idx + 1)),
            cerra.isel(time=slice(idx, idx + 1)),
        )

    n_train = split_index(len(era5_interp), train_ratio)
    n_test = len(era5_interp) - n_train
    if n_test <= 0:
        raise ValueError("Le split test est vide. Diminue TRAIN_RATIO.")
    local_idx = np.random.randint(0, n_test) if time_index is None else int(time_index)
    local_idx = max(0, min(local_idx, n_test - 1))
    idx = n_train + local_idx
    label = str(cerra.time.values[idx])
    print(f"Selected random/test map: {label} (test index {local_idx}, global index {idx})")
    return (
        era5_raw.isel(time=slice(idx, idx + 1)),
        era5_interp.isel(time=slice(idx, idx + 1)),
        cerra.isel(time=slice(idx, idx + 1)),
    )

def plot_prediction(era5_raw, cerra, pred, model_name, out_dir, date=None, time_index=None, variable=METEO_VARIABLE):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = frame_index_from_date(cerra.time, date, time_index)
    idx = max(0, min(idx, len(cerra.time) - 1))

    era5_plot = values_for_plot(era5_raw.isel(time=idx).values, variable)
    cerra_plot = values_for_plot(cerra.isel(time=idx).values, variable)
    pred_plot = values_for_plot(pred[idx], variable)
    err_plot = np.abs(pred_plot - cerra_plot)
    cmap = plot_cmap(variable)
    unit = plot_unit(variable)
    # Use CERRA as the color reference so noisy diffusion outliers do not flatten the map contrast.
    vmin, vmax = plot_limits(cerra_plot, variable)
    err_vmax = plot_limits(err_plot, "error")[1]

    fields = [
        ("ERA5", era5_plot, era5_raw.longitude.values, era5_raw.latitude.values, cmap, vmin, vmax, unit),
        ("CERRA", cerra_plot, cerra.longitude.values, cerra.latitude.values, cmap, vmin, vmax, unit),
        (model_name, pred_plot, cerra.longitude.values, cerra.latitude.values, cmap, vmin, vmax, unit),
        ("Erreur abs.", err_plot, cerra.longitude.values, cerra.latitude.values, "Reds", 0.0, err_vmax, unit),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (title, data, lon, lat, cmap_name, field_vmin, field_vmax, field_unit) in zip(axes, fields):
        kwargs = {}
        if field_vmin is not None and field_vmax is not None:
            kwargs = {"vmin": field_vmin, "vmax": field_vmax}
        im = ax.pcolormesh(lon, lat, data, shading="auto", cmap=cmap_name, **kwargs)
        ax.set_title(title)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(field_unit)
    label = str(cerra.time.values[idx])
    fig.suptitle(f"{model_name} - {variable} ({unit}) - {label}")
    fig.tight_layout()
    path = out_dir / f"{model_name.lower().replace(' ', '_')}_{variable_tag(variable)}_{idx:03d}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")
