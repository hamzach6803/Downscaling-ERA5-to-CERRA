# python results_extreme90.py --use-saved-predictions --palette-mode ch
import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from corrdiff_core import (
    ScoreUNet,
    apply_bilateral_filter,
    linear_baseline_predict,
    load_torch_checkpoint,
    sample_residual,
)
from utils import (
    DEFAULT_VARIABLE,
    evaluate,
    plot_cmap,
    plot_unit,
    values_for_plot,
    variable_kind,
    variable_tag,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
OUT_DIR = ROOT / "outputs" / "comparisons" / "selected_test" / "results_90times"
PRED_DIR = OUT_DIR / "predictions"
PALETTE_MODE = "uniform"  # "uniform" meme echelle, "adaptive" chaque carte, "ch" DDPM/CorrDiff a part
RESULTS_LABEL = "90 cartes"

DEFAULT_ERA5_90 = DATA_DIR / "era5_2025_90times.nc"
DEFAULT_CERRA_90 = DATA_DIR / "cerra_2025_90times.nc"
PREFIX_ALL_MODELS = "all_models"
KELVIN_OFFSET = np.float32(273.15)
_UNIT_ALIGNMENT_LOGGED = set()
MODELS = {
    "Interpolation": None,
    "PyESD": ROOT / "PyESD" / "checkpoints" / f"pyesd_linear_{variable_tag(DEFAULT_VARIABLE)}.npz",
    "DeepSD": ROOT / "DeepSD" / "checkpoints" / f"deepsd_best_{variable_tag(DEFAULT_VARIABLE)}.pth",
    "CAE": ROOT / "CAE" / "checkpoints" / f"cae_best_{variable_tag(DEFAULT_VARIABLE)}.pth",
    "ESRGAN": ROOT / "ESRGAN" / "checkpoints" / f"esrgan_generator_{variable_tag(DEFAULT_VARIABLE)}.pth",
    "DDPM": ROOT / "DDPM" / "checkpoints" / f"ddpm_cosine_best_{variable_tag(DEFAULT_VARIABLE)}.pth",
    "CorrDiff": ROOT / "CorrDiff" / "checkpoints" / f"corrdiff_score_{variable_tag(DEFAULT_VARIABLE)}.pth",
}


def align_prediction_units(pred, true, variable, model_name=None):
    if variable_kind(variable) != "temperature":
        return pred

    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    pred_finite = pred[np.isfinite(pred)]
    true_finite = true[np.isfinite(true)]
    if pred_finite.size == 0 or true_finite.size == 0:
        return pred

    pred_median = float(np.nanmedian(pred_finite))
    true_median = float(np.nanmedian(true_finite))
    label = model_name or "prediction"
    if true_median > 100.0 and pred_median < 100.0:
        if label not in _UNIT_ALIGNMENT_LOGGED:
            print(f"{label}: conversion automatique degC -> K avant calcul des metriques.")
            _UNIT_ALIGNMENT_LOGGED.add(label)
        return pred + KELVIN_OFFSET
    if true_median < 100.0 and pred_median > 100.0:
        if label not in _UNIT_ALIGNMENT_LOGGED:
            print(f"{label}: conversion automatique K -> degC avant calcul des metriques.")
            _UNIT_ALIGNMENT_LOGGED.add(label)
        return pred - KELVIN_OFFSET
    return pred


def import_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_90_data(era5_path, cerra_path, variable):
    era5_ds = xr.open_dataset(era5_path, decode_timedelta=False).sortby("time")
    cerra_ds = xr.open_dataset(cerra_path, decode_timedelta=False).sortby("time")
    if variable not in era5_ds or variable not in cerra_ds:
        raise KeyError(
            f"Variable '{variable}' introuvable. ERA5={list(era5_ds.data_vars)}, "
            f"CERRA={list(cerra_ds.data_vars)}"
        )

    era5 = era5_ds[variable]
    cerra = cerra_ds[variable].astype(np.float32)
    common_times = np.intersect1d(era5.time.values, cerra.time.values)
    if len(common_times) == 0:
        raise ValueError("Aucun timestep commun entre ERA5 et CERRA.")

    era5 = era5.sel(time=common_times).load()
    cerra = cerra.sel(time=common_times).load()
    era5_interp = era5.interp(latitude=cerra.latitude, longitude=cerra.longitude, method="linear")
    era5_interp = era5_interp.fillna(float(era5_interp.mean())).astype(np.float32).load()
    cerra = cerra.fillna(float(cerra.mean())).astype(np.float32).load()
    era5_ds.close()
    cerra_ds.close()
    print(f"ERA5 90 interp : {era5_interp.shape}")
    print(f"CERRA 90 target: {cerra.shape}")
    return era5_interp, cerra


def resize_batch(data, target_shape):
    arr = np.asarray(data, dtype=np.float32)
    if arr.shape[-2:] == tuple(target_shape):
        return arr
    tensor = torch.tensor(arr, dtype=torch.float32)
    squeeze = False
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
        squeeze = True
    resized = F.interpolate(tensor.unsqueeze(1), size=target_shape, mode="bilinear", align_corners=False)
    resized = resized.squeeze(1).numpy()
    return resized[0] if squeeze else resized


def predict_torch_full(model, x, target_shape, y_mu, y_std, device, batch_size):
    model.eval()
    h, w = target_shape
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start:start + batch_size], dtype=torch.float32, device=device).unsqueeze(1)
            pred = model(xb)
            if pred.shape[-2:] != (h, w):
                pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
            out.append(pred.detach().cpu().numpy())
    return np.concatenate(out, axis=0).squeeze(1) * float(y_std) + float(y_mu)


def predict_ddpm_full(model, x, target_shape, y_mu, y_std, steps, device, batch_size):
    model.eval()
    h, w = target_shape
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start:start + batch_size], dtype=torch.float32, device=device).unsqueeze(1)
            cond = F.interpolate(xb, size=(h, w), mode="bilinear", align_corners=False)
            y = torch.randn(len(xb), 1, h, w, device=device)
            for _ in range(steps):
                y = y - model(cond, y) / steps
            out.append(y.detach().cpu().numpy())
    return np.concatenate(out, axis=0).squeeze(1) * float(y_std) + float(y_mu)


def compute_prediction(name, checkpoint, era5_interp, cerra, args, device):
    if name in {"Interpolation", "Interpolation lineaire"}:
        return np.asarray(era5_interp.values, dtype=np.float32)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint}")

    if name == "PyESD":
        ckpt = np.load(checkpoint)
        x = ((era5_interp - float(ckpt["x_mu"])) / float(ckpt["x_std"])).values
        slope = resize_batch(ckpt["slope"], cerra.shape[-2:])
        intercept = resize_batch(ckpt["intercept"], cerra.shape[-2:])
        pred = (x * slope + intercept) * float(ckpt["y_std"]) + float(ckpt["y_mu"])

    elif name == "DeepSD":
        module = import_from_path("results90_deepsd_model", ROOT / "DeepSD" / "model.py")
        ckpt = torch.load(checkpoint, map_location=device)
        model = module.DeepSD(scale=int(ckpt["scale"])).to(device)
        model.load_state_dict(ckpt["model_state"])
        x = ((era5_interp - ckpt["x_mu"]) / ckpt["x_std"]).values
        pred = predict_torch_full(model, x, cerra.shape[-2:], ckpt["y_mu"], ckpt["y_std"], device, args.batch_size)

    elif name == "CAE":
        module = import_from_path("results90_cae_model", ROOT / "CAE" / "model.py")
        ckpt = torch.load(checkpoint, map_location=device)
        model = module.CAE(
            channels=int(ckpt.get("channels", 32)),
            latent_channels=int(ckpt.get("latent_channels", 128)),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        x = ((era5_interp - ckpt["x_mu"]) / ckpt["x_std"]).values
        pred = predict_torch_full(model, x, cerra.shape[-2:], ckpt["y_mu"], ckpt["y_std"], device, args.batch_size)

    elif name == "ESRGAN":
        module = import_from_path("results90_esrgan_model", ROOT / "ESRGAN" / "model.py")
        ckpt = torch.load(checkpoint, map_location=device)
        model = module.ESRGenerator(
            channels=int(ckpt.get("channels", 64)),
            num_rrdb=int(ckpt.get("rrdb_blocks", 4)),
        ).to(device)
        model.load_state_dict(ckpt["generator_state"])
        x = ((era5_interp - ckpt["x_mu"]) / ckpt["x_std"]).values
        pred = predict_torch_full(model, x, cerra.shape[-2:], ckpt["y_mu"], ckpt["y_std"], device, args.batch_size)

    elif name == "DDPM":
        module = import_from_path("results90_ddpm_model", ROOT / "DDPM" / "model.py")
        ckpt = torch.load(checkpoint, map_location=device)
        model = module.CosineDDPM().to(device)
        model.load_state_dict(ckpt["model_state"])
        x = ((era5_interp - ckpt["x_mu"]) / ckpt["x_std"]).values
        steps = args.ddpm_steps or int(ckpt.get("steps", module.DENOISING_STEPS))
        pred = predict_ddpm_full(model, x, cerra.shape[-2:], ckpt["y_mu"], ckpt["y_std"], steps, device, args.batch_size)
        pred = apply_bilateral_filter(pred, window=args.bilateral_window, sigma_spatial=args.bilateral_sigma_spatial)

    elif name == "CorrDiff":
        ckpt = load_torch_checkpoint(checkpoint, device)
        if "slope" not in ckpt or "intercept" not in ckpt:
            raise KeyError("Checkpoint CorrDiff ancien format. Relance CorrDiff/train.py.")
        x = ((era5_interp - ckpt["x_mu"]) / ckpt["x_std"]).values.astype(np.float32)
        slope = resize_batch(ckpt["slope"], cerra.shape[-2:])
        intercept = resize_batch(ckpt["intercept"], cerra.shape[-2:])
        baseline = linear_baseline_predict(x, slope, intercept)
        model = ScoreUNet(in_channels=4, channels=int(ckpt["channels"])).to(device)
        model.load_state_dict(ckpt["model_state"])
        residual = sample_residual(
            model,
            x,
            baseline,
            None,
            device,
            float(ckpt["sigma_min"]),
            float(ckpt["sigma_max"]),
            sample_steps=args.corrdiff_sample_steps,
            langevin_steps=args.corrdiff_langevin_steps,
            snr=args.corrdiff_snr,
            batch_size=args.batch_size,
        )
        pred = (baseline + residual) * float(ckpt["y_std"]) + float(ckpt["y_mu"])
        pred = apply_bilateral_filter(pred, window=args.bilateral_window, sigma_spatial=args.bilateral_sigma_spatial)

    else:
        raise ValueError(f"Modele non supporte: {name}")

    pred = np.asarray(pred, dtype=np.float32)
    if pred.shape != cerra.shape:
        raise ValueError(f"{name}: forme prediction {pred.shape}, attendu {cerra.shape}")
    return pred


def rmse(pred, true):
    return float(np.sqrt(np.nanmean((np.asarray(pred) - np.asarray(true)) ** 2)))


def mae(pred, true):
    return float(np.nanmean(np.abs(np.asarray(pred) - np.asarray(true))))


def save_prediction_nc(name, pred, template, variable):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    tag = variable_tag(variable)
    path = PRED_DIR / f"{name.lower().replace(' ', '_')}_pred_90times_{tag}.nc"
    da = xr.DataArray(
        pred,
        dims=template.dims,
        coords=template.coords,
        name=f"{variable}_prediction",
        attrs={"model": name, "variable": variable, "bilateral_filter": str(name == "CorrDiff")},
    )
    da.to_dataset().to_netcdf(path)
    print(f"Prediction saved: {path}")


def load_prediction_nc(name, template, variable):
    tag = variable_tag(variable)
    path = PRED_DIR / f"{name.lower().replace(' ', '_')}_pred_90times_{tag}.nc"
    if not path.exists():
        raise FileNotFoundError(f"Prediction sauvegardee introuvable: {path}")

    with xr.open_dataset(path, decode_timedelta=False) as ds:
        var_name = f"{variable}_prediction"
        if var_name not in ds:
            if len(ds.data_vars) != 1:
                raise KeyError(f"Variable '{var_name}' introuvable dans {path}")
            var_name = next(iter(ds.data_vars))
        da = ds[var_name].load()

    if tuple(da.shape) != tuple(template.shape):
        raise ValueError(f"{name}: forme prediction sauvegardee {da.shape}, attendu {template.shape}")
    return np.asarray(da.values, dtype=np.float32)


def image_color_settings(image, variable):
    arr = values_for_plot(image, variable)
    finite = arr[np.isfinite(arr)]
    colors = {"cmap": plot_cmap(variable)}
    if finite.size:
        colors["vmin"] = float(np.nanmin(finite))
        colors["vmax"] = float(np.nanmax(finite))
    return arr, colors


def shared_color_settings(images, variable):
    arrays = [values_for_plot(image, variable) for image in images]
    colors = {"cmap": plot_cmap(variable)}
    valid_arrays = [arr[np.isfinite(arr)].reshape(-1) for arr in arrays if np.isfinite(arr).any()]
    flat = np.concatenate(valid_arrays) if valid_arrays else np.array([], dtype=np.float32)
    if flat.size:
        colors["vmin"] = float(np.nanmin(flat))
        colors["vmax"] = float(np.nanmax(flat))
    return arrays, colors


def use_uniform_palette():
    return PALETTE_MODE.lower() in {"uniform", "uniforme", "unforme", "shared", "unified", "global"}


def use_ch_palette():
    return PALETTE_MODE.lower() in {"ch", "ddpm"}


def has_ch_palette(name):
    special = {"DDPM", "DDMP"}
    if PALETTE_MODE.lower() == "ch":
        special.add("CORRDIFF")
    return str(name).upper() in special

def plot_model_pair(name, pred_map, true_map, lon, lat, variable, title, path, score=None, mae_map=None):
    unit = plot_unit(variable)
    if use_uniform_palette() or (use_ch_palette() and not has_ch_palette(name)):
        (true_plot, pred_plot), shared_colors = shared_color_settings([true_map, pred_map], variable)
        true_colors = shared_colors
        pred_colors = shared_colors
    else:
        true_plot, true_colors = image_color_settings(true_map, variable)
        pred_plot, pred_colors = image_color_settings(pred_map, variable)
    if score is None:
        score = mae(pred_plot, true_plot)
    if mae_map is None:
        err_map = np.abs(pred_plot - true_plot)
        err_title = f"Erreur absolue\nMAE={score:.3f}"
    else:
        err_map = mae_map
        err_title = f"MAE par pixel\nMAE global={score:.3f}"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), squeeze=False)
    axes = axes[0]
    im = axes[0].pcolormesh(lon, lat, true_plot, shading="auto", **true_colors)
    axes[0].set_title("CERRA")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04).set_label(unit)
    im = axes[1].pcolormesh(lon, lat, pred_plot, shading="auto", **pred_colors)
    axes[1].set_title(f"{name}\nMAE={score:.3f}")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04).set_label(unit)
    im = axes[2].pcolormesh(lon, lat, err_map, shading="auto", cmap="Reds", vmin=0.0)
    axes[2].set_title(err_title)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04).set_label(unit)
    for ax in axes:
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_box_aspect(1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")


def plot_all_models(pred_maps, true_map, lon, lat, variable, title, path, scores=None):
    names = list(pred_maps)
    unit = plot_unit(variable)
    nplots = len(names) + 1
    ncols = 3
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)
    axes = axes.reshape(-1)

    ch_names = [name for name in names if use_ch_palette() and has_ch_palette(name)]
    ch_pred_plots = {}
    ch_colors = None

    if use_uniform_palette() or use_ch_palette():
        shared_names = [name for name in names if name not in ch_names]
        all_images = [true_map, *[pred_maps[name] for name in shared_names]]
        plots, shared_colors = shared_color_settings(all_images, variable)
        true_plot = plots[0]
        pred_plots = dict(zip(shared_names, plots[1:]))
        true_colors = shared_colors
        if ch_names:
            ch_plots, ch_colors = shared_color_settings([pred_maps[name] for name in ch_names], variable)
            ch_pred_plots = dict(zip(ch_names, ch_plots))
    else:
        true_plot, true_colors = image_color_settings(true_map, variable)
        pred_plots = {}
    im = axes[0].pcolormesh(lon, lat, true_plot, shading="auto", **true_colors)
    axes[0].set_title("CERRA")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04).set_label(unit)

    for ax, name in zip(axes[1:], names):
        if use_uniform_palette():
            pred_plot = pred_plots[name]
            pred_colors = shared_colors
        elif use_ch_palette() and name not in ch_names:
            pred_plot = pred_plots[name]
            pred_colors = shared_colors
        elif use_ch_palette() and name in ch_names:
            pred_plot = ch_pred_plots[name]
            pred_colors = ch_colors
        else:
            pred_plot, pred_colors = image_color_settings(pred_maps[name], variable)
        score = scores[name] if scores is not None else mae(pred_plot, true_plot)
        im = ax.pcolormesh(lon, lat, pred_plot, shading="auto", **pred_colors)
        ax.set_title(f"{name}\nMAE={score:.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(unit)

    for ax in axes[nplots:]:
        ax.axis("off")
    for ax in axes[:nplots]:
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_box_aspect(1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")


def plot_mae_maps(predictions, cerra, variable, title, path, scores=None):
    names = list(predictions)
    lon = cerra.longitude.values
    lat = cerra.latitude.values
    true = np.asarray(cerra.values, dtype=np.float32)
    mae_maps = {
        name: np.nanmean(np.abs(np.asarray(pred, dtype=np.float32) - true), axis=0)
        for name, pred in predictions.items()
    }
    vmax = None
    ch_vmax = None
    if use_uniform_palette() or use_ch_palette():
        shared_maps = [
            arr
            for name, arr in mae_maps.items()
            if not (use_ch_palette() and has_ch_palette(name))
        ]
        valid = [arr[np.isfinite(arr)].reshape(-1) for arr in shared_maps if np.isfinite(arr).any()]
        flat = np.concatenate(valid) if valid else np.array([], dtype=np.float32)
        vmax = float(np.nanpercentile(flat, 98)) if flat.size else None
        if use_ch_palette():
            ch_maps = [
                arr
                for name, arr in mae_maps.items()
                if has_ch_palette(name)
            ]
            valid_ch = [arr[np.isfinite(arr)].reshape(-1) for arr in ch_maps if np.isfinite(arr).any()]
            flat_ch = np.concatenate(valid_ch) if valid_ch else np.array([], dtype=np.float32)
            ch_vmax = float(np.nanpercentile(flat_ch, 98)) if flat_ch.size else None

    ncols = 3
    nplots = len(names)
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)
    axes = axes.reshape(-1)

    for ax, name in zip(axes, names):
        mae_map = mae_maps[name]
        score = scores[name] if scores is not None else float(np.nanmean(mae_map))
        current_vmax = vmax
        if use_ch_palette() and has_ch_palette(name):
            current_vmax = ch_vmax
        im = ax.pcolormesh(lon, lat, mae_map, shading="auto", cmap="Reds", vmin=0.0, vmax=current_vmax)
        ax.set_title(f"{name}\nMAE global={score:.3f}")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_box_aspect(1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(plot_unit(variable))

    for ax in axes[nplots:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"MAE map saved: {path}")


def subset_by_case_type(predictions, cerra, case_type):
    if "case_type" not in cerra.coords:
        return None, None
    mask = np.asarray(cerra["case_type"].values) == case_type
    if not mask.any():
        return None, None
    return {name: pred[mask] for name, pred in predictions.items()}, cerra.isel(time=mask)


def selected_time_index(cerra, date, time_index, seed=None):
    if date is not None:
        target = np.datetime64(date)
        idx = int(np.argmin(np.abs(cerra.time.values - target)))
        print(f"Date demandee: {date} -> date utilisee: {cerra.time.values[idx]} (index {idx})")
        return idx
    if time_index is not None:
        idx = max(0, min(int(time_index), len(cerra.time) - 1))
        print(f"Index demande: {time_index} -> date utilisee: {cerra.time.values[idx]} (index {idx})")
        return idx
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(cerra.time)))
    print(f"Date aleatoire utilisee: {cerra.time.values[idx]} (index {idx})")
    return idx


def make_plots(predictions, cerra, args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = variable_tag(args.variable)
    lon = cerra.longitude.values
    lat = cerra.latitude.values

    groups = [(RESULTS_LABEL, "90times", predictions, cerra)]
    for label, suffix, case_type in (("45 cas froids", "45cold", "cold"), ("45 cas chauds", "45hot", "hot")):
        subset_predictions, subset_cerra = subset_by_case_type(predictions, cerra, case_type)
        if subset_predictions is not None:
            groups.append((label, suffix, subset_predictions, subset_cerra))

    group_score_rows = []
    for label, suffix, group_predictions, group_cerra in groups:
        group_lon = group_cerra.longitude.values
        group_lat = group_cerra.latitude.values
        group_true = np.asarray(group_cerra.values, dtype=np.float32)
        group_scores = {
            name: mae(pred, group_true)
            for name, pred in group_predictions.items()
        }
        for name, score in group_scores.items():
            group_score_rows.append((suffix, label, name, score, len(group_cerra.time)))
        true_mean = group_cerra.mean("time").values
        mean_pred_maps = {name: pred.mean(axis=0) for name, pred in group_predictions.items()}
        mae_maps = {
            name: np.nanmean(np.abs(np.asarray(pred, dtype=np.float32) - group_true), axis=0)
            for name, pred in group_predictions.items()
        }
        plot_all_models(
            mean_pred_maps,
            true_mean,
            group_lon,
            group_lat,
            args.variable,
            f"Moyenne des {label} - {args.variable}",
            OUT_DIR / f"{PREFIX_ALL_MODELS}_mean_{suffix}_{tag}.png",
            scores=group_scores,
        )
        for name, pred_map in mean_pred_maps.items():
            plot_model_pair(
                name,
                pred_map,
                true_mean,
                group_lon,
                group_lat,
                args.variable,
                f"{name} - moyenne des {label}",
                OUT_DIR / f"{name.lower().replace(' ', '_')}_mean_{suffix}_{tag}.png",
                score=group_scores[name],
                mae_map=mae_maps[name],
            )

        plot_mae_maps(
            group_predictions,
            group_cerra,
            args.variable,
            f"Cartes MAE sur les {label} - {args.variable}",
            OUT_DIR / f"{PREFIX_ALL_MODELS}_mae_maps_{suffix}_{tag}.png",
            scores=group_scores,
        )

    save_group_mae_csv(group_score_rows, args.variable)

    idx = selected_time_index(cerra, args.date, args.time_index, args.seed)
    selected_time = str(cerra.time.values[idx])
    true_date = cerra.isel(time=idx).values
    date_pred_maps = {name: pred[idx] for name, pred in predictions.items()}
    date_suffix = selected_time[:10].replace("-", "")
    plot_all_models(
        date_pred_maps,
        true_date,
        lon,
        lat,
        args.variable,
        f"Carte du {selected_time} - {args.variable}",
        OUT_DIR / f"{PREFIX_ALL_MODELS}_date_{date_suffix}_{tag}.png",
    )
    for name, pred_map in date_pred_maps.items():
        plot_model_pair(
            name,
            pred_map,
            true_date,
            lon,
            lat,
            args.variable,
            f"{name} - carte du {selected_time}",
            OUT_DIR / f"{name.lower().replace(' ', '_')}_date_{date_suffix}_{tag}.png",
        )


def save_metrics(predictions, cerra, variable):
    path = OUT_DIR / f"metrics_90times_{variable_tag(variable)}.csv"
    with open(path, "w", encoding="utf-8") as f:
        f.write("model,RMSE,MAE,R2,Bias,n_maps,bilateral_filter\n")
        for name, pred in predictions.items():
            metrics = evaluate(pred, cerra.values, name)
            f.write(
                f"{name},{metrics['RMSE']},{metrics['MAE']},{metrics['R2']},"
                f"{metrics['Bias']},{len(cerra.time)},{name == 'CorrDiff'}\n"
            )
    print(f"Metrics saved: {path}")


def save_group_mae_csv(rows, variable):
    path = OUT_DIR / f"mae_groups_{variable_tag(variable)}.csv"
    with open(path, "w", encoding="utf-8") as f:
        f.write("group_suffix,group_label,model,MAE,n_maps\n")
        for suffix, label, name, score, n_maps in rows:
            f.write(f"{suffix},{label},{name},{score},{n_maps}\n")
    print(f"Group MAE saved: {path}")


def main():
    global PALETTE_MODE
    parser = argparse.ArgumentParser(description="Predire et tracer les resultats sur les 90 cartes 2025.")
    parser.add_argument("--era5", type=Path, default=DEFAULT_ERA5_90)
    parser.add_argument("--cerra", type=Path, default=DEFAULT_CERRA_90)
    parser.add_argument("--variable", default=DEFAULT_VARIABLE)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--date", default=None, help="Date a tracer, ex: 2025-01-03 ou 2025-01-03T12:00")
    parser.add_argument(
        "--time-index",
        "--time_index",
        dest="time_index",
        type=int,
        default=None,
        help="Index a tracer si --date n'est pas fourni. Si absent, une date aleatoire est choisie.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Graine optionnelle pour reproduire la date aleatoire.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--corrdiff-sample-steps", type=int, default=20)
    parser.add_argument("--corrdiff-langevin-steps", type=int, default=1)
    parser.add_argument("--corrdiff-snr", type=float, default=0.15)
    parser.add_argument("--ddpm-steps", type=int, default=None)
    parser.add_argument("--bilateral-window", type=int, default=5)
    parser.add_argument("--bilateral-sigma-spatial", type=float, default=2.0)
    parser.add_argument(
        "--palette-mode",
        choices=["uniform", "uniforme", "unforme", "shared", "global", "adaptive", "adaptative", "ch", "ddpm"],
        default=PALETTE_MODE,
        help=(
            "uniform: meme echelle pour toutes les cartes; adaptive/adaptative: echelle propre a chaque carte; "
            "ch: DDPM et CorrDiff utilisent une palette adaptee, les autres modeles partagent la meme palette; "
            "ddpm: seul DDPM a une palette adaptee, tous les autres modeles partagent la meme palette."
        ),
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--use-saved-predictions",
        action="store_true",
        help="Charger les predictions .nc deja sauvegardees au lieu de recalculer les modeles.",
    )
    args = parser.parse_args()
    PALETTE_MODE = args.palette_mode

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    era5_interp, cerra = load_90_data(args.era5, args.cerra, args.variable)

    predictions = {}
    failures = {}
    for name in args.models:
        print(f"\n=== {name} ===")
        try:
            if args.use_saved_predictions:
                pred = load_prediction_nc(name, cerra, args.variable)
            else:
                pred = compute_prediction(name, MODELS[name], era5_interp, cerra, args, device)
            pred = align_prediction_units(pred, cerra.values, args.variable, name)
            predictions[name] = pred
            print(f"{name}: MAE={mae(pred, cerra.values):.4f}")
            if args.save_predictions and not args.use_saved_predictions:
                save_prediction_nc(name, pred, cerra, args.variable)
        except Exception as exc:
            failures[name] = str(exc)
            print(f"SKIP {name}: {exc}")

    if not predictions:
        raise RuntimeError("Aucun modele n'a pu etre predit.")
    if failures:
        print("\nModeles non traces:")
        for name, reason in failures.items():
            print(f"  - {name}: {reason}")

    make_plots(predictions, cerra, args)
    save_metrics(predictions, cerra, args.variable)


if __name__ == "__main__":
    main()
