# python compare_selected_test_mean.py --reuse-cache --palette-mode ch --device cpu
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
    CERRA_DATA_PATH,
    DEFAULT_ERA5,
    DEFAULT_VARIABLE,
    TRAIN_RATIO,
    plot_cmap,
    plot_limits,
    plot_unit,
    split_index,
    values_for_plot,
    variable_kind,
    variable_tag,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "comparisons" / "selected_test"
OUT_DIR.mkdir(exist_ok=True)
PALETTE_MODE = "uniform"  # "uniform" meme echelle, "adaptive" chaque carte, "ch" DDPM/CorrDiff a part
KELVIN_OFFSET = np.float32(273.15)
_UNIT_ALIGNMENT_LOGGED = set()

SELECTED = {
    "Interpolation": None,
    "PyESD": ROOT / "PyESD" / "checkpoints" / f"pyesd_linear_{variable_tag(DEFAULT_VARIABLE)}.npz",
    "DeepSD": ROOT / "DeepSD" / "checkpoints" / f"deepsd_best_{variable_tag(DEFAULT_VARIABLE)}.pth",
    "CAE": ROOT / "CAE" / "checkpoints" / f"cae_best_{variable_tag(DEFAULT_VARIABLE)}.pth",
    "ESRGAN": ROOT / "ESRGAN" / "checkpoints" / f"esrgan_generator_{variable_tag(DEFAULT_VARIABLE)}.pth",
    "DDPM": ROOT / "DDPM" / "checkpoints" / f"ddpm_cosine_best_{variable_tag(DEFAULT_VARIABLE)}.pth",
    "CorrDiff": ROOT / "CorrDiff" / "checkpoints" / f"corrdiff_score_{variable_tag(DEFAULT_VARIABLE)}.pth",
}


def import_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_test_chunk(era5, cerra, times):
    era5_chunk = era5.sel(time=times).load()
    cerra_chunk = cerra.sel(time=times).load().astype(np.float32)
    era5_interp = era5_chunk.interp(latitude=cerra.latitude, longitude=cerra.longitude, method="linear")
    era5_interp = era5_interp.fillna(float(era5_interp.mean())).astype(np.float32)
    cerra_chunk = cerra_chunk.fillna(float(cerra_chunk.mean())).astype(np.float32)
    return era5_interp, cerra_chunk


def iter_test_chunks(chunk_size, max_test_samples=None):
    era5_ds = xr.open_dataset(DEFAULT_ERA5, decode_timedelta=False).sortby("time")
    cerra_ds = xr.open_dataset(CERRA_DATA_PATH, decode_timedelta=False).sortby("time")
    if DEFAULT_VARIABLE not in era5_ds or DEFAULT_VARIABLE not in cerra_ds:
        raise KeyError(f"Variable {DEFAULT_VARIABLE} absente dans ERA5 ou CERRA.")

    era5 = era5_ds[DEFAULT_VARIABLE]
    cerra = cerra_ds[DEFAULT_VARIABLE]
    common_times = np.intersect1d(era5.time.values, cerra.time.values)
    n_train = split_index(len(common_times), TRAIN_RATIO)
    test_times = common_times[n_train:]
    if max_test_samples is not None:
        test_times = test_times[: int(max_test_samples)]
    if len(test_times) == 0:
        raise ValueError("Aucune donnee test disponible.")

    for start in range(0, len(test_times), chunk_size):
        times = test_times[start : start + chunk_size]
        yield load_test_chunk(era5, cerra, times)


def init_stats(template):
    shape = template.shape[-2:]
    return {
        "pred_sum": np.zeros(shape, dtype=np.float64),
        "true_sum": np.zeros(shape, dtype=np.float64),
        "sse_map": np.zeros(shape, dtype=np.float64),
        "sae_map": np.zeros(shape, dtype=np.float64),
        "count_map": np.zeros(shape, dtype=np.float64),
        "sse": 0.0,
        "sae": 0.0,
        "count": 0,
        "n_maps": 0,
        "lat": template.latitude.values,
        "lon": template.longitude.values,
    }


def align_prediction_units(pred, true, model_name=None):
    if variable_kind(DEFAULT_VARIABLE) != "temperature":
        return pred

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


def update_stats(stats, pred, true, model_name=None):
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    pred = align_prediction_units(pred, true, model_name)
    valid = np.isfinite(pred) & np.isfinite(true)
    stats["pred_sum"] += np.nan_to_num(pred, nan=0.0).sum(axis=0, dtype=np.float64)
    stats["true_sum"] += np.nan_to_num(true, nan=0.0).sum(axis=0, dtype=np.float64)
    sq = np.where(valid, (pred.astype(np.float64) - true.astype(np.float64)) ** 2, 0.0)
    abs_err = np.where(valid, np.abs(pred.astype(np.float64) - true.astype(np.float64)), 0.0)
    stats["sse_map"] += sq.sum(axis=0, dtype=np.float64)
    stats["sae_map"] += abs_err.sum(axis=0, dtype=np.float64)
    stats["count_map"] += valid.sum(axis=0, dtype=np.float64)
    diff = pred[valid].astype(np.float64) - true[valid].astype(np.float64)
    stats["sse"] += float(np.sum(diff * diff))
    stats["sae"] += float(np.sum(np.abs(diff)))
    stats["count"] += int(valid.sum())
    stats["n_maps"] += int(pred.shape[0])


def finalize_stats(stats):
    return {
        "mean_pred": stats["pred_sum"] / max(stats["n_maps"], 1),
        "mean_true": stats["true_sum"] / max(stats["n_maps"], 1),
        "rmse_map": np.sqrt(stats["sse_map"] / np.maximum(stats["count_map"], 1.0)),
        "rmse": float(np.sqrt(stats["sse"] / max(stats["count"], 1))),
        "mae_map": stats["sae_map"] / np.maximum(stats["count_map"], 1.0),
        "mae": float(stats["sae"] / max(stats["count"], 1)),
        "n_maps": stats["n_maps"],
        "lat": stats["lat"],
        "lon": stats["lon"],
    }


def torch_predict_full(model, x, target_shape, mean, std, device):
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(1)
    h, w = target_shape
    with torch.no_grad():
        pred = model(xt)
        if pred.shape[-2:] != (h, w):
            pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
    return pred.detach().cpu().numpy().squeeze(1) * float(std) + float(mean)


def torch_predict_ddpm_full(model, x, target_shape, mean, std, steps, device):
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(1)
    h, w = target_shape
    out = []
    with torch.no_grad():
        cond = F.interpolate(xt, size=(h, w), mode="bilinear", align_corners=False)
        y = torch.randn(len(xt), 1, h, w, device=device)
        for _ in range(steps):
            y = y - model(cond, y) / steps
        out.append(y.detach().cpu().numpy())
    return np.concatenate(out, axis=0).squeeze(1) * float(std) + float(mean)


def compute_model(name, checkpoint, args, device):
    if name in {"Interpolation", "Interpolation lineaire"}:
        stats = None
        for x_da, y_da in iter_test_chunks(args.chunk_size, args.max_test_samples):
            pred = x_da.values.astype(np.float32)
            stats = stats or init_stats(y_da)
            update_stats(stats, pred, y_da.values, name)
        return finalize_stats(stats)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint}")

    stats = None
    if name == "PyESD":
        ckpt = np.load(checkpoint)
        for x_da, y_da in iter_test_chunks(args.chunk_size, args.max_test_samples):
            x = ((x_da - float(ckpt["x_mu"])) / float(ckpt["x_std"])).values
            pred = (x * ckpt["slope"] + ckpt["intercept"]) * float(ckpt["y_std"]) + float(ckpt["y_mu"])
            stats = stats or init_stats(y_da)
            update_stats(stats, pred, y_da.values, name)

    elif name == "DeepSD":
        module = import_from_path("selected_deepsd_model", ROOT / "DeepSD" / "model.py")
        ckpt = torch.load(checkpoint, map_location=device)
        model = module.DeepSD(scale=int(ckpt["scale"])).to(device)
        model.load_state_dict(ckpt["model_state"])
        for x_da, y_da in iter_test_chunks(args.chunk_size, args.max_test_samples):
            x = ((x_da - ckpt["x_mu"]) / ckpt["x_std"]).values
            pred = torch_predict_full(model, x, y_da.shape[-2:], ckpt["y_mu"], ckpt["y_std"], device)
            stats = stats or init_stats(y_da)
            update_stats(stats, pred, y_da.values, name)

    elif name == "CAE":
        module = import_from_path("selected_cae_model", ROOT / "CAE" / "model.py")
        ckpt = torch.load(checkpoint, map_location=device)
        model = module.CAE(
            channels=int(ckpt.get("channels", 32)),
            latent_channels=int(ckpt.get("latent_channels", 128)),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        for x_da, y_da in iter_test_chunks(args.chunk_size, args.max_test_samples):
            x = ((x_da - ckpt["x_mu"]) / ckpt["x_std"]).values
            pred = torch_predict_full(model, x, y_da.shape[-2:], ckpt["y_mu"], ckpt["y_std"], device)
            stats = stats or init_stats(y_da)
            update_stats(stats, pred, y_da.values, name)

    elif name == "ESRGAN":
        module = import_from_path("selected_esrgan_model", ROOT / "ESRGAN" / "model.py")
        ckpt = torch.load(checkpoint, map_location=device)
        model = module.ESRGenerator(
            channels=int(ckpt.get("channels", 64)),
            num_rrdb=int(ckpt.get("rrdb_blocks", 4)),
        ).to(device)
        model.load_state_dict(ckpt["generator_state"])
        for x_da, y_da in iter_test_chunks(args.chunk_size, args.max_test_samples):
            x = ((x_da - ckpt["x_mu"]) / ckpt["x_std"]).values
            pred = torch_predict_full(model, x, y_da.shape[-2:], ckpt["y_mu"], ckpt["y_std"], device)
            stats = stats or init_stats(y_da)
            update_stats(stats, pred, y_da.values, name)

    elif name == "DDPM":
        module = import_from_path("selected_ddpm_model", ROOT / "DDPM" / "model.py")
        ckpt = torch.load(checkpoint, map_location=device)
        model = module.CosineDDPM().to(device)
        model.load_state_dict(ckpt["model_state"])
        steps = args.ddpm_steps or int(ckpt.get("steps", module.DENOISING_STEPS))
        for x_da, y_da in iter_test_chunks(args.chunk_size, args.max_test_samples):
            x = ((x_da - ckpt["x_mu"]) / ckpt["x_std"]).values
            pred = torch_predict_ddpm_full(model, x, y_da.shape[-2:], ckpt["y_mu"], ckpt["y_std"], steps, device)
            pred = apply_bilateral_filter(pred, window=args.bilateral_window, sigma_spatial=args.bilateral_sigma_spatial)
            stats = stats or init_stats(y_da)
            update_stats(stats, pred, y_da.values, name)

    elif name == "CorrDiff":
        ckpt = load_torch_checkpoint(checkpoint, device)
        if "slope" not in ckpt or "intercept" not in ckpt:
            raise KeyError("Checkpoint CorrDiff ancien format. Relance CorrDiff/train.py.")
        model = ScoreUNet(in_channels=4, channels=int(ckpt["channels"])).to(device)
        model.load_state_dict(ckpt["model_state"])
        for x_da, y_da in iter_test_chunks(args.chunk_size, args.max_test_samples):
            x = ((x_da - ckpt["x_mu"]) / ckpt["x_std"]).values.astype(np.float32)
            baseline = linear_baseline_predict(x, ckpt["slope"], ckpt["intercept"])
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
                batch_size=args.chunk_size,
            )
            pred = (baseline + residual) * float(ckpt["y_std"]) + float(ckpt["y_mu"])
            pred = apply_bilateral_filter(pred, window=args.bilateral_window, sigma_spatial=args.bilateral_sigma_spatial)
            stats = stats or init_stats(y_da)
            update_stats(stats, pred, y_da.values, name)

    return finalize_stats(stats)


def use_uniform_palette():
    return PALETTE_MODE.lower() in {"uniform", "uniforme", "unforme", "shared", "share","same","unified", "global"}


def use_ch_palette():
    return PALETTE_MODE.lower() in {"ch", "ddpm"}


def has_ch_palette(name):
    special = {"DDPM", "DDMP"}
    if PALETTE_MODE.lower() == "ch":
        special.add("CORRDIFF")
    return str(name).upper() in special


def image_color_settings(image):
    arr = values_for_plot(image, DEFAULT_VARIABLE)
    finite = arr[np.isfinite(arr)]
    vmin, vmax = plot_limits(finite, DEFAULT_VARIABLE) if finite.size else (None, None)
    return arr, {"cmap": plot_cmap(DEFAULT_VARIABLE), "vmin": vmin, "vmax": vmax}


def shared_color_settings(images):
    arrays = [values_for_plot(image, DEFAULT_VARIABLE) for image in images]
    valid = [arr[np.isfinite(arr)].reshape(-1) for arr in arrays if np.isfinite(arr).any()]
    flat = np.concatenate(valid) if valid else np.array([], dtype=np.float32)
    vmin, vmax = plot_limits(flat, DEFAULT_VARIABLE) if flat.size else (None, None)
    return arrays, {"cmap": plot_cmap(DEFAULT_VARIABLE), "vmin": vmin, "vmax": vmax}


def plot_model_pair(name, result, path):
    unit = plot_unit(DEFAULT_VARIABLE)
    if use_uniform_palette() or (use_ch_palette() and not has_ch_palette(name)):
        (true_plot, pred_plot), colors = shared_color_settings([result["mean_true"], result["mean_pred"]])
        true_colors = colors
        pred_colors = colors
    else:
        true_plot, true_colors = image_color_settings(result["mean_true"])
        pred_plot, pred_colors = image_color_settings(result["mean_pred"])
    mae_map = result["mae_map"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), squeeze=False)
    axes = axes[0]
    im = axes[0].pcolormesh(result["lon"], result["lat"], true_plot, shading="auto", **true_colors)
    axes[0].set_title("CERRA")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04).set_label(unit)
    im = axes[1].pcolormesh(result["lon"], result["lat"], pred_plot, shading="auto", **pred_colors)
    axes[1].set_title(f"{name}\nMAE={result['mae']:.3f}")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04).set_label(unit)
    im = axes[2].pcolormesh(result["lon"], result["lat"], mae_map, shading="auto", cmap="Reds", vmin=0.0)
    axes[2].set_title(f"MAE par pixel\nMAE global={result['mae']:.3f}")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04).set_label(unit)
    for ax in axes:
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_box_aspect(1)
    fig.suptitle(f"{name} - moyenne test - {DEFAULT_VARIABLE}")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Model mean map saved: {path}")


def plot_mean_maps(results, path):
    names = list(results)
    unit = plot_unit(DEFAULT_VARIABLE)

    ncols = 3
    nplots = len(names) + 1
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)
    axes = axes.reshape(-1)

    first = next(iter(results.values()))
    ch_names = [name for name in names if use_ch_palette() and has_ch_palette(name)]
    ch_pred_plots = {}
    ch_colors = None

    if use_uniform_palette() or use_ch_palette():
        shared_names = [name for name in names if name not in ch_names]
        images = [first["mean_true"], *[results[name]["mean_pred"] for name in shared_names]]
        plots, shared_colors = shared_color_settings(images)
        true_plot = plots[0]
        pred_plots = dict(zip(shared_names, plots[1:]))
        true_colors = shared_colors
        if ch_names:
            ch_plots, ch_colors = shared_color_settings([results[name]["mean_pred"] for name in ch_names])
            ch_pred_plots = dict(zip(ch_names, ch_plots))
    else:
        true_plot, true_colors = image_color_settings(first["mean_true"])
        pred_plots = {}

    im = axes[0].pcolormesh(first["lon"], first["lat"], true_plot, shading="auto", **true_colors)
    axes[0].set_title(f"CERRA moyenne test\nn={first['n_maps']}")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04).set_label(unit)

    for ax, name in zip(axes[1:], names):
        result = results[name]
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
            pred_plot, pred_colors = image_color_settings(result["mean_pred"])
        im = ax.pcolormesh(result["lon"], result["lat"], pred_plot, shading="auto", **pred_colors)
        ax.set_title(f"{name}\nMAE={result['mae']:.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(unit)

    for ax in axes[nplots:]:
        ax.axis("off")
    for ax in axes[:nplots]:
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_box_aspect(1)

    fig.suptitle(f"Cartes moyennes sur les donnees test - {DEFAULT_VARIABLE}")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Mean maps saved: {path}")


def plot_mae(results, path):
    names = list(results)
    mae_values = [results[name]["mae"] for name in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, mae_values, color="#4777b3")
    ax.set_ylabel("MAE")
    ax.set_title(f"MAE sur les donnees test - {DEFAULT_VARIABLE}")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, mae_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"MAE plot saved: {path}")


def plot_mae_maps(results, path):
    names = list(results)
    vmax = None
    ch_vmax = None
    if use_uniform_palette() or use_ch_palette():
        valid = [
            result["mae_map"][np.isfinite(result["mae_map"])].reshape(-1)
            for name, result in results.items()
            if not (use_ch_palette() and has_ch_palette(name))
            if np.isfinite(result["mae_map"]).any()
        ]
        flat = np.concatenate(valid) if valid else np.array([], dtype=np.float32)
        vmax = float(np.nanpercentile(flat, 98)) if flat.size else None
        if use_ch_palette():
            valid_ch = [
                result["mae_map"][np.isfinite(result["mae_map"])].reshape(-1)
                for name, result in results.items()
                if has_ch_palette(name)
                if np.isfinite(result["mae_map"]).any()
            ]
            flat_ch = np.concatenate(valid_ch) if valid_ch else np.array([], dtype=np.float32)
            ch_vmax = float(np.nanpercentile(flat_ch, 98)) if flat_ch.size else None

    ncols = 3
    nplots = len(names)
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)
    axes = axes.reshape(-1)

    for ax, name in zip(axes, names):
        result = results[name]
        mae_map = result["mae_map"]
        current_vmax = vmax
        if use_ch_palette() and has_ch_palette(name):
            current_vmax = ch_vmax
        im = ax.pcolormesh(
            result["lon"],
            result["lat"],
            mae_map,
            shading="auto",
            cmap="Reds",
            vmin=0.0,
            vmax=current_vmax,
        )
        ax.set_title(f"{name}\nMAE global={result['mae']:.3f}")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_box_aspect(1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(plot_unit(DEFAULT_VARIABLE))

    for ax in axes[nplots:]:
        ax.axis("off")
    fig.suptitle(f"Cartes MAE sur les donnees test - {DEFAULT_VARIABLE}")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"MAE maps saved: {path}")


def save_results_csv(results, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("model,MAE,RMSE,n_test_maps\n")
        for name, result in results.items():
            f.write(f"{name},{result['mae']},{result['rmse']},{result['n_maps']}\n")
    print(f"MAE csv saved: {path}")


def default_cache_path(suffix):
    return OUT_DIR / f"selected_test_results_cache_{suffix}.npz"


def save_results_cache(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(results)
    payload = {"models": np.array(names, dtype=str)}
    scalar_keys = ["mae", "rmse", "n_maps"]
    array_keys = ["mean_pred", "mean_true", "rmse_map", "mae_map", "lat", "lon"]
    for idx, name in enumerate(names):
        result = results[name]
        for key in scalar_keys:
            payload[f"{key}_{idx}"] = np.array(result[key])
        for key in array_keys:
            payload[f"{key}_{idx}"] = np.asarray(result[key])
    np.savez_compressed(path, **payload)
    print(f"Results cache saved: {path}")


def load_results_cache(path, requested_models=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cache introuvable: {path}. Relance sans --reuse-cache une premiere fois.")
    data = np.load(path, allow_pickle=False)
    names = [str(name) for name in data["models"]]
    results = {}
    for idx, name in enumerate(names):
        results[name] = {
            "mean_pred": data[f"mean_pred_{idx}"],
            "mean_true": data[f"mean_true_{idx}"],
            "rmse_map": data[f"rmse_map_{idx}"],
            "mae_map": data[f"mae_map_{idx}"],
            "lat": data[f"lat_{idx}"],
            "lon": data[f"lon_{idx}"],
            "mae": float(data[f"mae_{idx}"]),
            "rmse": float(data[f"rmse_{idx}"]),
            "n_maps": int(data[f"n_maps_{idx}"]),
        }
    if requested_models is not None:
        missing = [name for name in requested_models if name not in results]
        if missing:
            print(f"WARNING: modeles absents du cache {path}: {missing}")
        available = [name for name in requested_models if name in results]
        if not available:
            raise KeyError(f"Aucun des modeles demandes n'est present dans le cache {path}: {requested_models}")
        results = {name: results[name] for name in available}
    print(f"Results cache loaded: {path}")
    return results


def main():
    global PALETTE_MODE
    parser = argparse.ArgumentParser(description="Compare selected models on test mean maps.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SELECTED),
        choices=list(SELECTED),
        help="Subset of models to evaluate.",
    )
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--corrdiff-sample-steps", type=int, default=20)
    parser.add_argument("--corrdiff-langevin-steps", type=int, default=1)
    parser.add_argument("--corrdiff-snr", type=float, default=0.1)
    parser.add_argument("--ddpm-steps", type=int, default=None)
    parser.add_argument("--bilateral-window", type=int, default=5)
    parser.add_argument("--bilateral-sigma-spatial", type=float, default=2.0)
    parser.add_argument(
        "--palette-mode",
        choices=["uniform", "uniforme", "unforme", "share", "same", "shared", "global", "adaptive", "adaptative", "ch", "ddpm"],
        default=PALETTE_MODE,
        help=(
            "uniforme: meme echelle couleur; "
            "adaptive/adaptative: echelle propre a chaque carte; "
            "ch: DDPM et CorrDiff ont une palette adaptee, les autres modeles partagent la meme palette; "
            "ddpm: seul DDPM a une palette adaptee, tous les autres modeles partagent la meme palette."
        ),
    )
    parser.add_argument("--reuse-cache", action="store_true", help="Recharge les resultats calcules sans refaire les predictions.")
    parser.add_argument("--no-save-cache", action="store_true", help="Ne sauvegarde pas le cache apres calcul.")
    parser.add_argument("--cache-path", type=Path, default=None, help="Chemin du cache .npz a lire/ecrire.")
    args = parser.parse_args()
    PALETTE_MODE = args.palette_mode

    suffix = variable_tag(DEFAULT_VARIABLE)
    if args.max_test_samples is not None:
        suffix += f"_n{args.max_test_samples}"
    cache_path = args.cache_path if args.cache_path is not None else default_cache_path(suffix)

    if args.reuse_cache:
        results = load_results_cache(cache_path, args.models)
    else:
        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
        results = {}
        failures = {}
        for name in args.models:
            checkpoint = SELECTED[name]
            print(f"\n=== {name} ===")
            try:
                results[name] = compute_model(name, checkpoint, args, device)
                print(f"{name}: MAE={results[name]['mae']:.4f} n={results[name]['n_maps']}")
            except Exception as exc:
                failures[name] = str(exc)
                print(f"SKIP {name}: {exc}")

        if not results:
            raise RuntimeError("Aucun modele n'a pu etre evalue.")
        if failures:
            print("\nModeles non traces:")
            for name, reason in failures.items():
                print(f"  - {name}: {reason}")
        if not args.no_save_cache:
            save_results_cache(results, cache_path)

    plot_mean_maps(results, OUT_DIR / f"selected_test_mean_maps_{suffix}.png")
    for name, result in results.items():
        safe_name = name.lower().replace(" ", "_")
        plot_model_pair(name, result, OUT_DIR / f"selected_test_{safe_name}_mean_{suffix}.png")
    plot_mae_maps(results, OUT_DIR / f"selected_test_mae_maps_{suffix}.png")
    plot_mae(results, OUT_DIR / f"selected_test_mae_{suffix}.png")
    save_results_csv(results, OUT_DIR / f"selected_test_mae_{suffix}.csv")


if __name__ == "__main__":
    main()
