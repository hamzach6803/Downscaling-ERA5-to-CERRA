import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
import yaml
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

from results_90times import MODELS as CHECKPOINTS
from results_90times import compute_prediction as compute_model_prediction

try:
    import cartopy.crs as ccrs
except Exception:
    ccrs = None


MODEL_FILES = {
    "Interpolation": "interpolation_pred_90times_{variable}.nc",
    "PyESD": "pyesd_pred_90times_{variable}.nc",
    "DeepSD": "deepsd_pred_90times_{variable}.nc",
    "ESRGAN": "esrgan_pred_90times_{variable}.nc",
    "DDPM": "ddpm_pred_90times_{variable}.nc",
    "CorrDiff": "corrdiff_pred_90times_{variable}.nc",
}
DEFAULT_MODELS = [model for model in MODEL_FILES if (model != "DDPM" and model != "CorrDiff")]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse pixel par pixel d'une vague de froid/chaleur avec land mask."
    )
    parser.add_argument("--date", default=None, help="Date civile a analyser, ex: 2025-08-11.")
    parser.add_argument(
        "--time-index",
        "--time_index",
        dest="time_index",
        type=int,
        default=None,
        help=(
            "Avec --date: heure a analyser parmi 0, 6, 12, 18. "
            "Sans --date: index temporel global dans les fichiers NetCDF."
        ),
    )
    parser.add_argument("--type_event", required=True, choices=["heat", "cold"], help="Type d'evenement.")
    parser.add_argument(
        "--model",
        default=None,
        choices=list(MODEL_FILES),
        help="Modele a comparer a CERRA. Si absent, tous les modeles sauf DDPM sont traites.",
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yml"))
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=None,
        help="Prediction NetCDF optionnelle. Si absent, la prediction est calculee directement.",
    )
    parser.add_argument(
        "--use-saved-predictions",
        action="store_true",
        help="Lire les predictions NetCDF sauvegardees au lieu de calculer directement le modele.",
    )
    parser.add_argument("--land-mask-file", type=Path, default=None, help="Land mask NetCDF optionnel.")
    parser.add_argument("--land-mask-variable", default=None, help="Variable du land mask optionnelle.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--corrdiff-sample-steps", type=int, default=20)
    parser.add_argument("--corrdiff-langevin-steps", type=int, default=1)
    parser.add_argument("--corrdiff-snr", type=float, default=0.15)
    parser.add_argument("--ddpm-steps", type=int, default=None)
    parser.add_argument("--bilateral-window", type=int, default=5)
    parser.add_argument("--bilateral-sigma-spatial", type=float, default=2.0)
    args = parser.parse_args()
    if args.date is None and args.time_index is None:
        parser.error("Il faut fournir --date ou --time-index.")
    if args.date is not None and args.time_index is not None and args.time_index not in {0, 6, 12, 18}:
        parser.error("Avec --date, --time-index doit etre une heure parmi 0, 6, 12, 18.")
    return args


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier config introuvable: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config, path.parent.resolve()


def resolve_path(path_value, base_dir):
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def require_section(config, name):
    section = config.get(name)
    if not isinstance(section, dict):
        raise KeyError(f"Section '{name}' manquante ou invalide dans config.yml.")
    return section


def variable_tag(variable):
    return str(variable).replace("/", "_").replace("\\", "_").replace(" ", "_")


def prediction_path_from_model(config, base_dir, model, variable, override=None):
    if override is not None:
        return resolve_path(override, Path.cwd())
    data_cfg = require_section(config, "data")
    predictions_dir = data_cfg.get("predictions_dir")
    if not predictions_dir:
        raise KeyError("Champ 'data.predictions_dir' manquant dans config.yml.")
    filename = MODEL_FILES[model].format(variable=variable_tag(variable))
    return resolve_path(predictions_dir, base_dir) / filename


def selected_models(model, prediction_file):
    if model is not None:
        return [model]
    if prediction_file is not None:
        raise ValueError("--prediction-file ne peut etre utilise que avec --model.")
    return DEFAULT_MODELS


def load_netcdf_data(path, variable, role):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{role}: fichier NetCDF introuvable: {path}")
    ds = xr.open_dataset(path, decode_timedelta=False).sortby("time") if role != "land_mask" else xr.open_dataset(path)
    try:
        if variable in ds.data_vars:
            da = ds[variable]
        elif role == "prediction" and f"{variable}_prediction" in ds.data_vars:
            da = ds[f"{variable}_prediction"]
        elif role == "prediction" and len(ds.data_vars) == 1:
            da = ds[next(iter(ds.data_vars))]
        else:
            raise KeyError(
                f"{role}: variable '{variable}' introuvable dans {path}. "
                f"Variables disponibles: {list(ds.data_vars)}"
            )
        da = standardize_lat_lon_names(da).load()
    finally:
        ds.close()
    return da


def standardize_lat_lon_names(da):
    rename = {}
    if "lat" in da.dims or "lat" in da.coords:
        rename["lat"] = "latitude"
    if "lon" in da.dims or "lon" in da.coords:
        rename["lon"] = "longitude"
    if rename:
        da = da.rename(rename)
    if "latitude" not in da.coords or "longitude" not in da.coords:
        raise ValueError("Les donnees doivent contenir des coordonnees latitude/longitude ou lat/lon.")
    da = normalize_longitudes(da)
    return da.sortby(["latitude", "longitude"])


def normalize_longitudes(da):
    lon = da["longitude"]
    if float(lon.max()) > 180.0:
        new_lon = ((lon + 180.0) % 360.0) - 180.0
        da = da.assign_coords(longitude=new_lon).sortby("longitude")
    return da


def hours_from_times(times):
    values = np.asarray(times)
    days = values.astype("datetime64[D]")
    hours = values.astype("datetime64[h]") - days.astype("datetime64[h]")
    return (hours / np.timedelta64(1, "h")).astype(int)


def select_time(da, date_text=None, time_index=None):
    if "time" not in da.dims:
        return da

    times = da["time"].values
    if date_text is None:
        idx = int(time_index)
        if idx < 0 or idx >= len(times):
            raise IndexError(f"--time-index hors limites: {idx}. Indices valides: 0 a {len(times) - 1}.")
        selected = da.isel(time=idx)
        print(f"Index temporel demande: {idx} -> timestamp utilise: {times[idx]}")
        return selected

    target_day = np.datetime64(date_text, "D")
    days = times.astype("datetime64[D]")
    day_matches = np.where(days == target_day)[0]
    if len(day_matches) == 0:
        first = str(times[0])[:10]
        last = str(times[-1])[:10]
        raise ValueError(f"Date absente du fichier NetCDF: {date_text}. Periode disponible: {first} a {last}.")

    if time_index is None:
        idx = int(day_matches[0])
    else:
        hours = hours_from_times(times)
        matches = [idx for idx in day_matches if int(hours[idx]) == int(time_index)]
        if not matches:
            available = sorted({int(hours[idx]) for idx in day_matches})
            available_text = ", ".join(f"{hour:02d}h" for hour in available)
            raise ValueError(
                f"Heure {time_index:02d}h absente pour la date {date_text}. "
                f"Heures disponibles: {available_text}."
            )
        idx = int(matches[0])

    selected = da.isel(time=idx)
    print(f"Date demandee: {date_text} -> timestamp utilise: {times[idx]}")
    return selected


def time_label(da, fallback_date=None, fallback_index=None):
    if "time" in da.coords:
        timestamp = np.datetime64(da["time"].values, "h")
        day = str(timestamp.astype("datetime64[D]"))
        hour = int((timestamp - timestamp.astype("datetime64[D]")) / np.timedelta64(1, "h"))
        return f"{day}_{hour:02d}h"
    if fallback_date is not None:
        return str(fallback_date)
    return f"time_index_{fallback_index}"


def validate_domain(domain):
    required = ["lat_min", "lat_max", "lon_min", "lon_max"]
    missing = [key for key in required if key not in domain]
    if missing:
        raise KeyError(f"Domaine incomplet dans config.yml. Champs manquants: {missing}")
    lat_min = float(domain["lat_min"])
    lat_max = float(domain["lat_max"])
    lon_min = float(domain["lon_min"])
    lon_max = float(domain["lon_max"])
    if lat_min >= lat_max or lon_min >= lon_max:
        raise ValueError("Domaine mal defini: lat_min < lat_max et lon_min < lon_max sont requis.")
    return {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max}


def crop_domain(da, domain):
    lat = da["latitude"]
    lon = da["longitude"]
    lat_slice = slice(domain["lat_min"], domain["lat_max"])
    if float(lat[0]) > float(lat[-1]):
        lat_slice = slice(domain["lat_max"], domain["lat_min"])
    cropped = da.sel(latitude=lat_slice, longitude=slice(domain["lon_min"], domain["lon_max"]))
    if cropped.sizes.get("latitude", 0) == 0 or cropped.sizes.get("longitude", 0) == 0:
        raise ValueError("Le domaine demande ne recoupe pas les donnees.")
    return cropped


def load_land_mask(config, base_dir, target_da, domain, file_override=None, var_override=None):
    mask_cfg = require_section(config, "land_mask")
    mask_file = file_override or mask_cfg.get("file")
    mask_var = var_override or mask_cfg.get("variable")
    land_threshold = float(mask_cfg.get("land_threshold", 0.5))
    if not mask_file:
        raise KeyError("Champ 'land_mask.file' manquant dans config.yml.")
    if not mask_var:
        raise KeyError("Champ 'land_mask.variable' manquant dans config.yml.")

    mask_path = resolve_path(mask_file, base_dir) if file_override is None else resolve_path(file_override, Path.cwd())
    mask = load_netcdf_data(mask_path, mask_var, "land_mask")
    for dim in ("time", "band", "number", "step"):
        if dim in mask.dims:
            mask = mask.isel({dim: 0})
    mask = crop_domain(mask, domain)
    mask = mask.interp(latitude=target_da.latitude, longitude=target_da.longitude, method="nearest")
    land_mask = (np.isfinite(mask) & (mask >= land_threshold)).astype(bool)
    n_land = int(land_mask.sum().item())
    n_total = int(land_mask.size)
    print(f"Land mask applique: {n_land}/{n_total} pixels terre avec seuil >= {land_threshold}")
    return land_mask


def apply_land_mask(da, land_mask):
    if da.sizes.get("latitude") != land_mask.sizes.get("latitude") or da.sizes.get("longitude") != land_mask.sizes.get("longitude"):
        raise ValueError("Dimensions incompatibles entre la carte et le land mask interpole.")
    return da.where(land_mask)


def with_time_dim(da):
    if "time" in da.dims:
        return da
    if "time" in da.coords:
        time_value = da["time"].values
        da = da.reset_coords("time", drop=True)
    else:
        time_value = 0
    return da.expand_dims(time=[time_value])


def direct_prediction_da(model, era5_interp, cerra, args, variable):
    if variable_tag(variable) != variable_tag("t2m"):
        raise ValueError("La prediction directe utilise les checkpoints t2m disponibles dans results_90times.py.")
    if model not in CHECKPOINTS:
        raise ValueError(f"Modele non supporte pour prediction directe: {model}")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    era5_batch = with_time_dim(era5_interp)
    cerra_batch = with_time_dim(cerra)
    pred = compute_model_prediction(model, CHECKPOINTS[model], era5_batch, cerra_batch, args, device)
    pred2d = np.asarray(pred, dtype=np.float32)[0]
    return xr.DataArray(
        pred2d,
        dims=("latitude", "longitude"),
        coords={
            "latitude": cerra.latitude,
            "longitude": cerra.longitude,
            "time": cerra.coords["time"] if "time" in cerra.coords else era5_interp.coords.get("time"),
        },
        name=f"{variable}_prediction",
        attrs={"model": model, "source": "direct_model_prediction"},
    )


def get_threshold(config, type_event):
    thresholds = require_section(config, "thresholds")
    key = "heat_threshold" if type_event == "heat" else "cold_threshold"
    if key not in thresholds:
        raise KeyError(f"Seuil manquant dans config.yml: thresholds.{key}")
    unit = str(thresholds.get("unit", "C")).strip().lower()
    if unit not in {"c", "degc", "celsius", "degree", "degrees"}:
        raise ValueError("Les seuils doivent etre exprimes en degres Celsius: thresholds.unit doit valoir 'C'.")
    return float(thresholds[key])


def temperature_to_celsius(da):
    values = np.asarray(da.values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size and float(np.nanmedian(finite)) > 100.0:
        return da - 273.15
    return da


def threshold_input(da, variable):
    if str(variable).lower() in {"t2m", "temperature", "temp"}:
        return temperature_to_celsius(da)
    return da


def get_event_labels(config, type_event):
    defaults = {
        "heat": {"event": "evenement chaleur", "no_event": "pas d'evenement chaleur"},
        "cold": {"event": "evenement froid", "no_event": "pas d'evenement froid"},
    }
    labels = config.get("labels", {})
    event_labels = labels.get(type_event, {}) if isinstance(labels, dict) else {}
    return {
        "event": str(event_labels.get("event", defaults[type_event]["event"])),
        "no_event": str(event_labels.get("no_event", defaults[type_event]["no_event"])),
    }


def apply_threshold(da, type_event, threshold, land_mask=None):
    valid = np.isfinite(da)
    if type_event == "heat":
        binary = xr.where(valid, xr.where(da > threshold, 1.0, 0.0), np.nan)
    else:
        binary = xr.where(valid, xr.where(da < threshold, 1.0, 0.0), np.nan)
    if land_mask is not None:
        binary = binary.where(land_mask)
    return binary.astype(np.float32)


def ensure_compatible(cerra, pred):
    if tuple(cerra.sizes[dim] for dim in ("latitude", "longitude")) != tuple(
        pred.sizes[dim] for dim in ("latitude", "longitude")
    ):
        raise ValueError(f"Dimensions incompatibles: CERRA={cerra.shape}, prediction={pred.shape}")
    if not np.allclose(cerra.latitude.values, pred.latitude.values) or not np.allclose(
        cerra.longitude.values, pred.longitude.values
    ):
        raise ValueError("Grilles incompatibles: les coordonnees latitude/longitude different.")


def compute_contingency(cerra_binary, pred_binary):
    c = np.asarray(cerra_binary.values)
    p = np.asarray(pred_binary.values)
    valid = np.isfinite(c) & np.isfinite(p)
    c = c[valid].astype(np.int8)
    p = p[valid].astype(np.int8)
    if c.size == 0:
        raise ValueError("Aucun pixel valide apres application du land mask.")
    h = int(np.sum((c == 1) & (p == 1)))
    m = int(np.sum((c == 1) & (p == 0)))
    f = int(np.sum((c == 0) & (p == 1)))
    cn = int(np.sum((c == 0) & (p == 0)))
    return {"H": h, "M": m, "F": f, "CN": cn, "valid_pixels": int(c.size)}


def safe_div(num, den):
    return float(num / den) if den else float("nan")


def compute_scores(contingency):
    h = contingency["H"]
    m = contingency["M"]
    f = contingency["F"]
    cn = contingency["CN"]
    total = h + m + f + cn
    return {
        "POD": safe_div(h, h + m),
        "FAR": safe_div(f, h + f),
        "CSI": safe_div(h, h + m + f),
        "Accuracy": safe_div(h + cn, total),
    }


def display_path(path_value, base_dir):
    path = resolve_path(path_value, base_dir)
    return path.name


def build_source_note(config, base_dir, variable, model=None, prediction_file=None, use_saved_predictions=False):
    figure_cfg = config.get("figure_info", {})
    if isinstance(figure_cfg, dict) and not bool(figure_cfg.get("show_file_info", True)):
        return None

    data_cfg = require_section(config, "data")
    mask_cfg = require_section(config, "land_mask")
    parts = []
    if data_cfg.get("cerra_file"):
        parts.append(f"CERRA: {display_path(data_cfg['cerra_file'], base_dir)}")
    if data_cfg.get("era5_file"):
        parts.append(f"ERA5: {display_path(data_cfg['era5_file'], base_dir)}")
    if model is None:
        if use_saved_predictions:
            predictions_dir = data_cfg.get("predictions_dir", "")
            if predictions_dir:
                parts.append(f"Predictions: {display_path(predictions_dir, base_dir)}")
        else:
            parts.append("Predictions: calcul direct")
    elif prediction_file is not None or use_saved_predictions:
        pred_path = prediction_path_from_model(config, base_dir, model, variable, prediction_file)
        parts.append(f"{model}: {pred_path.name}")
    else:
        parts.append(f"{model}: calcul direct")
    if mask_cfg.get("file"):
        parts.append(f"Land mask: {display_path(mask_cfg['file'], base_dir)}")
    parts.append(f"Variable: {variable}")
    return "Fichiers - " + " | ".join(parts)


def add_figure_source_note(fig, source_note):
    if not source_note:
        return
    fig.text(
        0.5,
        0.012,
        source_note,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#475569",
        wrap=True,
    )


def binary_plot_settings():
    cmap = ListedColormap(["#d9d9d9", "#d7191c"])
    cmap.set_bad(color="white", alpha=1.0)
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    return cmap, norm


def binary_legend_handles(event_labels):
    return [
        Patch(facecolor="#d9d9d9", edgecolor="#334155", label=f"0: {event_labels['no_event']}"),
        Patch(facecolor="#d7191c", edgecolor="#334155", label=f"1: {event_labels['event']}"),
    ]


def add_binary_legend(fig, event_labels, y=0.06):
    fig.legend(
        handles=binary_legend_handles(event_labels),
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=2,
        frameon=False,
        fontsize=9,
        handlelength=1.6,
        columnspacing=1.8,
    )


def draw_binary_legend_axis(ax, event_labels):
    ax.axis("off")
    ax.set_title("Legende", fontweight="bold", pad=8)
    entries = [
        ("#d9d9d9", f"0: {event_labels['no_event']}"),
        ("#d7191c", f"1: {event_labels['event']}"),
    ]
    for y, (color, label) in zip((0.58, 0.42), entries):
        ax.add_patch(
            Rectangle(
                (0.18, y - 0.045),
                0.08,
                0.09,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="#334155",
                linewidth=0.9,
            )
        )
        ax.text(0.31, y, label, transform=ax.transAxes, va="center", ha="left", fontsize=10, color="#111827")


def map_subplot_kwargs():
    if ccrs is None:
        return {}
    return {"projection": ccrs.PlateCarree()}


def pcolormesh_kwargs():
    if ccrs is None:
        return {}
    return {"transform": ccrs.PlateCarree()}


def format_longitude(value, _position=None):
    value = float(value)
    suffix = "E" if value >= 0 else "W"
    return f"{abs(value):.1f}{suffix}"


def format_latitude(value, _position=None):
    value = float(value)
    suffix = "N" if value >= 0 else "S"
    return f"{abs(value):.1f}{suffix}"


def add_coordinate_labels(ax):
    if ccrs is not None and hasattr(ax, "gridlines"):
        grid = ax.gridlines(
            draw_labels=True,
            linewidth=0.35,
            color="#64748b",
            alpha=0.55,
            linestyle="--",
        )
        grid.top_labels = False
        grid.right_labels = False
        grid.xlabel_style = {"size": 8, "color": "#334155"}
        grid.ylabel_style = {"size": 8, "color": "#334155"}
        return

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(format_longitude))
    ax.yaxis.set_major_formatter(FuncFormatter(format_latitude))
    ax.grid(True, color="#94a3b8", linewidth=0.35, alpha=0.55, linestyle="--")
    ax.tick_params(axis="both", labelsize=8, colors="#334155")


def set_map_aspect(ax):
    if ccrs is None:
        ax.set_box_aspect(1)


def add_map_overlays(ax, land_mask):
    if land_mask is None:
        return
    if ccrs is not None:
        lon_min = float(land_mask.longitude.min())
        lon_max = float(land_mask.longitude.max())
        lat_min = float(land_mask.latitude.min())
        lat_max = float(land_mask.latitude.max())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.coastlines(resolution="10m", color="black", linewidth=0.9)

    mask_values = np.asarray(land_mask.values, dtype=np.float32)
    finite = mask_values[np.isfinite(mask_values)]
    if finite.size == 0 or float(np.nanmin(finite)) == float(np.nanmax(finite)):
        return
    kwargs = pcolormesh_kwargs()
    ax.contour(
        land_mask.longitude.values,
        land_mask.latitude.values,
        mask_values,
        levels=[0.5],
        colors="black",
        linewidths=0.8,
        **kwargs,
    )


def plot_binary_map(binary_da, title, path, event_labels, land_mask=None, source_note=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmap, norm = binary_plot_settings()
    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=map_subplot_kwargs())
    ax.set_facecolor("white")
    im = ax.pcolormesh(
        binary_da.longitude.values,
        binary_da.latitude.values,
        np.ma.masked_invalid(binary_da.values),
        shading="auto",
        cmap=cmap,
        norm=norm,
        **pcolormesh_kwargs(),
    )
    add_map_overlays(ax, land_mask)
    ax.set_title(title)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    add_coordinate_labels(ax)
    set_map_aspect(ax)
    add_binary_legend(fig, event_labels)
    add_figure_source_note(fig, source_note)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(cerra_binary, pred_binary, model, title, path, event_labels, land_mask=None, source_note=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmap, norm = binary_plot_settings()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False, subplot_kw=map_subplot_kwargs())
    for ax, da, label in [
        (axes[0, 0], cerra_binary, "CERRA"),
        (axes[0, 1], pred_binary, model),
    ]:
        ax.set_facecolor("white")
        im = ax.pcolormesh(
            da.longitude.values,
            da.latitude.values,
            np.ma.masked_invalid(da.values),
            shading="auto",
            cmap=cmap,
            norm=norm,
            **pcolormesh_kwargs(),
        )
        add_map_overlays(ax, land_mask)
        ax.set_title(label)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        add_coordinate_labels(ax)
        set_map_aspect(ax)
    fig.suptitle(title)
    add_binary_legend(fig, event_labels)
    add_figure_source_note(fig, source_note)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_all_model_comparison(cerra_binary, pred_binaries, title, path, event_labels, land_mask=None, source_note=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmap, norm = binary_plot_settings()
    names = list(pred_binaries)
    nplots = len(names) + 1
    ncols = 3
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, 5.35 * nrows),
        squeeze=False,
        subplot_kw=map_subplot_kwargs(),
    )
    axes = axes.reshape(-1)

    fields = [("CERRA", cerra_binary), *[(name, pred_binaries[name]) for name in names]]
    for ax, (label, da) in zip(axes, fields):
        ax.set_facecolor("white")
        im = ax.pcolormesh(
            da.longitude.values,
            da.latitude.values,
            np.ma.masked_invalid(da.values),
            shading="auto",
            cmap=cmap,
            norm=norm,
            **pcolormesh_kwargs(),
        )
        add_map_overlays(ax, land_mask)
        ax.set_title(label)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        add_coordinate_labels(ax)
        set_map_aspect(ax)
    unused_axes = axes[nplots:]
    if len(unused_axes):
        draw_binary_legend_axis(unused_axes[0], event_labels)
        for ax in unused_axes[1:]:
            ax.axis("off")
    else:
        add_binary_legend(fig, event_labels)

    fig.suptitle(title)
    add_figure_source_note(fig, source_note)
    bottom_margin = 0.035 if len(unused_axes) else 0.12
    fig.tight_layout(rect=(0, bottom_margin, 1, 0.94))
    fig.subplots_adjust(hspace=0.24, wspace=0.12)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_contingency_csv(contingency, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("Hit", "H", contingency["H"]),
        ("Miss", "M", contingency["M"]),
        ("False Alarm", "F", contingency["F"]),
        ("Correct Negative", "CN", contingency["CN"]),
        ("Valid Pixels", "valid_pixels", contingency["valid_pixels"]),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "symbol", "count"])
        writer.writerows(rows)


def save_contingency_png(contingency, path, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_text = [
        [contingency["H"], contingency["F"]],
        [contingency["M"], contingency["CN"]],
    ]
    row_labels = ["Prediction = 1", "Prediction = 0"]
    col_labels = ["CERRA = 1", "CERRA = 0"]

    fig, ax = plt.subplots(figsize=(8.8, 4.8), facecolor="white")
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.05)

    label_map = {
        (1, 0): "Hit\n(H)",
        (1, 1): "Miss\n(M)",
        (2, 0): "False Alarm\n(F)",
        (2, 1): "Correct Negative\n(CN)",
    }
    for key, label in label_map.items():
        cell = table[key]
        value = int(float(cell.get_text().get_text()))
        cell.get_text().set_text(f"{label}\n{value:,}".replace(",", " "))

    colors = {
        (1, 0): "#d8f3dc",
        (1, 1): "#fff3bf",
        (2, 0): "#ffe5d9",
        (2, 1): "#e8f1fb",
    }
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        cell.set_linewidth(1.0)
        if row == 0 or col == -1:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(weight="bold", color="white")
        else:
            cell.set_facecolor(colors.get((row, col), "#f8f8f8"))
    ax.set_title(f"{title}\nPixels valides: {contingency['valid_pixels']:,}".replace(",", " "), pad=18, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_scores_csv(scores, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["score", "value"])
        for key, value in scores.items():
            writer.writerow([key, value])


def save_summary_scores_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "H", "M", "F", "CN", "valid_pixels", "POD", "FAR", "CSI", "Accuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_summary_contingency_png(rows, path, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "Modele",
        "Hit\n(H)",
        "Miss\n(M)",
        "False Alarm\n(F)",
        "Correct Negative\n(CN)",
        "Pixels valides",
    ]
    cell_text = [
        [
            row["model"],
            f"{int(row['H']):,}".replace(",", " "),
            f"{int(row['M']):,}".replace(",", " "),
            f"{int(row['F']):,}".replace(",", " "),
            f"{int(row['CN']):,}".replace(",", " "),
            f"{int(row['valid_pixels']):,}".replace(",", " "),
        ]
        for row in rows
    ]
    fig_height = max(4.2, 1.7 + 0.72 * len(cell_text))
    fig, ax = plt.subplots(figsize=(13.5, fig_height), facecolor="white")
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.75)
    column_colors = {
        1: "#d8f3dc",
        2: "#fff3bf",
        3: "#ffe5d9",
        4: "#e8f1fb",
        5: "#f1f5f9",
    }
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        cell.set_linewidth(0.9)
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(weight="bold", color="white")
        elif col == 0:
            cell.set_facecolor("#f8fafc")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor(column_colors.get(col, "#f8f8f8"))
    ax.text(
        0.5,
        1.06,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.5,
        1.005,
        "Hit: evenement observe et predit | Miss: evenement observe non predit | "
        "False Alarm: evenement predit non observe | Correct Negative: absence observee et predite",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def output_directory(config, base_dir, model, time_text, type_event):
    output_cfg = require_section(config, "output")
    output_dir = output_cfg.get("output_dir", "outputs/extreme_events")
    return resolve_path(output_dir, base_dir) / model / f"{time_text}_{type_event}"


def process_model(
    args,
    config,
    config_dir,
    variable,
    domain,
    cerra,
    era5_interp,
    land_mask,
    event_labels,
    threshold,
    model,
    time_text,
):
    if args.use_saved_predictions or args.prediction_file is not None:
        pred_path = prediction_path_from_model(config, config_dir, model, variable, args.prediction_file)
        prediction = select_time(load_netcdf_data(pred_path, variable, "prediction"), args.date, args.time_index)
        prediction = crop_domain(prediction, domain)
    else:
        print(f"{model}: calcul direct de la prediction depuis ERA5 et le checkpoint.")
        prediction = direct_prediction_da(model, era5_interp, cerra, args, variable)
    ensure_compatible(cerra, prediction)

    cerra_masked = apply_land_mask(cerra, land_mask)
    prediction_masked = apply_land_mask(prediction, land_mask)
    cerra_for_threshold = threshold_input(cerra_masked, variable)
    prediction_for_threshold = threshold_input(prediction_masked, variable)

    cerra_binary = apply_threshold(cerra_for_threshold, args.type_event, threshold, land_mask=land_mask)
    pred_binary = apply_threshold(prediction_for_threshold, args.type_event, threshold, land_mask=land_mask)
    contingency = compute_contingency(cerra_binary, pred_binary)
    scores = compute_scores(contingency)
    source_note = build_source_note(
        config,
        config_dir,
        variable,
        model=model,
        prediction_file=args.prediction_file,
        use_saved_predictions=args.use_saved_predictions,
    )

    out_dir = output_directory(config, config_dir, model, time_text, args.type_event)
    stem = f"{time_text}_{args.type_event}"
    plot_binary_map(
        cerra_binary,
        f"CERRA binaire - {event_labels['event']} - {time_text}",
        out_dir / f"cerra_binary_{stem}.png",
        event_labels,
        land_mask=land_mask,
        source_note=source_note,
    )
    plot_binary_map(
        pred_binary,
        f"{model} binaire - {event_labels['event']} - {time_text}",
        out_dir / f"prediction_binary_{stem}_{model}.png",
        event_labels,
        land_mask=land_mask,
        source_note=source_note,
    )
    plot_comparison(
        cerra_binary,
        pred_binary,
        model,
        f"CERRA vs {model} - {event_labels['event']} - {time_text}",
        out_dir / f"comparison_{stem}_{model}.png",
        event_labels,
        land_mask=land_mask,
        source_note=source_note,
    )
    save_contingency_csv(contingency, out_dir / f"contingency_{stem}_{model}.csv")
    save_contingency_png(
        contingency,
        out_dir / f"contingency_{stem}_{model}.png",
        f"Table de contingence - {model} - {event_labels['event']} - {time_text}",
    )
    save_scores_csv(scores, out_dir / f"scores_{stem}_{model}.csv")

    print(f"Resultats sauvegardes dans: {out_dir}")
    print(f"{model} - Contingence: {contingency}")
    print(f"{model} - Scores: {scores}")
    return cerra_binary, pred_binary, contingency, scores


def main():
    args = parse_args()
    config, config_dir = load_config(args.config)
    data_cfg = require_section(config, "data")
    domain = validate_domain(require_section(config, "domain"))

    variable = data_cfg.get("variable")
    if not variable:
        raise KeyError("Champ 'data.variable' manquant dans config.yml.")
    cerra_file = data_cfg.get("cerra_file")
    if not cerra_file:
        raise KeyError("Champ 'data.cerra_file' manquant dans config.yml.")
    era5_file = data_cfg.get("era5_file")
    if not era5_file:
        raise KeyError("Champ 'data.era5_file' manquant dans config.yml.")

    cerra_path = resolve_path(cerra_file, config_dir)
    era5_path = resolve_path(era5_file, config_dir)
    threshold = get_threshold(config, args.type_event)
    event_labels = get_event_labels(config, args.type_event)
    models = selected_models(args.model, args.prediction_file)

    cerra = select_time(load_netcdf_data(cerra_path, variable, "CERRA"), args.date, args.time_index)
    selected_time_label = time_label(cerra, fallback_date=args.date, fallback_index=args.time_index)
    cerra = crop_domain(cerra, domain)
    era5 = select_time(load_netcdf_data(era5_path, variable, "ERA5"), args.date, args.time_index)
    era5 = crop_domain(era5, domain)
    era5_interp = era5.interp(latitude=cerra.latitude, longitude=cerra.longitude, method="linear")
    era5_interp = era5_interp.fillna(float(era5_interp.mean(skipna=True))).astype(np.float32)

    land_mask = load_land_mask(
        config,
        config_dir,
        cerra,
        domain,
        file_override=args.land_mask_file,
        var_override=args.land_mask_variable,
    )

    all_pred_binaries = {}
    summary_rows = []
    cerra_binary_for_summary = None
    for model in models:
        cerra_binary, pred_binary, contingency, scores = process_model(
            args,
            config,
            config_dir,
            variable,
            domain,
            cerra,
            era5_interp,
            land_mask,
            event_labels,
            threshold,
            model,
            selected_time_label,
        )
        cerra_binary_for_summary = cerra_binary
        all_pred_binaries[model] = pred_binary
        summary_rows.append({"model": model, **contingency, **scores})

    if args.model is None:
        out_dir = output_directory(config, config_dir, "AllModels", selected_time_label, args.type_event)
        stem = f"{selected_time_label}_{args.type_event}"
        source_note = build_source_note(
            config,
            config_dir,
            variable,
            model=None,
            prediction_file=args.prediction_file,
            use_saved_predictions=args.use_saved_predictions,
        )
        plot_all_model_comparison(
            cerra_binary_for_summary,
            all_pred_binaries,
            f"CERRA vs modeles - {event_labels['event']} - {selected_time_label}",
            out_dir / f"comparison_all_models_{stem}.png",
            event_labels,
            land_mask=land_mask,
            source_note=source_note,
        )
        save_summary_scores_csv(summary_rows, out_dir / f"scores_all_models_{stem}.csv")
        save_summary_contingency_png(
            summary_rows,
            out_dir / f"contingency_all_models_{stem}.png",
            f"Table de contingence - tous les modeles - {event_labels['event']} - {selected_time_label}",
        )
        print(f"Resume tous modeles sauvegarde dans: {out_dir}")


if __name__ == "__main__":
    main()
