import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from matplotlib.ticker import FuncFormatter, MaxNLocator


MODEL_FILES = {
    "Interpolation": "interpolation_pred_90times_{variable}.nc",
    "PyESD": "pyesd_pred_90times_{variable}.nc",
    "DeepSD": "deepsd_pred_90times_{variable}.nc",
    "ESRGAN": "esrgan_pred_90times_{variable}.nc",
    "DDPM": "ddpm_pred_90times_{variable}.nc",
    "CorrDiff": "corrdiff_pred_90times_{variable}.nc",
}
DEFAULT_MODELS = ["Interpolation", "PyESD", "DeepSD", "ESRGAN"]
EVENTS = {
    "cold": {"case_type": "cold", "summary_suffix": "45cold", "label": "vague de froid"},
    "heat": {"case_type": "hot", "summary_suffix": "45hot", "label": "vague de chaleur"},
}
METRICS = ("POD", "FAR", "ACC", "F1")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculer POD, FAR, ACC et F1 carte par carte sur les 45 cas froids "
            "et/ou les 45 cas chauds, puis sauvegarder moyenne/ecart type/min/max."
        )
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yml"))
    parser.add_argument(
        "--type-event",
        "--type_event",
        dest="type_event",
        choices=["cold", "heat", "all"],
        default="all",
        help="Type de vague a analyser. Par defaut: cold et heat.",
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODEL_FILES))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--land-mask-file", type=Path, default=None)
    parser.add_argument("--land-mask-variable", default=None)
    return parser.parse_args()


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier config introuvable: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}, path.parent.resolve()


def require_section(config, name):
    section = config.get(name)
    if not isinstance(section, dict):
        raise KeyError(f"Section '{name}' manquante ou invalide dans config.yml.")
    return section


def resolve_path(path_value, base_dir):
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def variable_tag(variable):
    return str(variable).replace("/", "_").replace("\\", "_").replace(" ", "_")


def normalize_longitudes(da):
    lon = da["longitude"]
    if float(lon.max()) > 180.0:
        da = da.assign_coords(longitude=((lon + 180.0) % 360.0) - 180.0)
    return da.sortby("longitude")


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
    return normalize_longitudes(da).sortby("latitude")


def load_netcdf_data(path, variable, role):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{role}: fichier NetCDF introuvable: {path}")

    ds = xr.open_dataset(path, decode_timedelta=False)
    if "time" in ds.coords and role != "land_mask":
        ds = ds.sortby("time")
    try:
        var_name = variable
        if var_name not in ds.data_vars and str(role).startswith("prediction"):
            prediction_name = f"{variable}_prediction"
            if prediction_name in ds.data_vars:
                var_name = prediction_name
            elif len(ds.data_vars) == 1:
                var_name = next(iter(ds.data_vars))
        if var_name not in ds.data_vars:
            raise KeyError(
                f"{role}: variable '{variable}' introuvable dans {path}. "
                f"Variables disponibles: {list(ds.data_vars)}"
            )
        da = standardize_lat_lon_names(ds[var_name]).load()
    finally:
        ds.close()
    return da


def validate_domain(domain):
    required = ["lat_min", "lat_max", "lon_min", "lon_max"]
    missing = [key for key in required if key not in domain]
    if missing:
        raise KeyError(f"Domaine incomplet dans config.yml. Champs manquants: {missing}")
    out = {
        "lat_min": float(domain["lat_min"]),
        "lat_max": float(domain["lat_max"]),
        "lon_min": float(domain["lon_min"]),
        "lon_max": float(domain["lon_max"]),
    }
    if out["lat_min"] >= out["lat_max"] or out["lon_min"] >= out["lon_max"]:
        raise ValueError("Domaine mal defini: lat_min < lat_max et lon_min < lon_max sont requis.")
    return out


def crop_domain(da, domain):
    lat = da["latitude"]
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

    mask_path = resolve_path(mask_file, Path.cwd()) if file_override is not None else resolve_path(mask_file, base_dir)
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


def prediction_path(config, base_dir, model, variable):
    data_cfg = require_section(config, "data")
    predictions_dir = data_cfg.get("predictions_dir")
    if not predictions_dir:
        raise KeyError("Champ 'data.predictions_dir' manquant dans config.yml.")
    return resolve_path(predictions_dir, base_dir) / MODEL_FILES[model].format(variable=variable_tag(variable))


def display_path(path_value, base_dir):
    return resolve_path(path_value, base_dir).name


def build_source_note(config, base_dir, variable):
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
    if data_cfg.get("predictions_dir"):
        parts.append(f"Predictions: {display_path(data_cfg['predictions_dir'], base_dir)}")
    if data_cfg.get("metadata_file"):
        parts.append(f"Metadata: {display_path(data_cfg['metadata_file'], base_dir)}")
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


def format_longitude(value, _position=None):
    value = float(value)
    suffix = "E" if value >= 0 else "W"
    return f"{abs(value):.1f}{suffix}"


def format_latitude(value, _position=None):
    value = float(value)
    suffix = "N" if value >= 0 else "S"
    return f"{abs(value):.1f}{suffix}"


def add_coordinate_labels(ax):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(format_longitude))
    ax.yaxis.set_major_formatter(FuncFormatter(format_latitude))
    ax.grid(True, color="#94a3b8", linewidth=0.35, alpha=0.55, linestyle="--")
    ax.tick_params(axis="both", labelsize=8, colors="#334155")


def get_threshold(config, type_event):
    thresholds = require_section(config, "thresholds")
    key = "heat_threshold" if type_event == "heat" else "cold_threshold"
    if key not in thresholds:
        raise KeyError(f"Seuil manquant dans config.yml: thresholds.{key}")
    unit = str(thresholds.get("unit", "C")).strip().lower()
    if unit not in {"c", "degc", "celsius", "degree", "degrees"}:
        raise ValueError("Les seuils doivent etre exprimes en degres Celsius.")
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


def apply_threshold(da, type_event, threshold, land_mask):
    valid = np.isfinite(da)
    if type_event == "heat":
        binary = xr.where(valid, xr.where(da > threshold, 1.0, 0.0), np.nan)
    else:
        binary = xr.where(valid, xr.where(da < threshold, 1.0, 0.0), np.nan)
    return binary.where(land_mask).astype(np.float32)


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
        "ACC": safe_div(h + cn, total),
        "F1": safe_div(2 * h, 2 * h + m + f),
    }


def event_types_from_args(type_event):
    return ["cold", "heat"] if type_event == "all" else [type_event]


def datetime_key(value):
    return np.datetime_as_string(np.datetime64(value), unit="ns")


def load_metadata_case_types(path):
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[datetime_key(row["time"])] = {
                "case_type": str(row["type"]).strip().lower(),
                "rank": int(row["rank"]),
            }
    return out


def select_event_subset(cerra, type_event, metadata):
    expected_case = EVENTS[type_event]["case_type"]
    if "case_type" in cerra.coords:
        values = np.asarray(cerra["case_type"].values).astype(str)
        mask = np.char.lower(np.char.strip(values)) == expected_case
        if mask.any():
            return cerra.isel(time=mask)

    if metadata:
        mask = []
        ranks = []
        for time_value in cerra.time.values:
            item = metadata.get(datetime_key(time_value))
            is_match = item is not None and item["case_type"] == expected_case
            mask.append(is_match)
            ranks.append(item["rank"] if is_match else -1)
        mask = np.asarray(mask, dtype=bool)
        if mask.any():
            subset = cerra.isel(time=mask)
            return subset.assign_coords(extreme_rank=("time", np.asarray(ranks, dtype=np.int32)[mask]))

    raise ValueError(
        f"Impossible d'identifier les cartes '{expected_case}'. "
        "Le fichier CERRA doit contenir une coordonnee case_type ou un metadata_file valide."
    )


def align_prediction(prediction, cerra_subset):
    if "time" not in prediction.dims:
        raise ValueError("Les predictions doivent contenir une dimension time.")
    aligned = prediction.sel(time=cerra_subset.time.values)
    if tuple(aligned.sizes[dim] for dim in ("latitude", "longitude")) != tuple(
        cerra_subset.sizes[dim] for dim in ("latitude", "longitude")
    ):
        raise ValueError(f"Dimensions incompatibles: CERRA={cerra_subset.shape}, prediction={aligned.shape}")
    if not np.allclose(aligned.latitude.values, cerra_subset.latitude.values) or not np.allclose(
        aligned.longitude.values, cerra_subset.longitude.values
    ):
        raise ValueError("Grilles incompatibles: les coordonnees latitude/longitude different.")
    return aligned


def case_rank(cerra_subset, idx):
    if "extreme_rank" in cerra_subset.coords:
        return int(cerra_subset["extreme_rank"].values[idx])
    return idx + 1


def time_label(value):
    return str(np.datetime64(value, "h"))


def mean_binary_map(binary_maps, template):
    values = np.stack([np.asarray(da.values, dtype=np.float32) for da in binary_maps], axis=0)
    valid = np.isfinite(values)
    counts = valid.sum(axis=0)
    totals = np.where(valid, values, 0.0).sum(axis=0)
    mean_values = np.full(values.shape[1:], np.nan, dtype=np.float32)
    np.divide(totals, counts, out=mean_values, where=counts > 0)
    return xr.DataArray(
        mean_values,
        dims=("latitude", "longitude"),
        coords={"latitude": template.latitude, "longitude": template.longitude},
        name="event_frequency",
    )


def process_model_event(model, prediction, cerra_subset, type_event, variable, threshold, land_mask):
    prediction = align_prediction(prediction, cerra_subset)
    rows = []
    cerra_binaries = []
    pred_binaries = []
    for idx in range(cerra_subset.sizes["time"]):
        cerra_map = cerra_subset.isel(time=idx)
        pred_map = prediction.isel(time=idx)
        cerra_for_threshold = threshold_input(cerra_map.where(land_mask), variable)
        pred_for_threshold = threshold_input(pred_map.where(land_mask), variable)
        cerra_binary = apply_threshold(cerra_for_threshold, type_event, threshold, land_mask)
        pred_binary = apply_threshold(pred_for_threshold, type_event, threshold, land_mask)
        contingency = compute_contingency(cerra_binary, pred_binary)
        scores = compute_scores(contingency)
        cerra_binaries.append(cerra_binary)
        pred_binaries.append(pred_binary)
        rows.append(
            {
                "type_event": type_event,
                "case_type": EVENTS[type_event]["case_type"],
                "model": model,
                "time": time_label(cerra_subset.time.values[idx]),
                "case_rank": case_rank(cerra_subset, idx),
                **contingency,
                **scores,
            }
        )
    return rows, mean_binary_map(cerra_binaries, cerra_subset.isel(time=0)), mean_binary_map(pred_binaries, cerra_subset.isel(time=0))


def plot_event_frequency_maps(frequency_maps, type_event, threshold, path, source_note=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(frequency_maps)
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 4.8 * nrows), squeeze=False)
    axes = axes.reshape(-1)
    for ax, name in zip(axes, names):
        da = frequency_maps[name]
        im = ax.pcolormesh(
            da.longitude.values,
            da.latitude.values,
            np.ma.masked_invalid(da.values),
            shading="auto",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        add_coordinate_labels(ax)
        ax.set_box_aspect(1)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("frequence")
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle(
        f"Frequence des pixels detectes - {EVENTS[type_event]['label']} - seuil {threshold:g} C",
        fontweight="bold",
        y=1.01,
    )
    add_figure_source_note(fig, source_note)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Carte sauvegardee: {path}")


def summarize_rows(rows, threshold):
    summary = {
        "type_event": rows[0]["type_event"],
        "case_type": rows[0]["case_type"],
        "model": rows[0]["model"],
        "threshold_C": threshold,
        "n_maps": len(rows),
    }
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[f"{metric}_mean"] = float(np.nanmean(values)) if finite.size else float("nan")
        summary[f"{metric}_std"] = float(np.nanstd(values)) if finite.size else float("nan")
        summary[f"{metric}_min"] = float(np.nanmin(values)) if finite.size else float("nan")
        summary[f"{metric}_max"] = float(np.nanmax(values)) if finite.size else float("nan")
        summary[f"{metric}_valid"] = int(finite.size)
    return summary


def write_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV sauvegarde: {path}")


def output_dir_from_config(config, config_dir, override):
    if override is not None:
        return resolve_path(override, Path.cwd())
    output_cfg = require_section(config, "output")
    return resolve_path(output_cfg.get("output_dir", "outputs"), config_dir)


def main():
    args = parse_args()
    config, config_dir = load_config(args.config)
    data_cfg = require_section(config, "data")
    variable = data_cfg.get("variable")
    if not variable:
        raise KeyError("Champ 'data.variable' manquant dans config.yml.")

    domain = validate_domain(require_section(config, "domain"))
    cerra_path = resolve_path(data_cfg["cerra_file"], config_dir)
    metadata_path = resolve_path(data_cfg.get("metadata_file", ""), config_dir) if data_cfg.get("metadata_file") else None
    out_dir = output_dir_from_config(config, config_dir, args.output_dir)

    print(f"CERRA extremes: {cerra_path}")
    cerra = crop_domain(load_netcdf_data(cerra_path, variable, "CERRA"), domain)
    land_mask = load_land_mask(
        config,
        config_dir,
        cerra.isel(time=0) if "time" in cerra.dims else cerra,
        domain,
        file_override=args.land_mask_file,
        var_override=args.land_mask_variable,
    )
    metadata = load_metadata_case_types(metadata_path) if metadata_path is not None else {}

    summary_rows = []
    per_map_rows = []
    for type_event in event_types_from_args(args.type_event):
        threshold = get_threshold(config, type_event)
        cerra_subset = select_event_subset(cerra, type_event, metadata)
        print(
            f"\n=== {EVENTS[type_event]['label']} ({EVENTS[type_event]['case_type']}) "
            f"- {cerra_subset.sizes['time']} cartes - seuil {threshold:g} C ==="
        )
        event_summary_rows = []
        event_per_map_rows = []
        frequency_maps = {}
        source_note = build_source_note(config, config_dir, variable)
        for model in args.models:
            pred_path = prediction_path(config, config_dir, model, variable)
            prediction = crop_domain(load_netcdf_data(pred_path, variable, f"prediction {model}"), domain)
            rows, cerra_frequency, prediction_frequency = process_model_event(
                model,
                prediction,
                cerra_subset,
                type_event,
                variable,
                threshold,
                land_mask,
            )
            if "CERRA" not in frequency_maps:
                frequency_maps["CERRA"] = cerra_frequency
            frequency_maps[model] = prediction_frequency
            summary = summarize_rows(rows, threshold)
            summary_rows.append(summary)
            per_map_rows.extend(rows)
            event_summary_rows.append(summary)
            event_per_map_rows.extend(rows)
            print(
                f"{model}: POD={summary['POD_mean']:.3f}, "
                f"FAR={summary['FAR_mean']:.3f}, ACC={summary['ACC_mean']:.3f}, "
                f"F1={summary['F1_mean']:.3f}"
            )

        suffix = EVENTS[type_event]["summary_suffix"]
        tag = variable_tag(variable)
        plot_event_frequency_maps(
            frequency_maps,
            type_event,
            threshold,
            out_dir / type_event / f"event_frequency_maps_{suffix}_{tag}.png",
            source_note=source_note,
        )
        write_csv(
            event_summary_rows,
            out_dir / type_event / f"scores_summary_{suffix}_{tag}.csv",
            summary_fieldnames(),
        )
        write_csv(
            event_per_map_rows,
            out_dir / type_event / f"scores_per_map_{suffix}_{tag}.csv",
            per_map_fieldnames(),
        )

    tag = variable_tag(variable)
    write_csv(summary_rows, out_dir / f"scores_extreme_summary_{tag}.csv", summary_fieldnames())
    write_csv(per_map_rows, out_dir / f"scores_extreme_per_map_{tag}.csv", per_map_fieldnames())


def summary_fieldnames():
    fields = ["type_event", "case_type", "model", "threshold_C", "n_maps"]
    for metric in METRICS:
        fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_min",
                f"{metric}_max",
                f"{metric}_valid",
            ]
        )
    return fields


def per_map_fieldnames():
    return [
        "type_event",
        "case_type",
        "model",
        "time",
        "case_rank",
        "H",
        "M",
        "F",
        "CN",
        "valid_pixels",
        "POD",
        "FAR",
        "ACC",
        "F1",
    ]


if __name__ == "__main__":
    main()
