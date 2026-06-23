import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyse_extreme import (
    DEFAULT_MODELS,
    MODEL_FILES,
    add_coordinate_labels,
    add_figure_source_note,
    apply_land_mask,
    build_source_note,
    crop_domain,
    ccrs,
    direct_prediction_da,
    load_config,
    load_land_mask,
    load_netcdf_data,
    map_subplot_kwargs,
    pcolormesh_kwargs,
    prediction_path_from_model,
    require_section,
    resolve_path,
    selected_models,
    select_time,
    set_map_aspect,
    time_label,
    validate_domain,
)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "figure.titlesize": 15,
    }
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tracer les cartes continues CERRA, ERA5, prediction et erreur absolue."
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
    parser.add_argument(
        "--model",
        default=None,
        choices=list(MODEL_FILES),
        help="Modele a tracer. Si absent, tous les modeles sauf DDPM sont traites.",
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
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
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


def temperature_to_celsius(da, variable):
    if str(variable).lower() not in {"t2m", "temperature", "temp"}:
        return da
    values = np.asarray(da.values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size and float(np.nanmedian(finite)) > 100.0:
        return da - 273.15
    return da


def plot_unit(variable):
    if str(variable).lower() in {"t2m", "temperature", "temp"}:
        return "degC"
    return str(variable)


def data_cmap(variable):
    if str(variable).lower() in {"t2m", "temperature", "temp"}:
        return "coolwarm"
    return "viridis"


def finite_limits(arrays, lower=2, upper=98):
    valid = []
    for arr in arrays:
        values = np.asarray(arr, dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size:
            valid.append(values)
    if not valid:
        return None, None
    flat = np.concatenate(valid)
    vmin = float(np.nanpercentile(flat, lower))
    vmax = float(np.nanpercentile(flat, upper))
    if vmin == vmax:
        pad = max(abs(vmin) * 0.05, 1.0)
        return vmin - pad, vmax + pad
    return vmin, vmax


def apply_domain_extent(ax, domain):
    extent = [domain["lon_min"], domain["lon_max"], domain["lat_min"], domain["lat_max"]]
    if ccrs is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_xlim(domain["lon_min"], domain["lon_max"])
        ax.set_ylim(domain["lat_min"], domain["lat_max"])


def add_map_overlays(ax, land_mask=None, domain=None):
    if domain is not None:
        apply_domain_extent(ax, domain)
    if land_mask is None:
        return
    from analyse_extreme import add_map_overlays as add_overlays

    add_overlays(ax, land_mask)


def draw_map(ax, da, title, cmap, vmin=None, vmax=None, land_mask=None, domain=None):
    ax.set_facecolor("#f8fafc")
    im = ax.pcolormesh(
        da.longitude.values,
        da.latitude.values,
        np.ma.masked_invalid(da.values),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **pcolormesh_kwargs(),
    )
    add_map_overlays(ax, land_mask, domain)
    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    add_coordinate_labels(ax)
    set_map_aspect(ax)
    return im


def save_model_maps(
    cerra,
    era5_raw,
    pred,
    model,
    variable,
    time_text,
    out_dir,
    land_mask,
    era5_land_mask,
    domain,
    source_note=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    unit = plot_unit(variable)
    cerra_plot = temperature_to_celsius(cerra, variable)
    era5_plot = temperature_to_celsius(era5_raw, variable)
    pred_plot = temperature_to_celsius(pred, variable)
    error = abs(pred_plot - cerra_plot)
    mean_mae = float(error.mean(skipna=True).item())

    vmin, vmax = finite_limits([cerra_plot.values, era5_plot.values, pred_plot.values])
    err_vmax = finite_limits([error.values], lower=0, upper=98)[1]
    fig, axes = plt.subplots(1, 4, figsize=(18.8, 5.1), subplot_kw=map_subplot_kwargs())
    fields = [
        (cerra_plot, "CERRA reference", data_cmap(variable), vmin, vmax, unit, land_mask),
        (era5_plot, "ERA5 brut 0.25 deg", data_cmap(variable), vmin, vmax, unit, era5_land_mask),
        (pred_plot, f"Prediction {model}", data_cmap(variable), vmin, vmax, unit, land_mask),
        (error, f"Erreur absolue\nMAE = {mean_mae:.3f} {unit}", "Reds", 0.0, err_vmax, unit, land_mask),
    ]
    for ax, (da, title, cmap, current_vmin, current_vmax, label, overlay_mask) in zip(axes, fields):
        im = draw_map(ax, da, title, cmap, current_vmin, current_vmax, overlay_mask, domain)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label)
    fig.suptitle(f"Cartes continues - {model} - {time_text}", fontweight="bold", y=1.02)
    add_figure_source_note(fig, source_note)
    fig.tight_layout(pad=1.1, rect=(0, 0.035, 1, 1))
    path = out_dir / f"maps_{time_text}_{model}_{variable}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot sauvegarde: {path}")
    return error, mean_mae


def save_all_predictions(
    cerra,
    era5_raw,
    predictions,
    variable,
    time_text,
    out_dir,
    land_mask,
    era5_land_mask,
    domain,
    source_note=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    unit = plot_unit(variable)
    cerra_plot = temperature_to_celsius(cerra, variable)
    era5_plot = temperature_to_celsius(era5_raw, variable)
    pred_plots = {name: temperature_to_celsius(pred, variable) for name, pred in predictions.items()}
    vmin, vmax = finite_limits([cerra_plot.values, era5_plot.values, *[da.values for da in pred_plots.values()]])

    fields = [("CERRA reference", cerra_plot, land_mask), ("ERA5 brut 0.25 deg", era5_plot, era5_land_mask)]
    fields.extend((f"Prediction {name}", da, land_mask) for name, da in pred_plots.items())
    ncols = 3
    nrows = int(np.ceil(len(fields) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 5.0 * nrows), subplot_kw=map_subplot_kwargs())
    axes = np.asarray(axes).reshape(-1)
    for ax, (title, da, overlay_mask) in zip(axes, fields):
        im = draw_map(ax, da, title, data_cmap(variable), vmin, vmax, overlay_mask, domain)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(unit)
    for ax in axes[len(fields):]:
        ax.axis("off")
    fig.suptitle(f"CERRA, ERA5 brut et predictions - tous les modeles - {time_text}", fontweight="bold", y=1.01)
    add_figure_source_note(fig, source_note)
    fig.tight_layout(pad=1.1, rect=(0, 0.035, 1, 1))
    path = out_dir / f"all_models_maps_{time_text}_{variable}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot sauvegarde: {path}")


def save_all_mae_maps(errors, variable, time_text, out_dir, land_mask, domain, source_note=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    unit = plot_unit(variable)
    err_vmax = finite_limits([err.values for err in errors.values()], lower=0, upper=98)[1]
    names = list(errors)
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 5.0 * nrows), subplot_kw=map_subplot_kwargs())
    axes = np.asarray(axes).reshape(-1)
    for ax, name in zip(axes, names):
        err = errors[name]
        mae = float(err.mean(skipna=True).item())
        im = draw_map(ax, err, f"{name}\nMAE = {mae:.3f} {unit}", "Reds", 0.0, err_vmax, land_mask, domain)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(unit)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle(f"Cartes d'erreur absolue - tous les modeles - {time_text}", fontweight="bold", y=1.01)
    add_figure_source_note(fig, source_note)
    fig.tight_layout(pad=1.1, rect=(0, 0.035, 1, 1))
    path = out_dir / f"all_models_mae_{time_text}_{variable}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot sauvegarde: {path}")


def output_base(config, config_dir):
    output_cfg = require_section(config, "output")
    output_dir = output_cfg.get("output_dir", "outputs")
    return resolve_path(output_dir, config_dir) / "continuous_maps"


def main():
    args = parse_args()
    config, config_dir = load_config(args.config)
    data_cfg = require_section(config, "data")
    domain = validate_domain(require_section(config, "domain"))
    variable = data_cfg.get("variable")
    if not variable:
        raise KeyError("Champ 'data.variable' manquant dans config.yml.")
    if "cerra_file" not in data_cfg:
        raise KeyError("Champ 'data.cerra_file' manquant dans config.yml.")
    if "era5_file" not in data_cfg:
        raise KeyError("Champ 'data.era5_file' manquant dans config.yml.")

    models = selected_models(args.model, args.prediction_file)
    cerra = select_time(
        load_netcdf_data(resolve_path(data_cfg["cerra_file"], config_dir), variable, "CERRA"),
        args.date,
        args.time_index,
    )
    time_text = time_label(cerra, fallback_date=args.date, fallback_index=args.time_index)
    cerra = crop_domain(cerra, domain)

    era5_raw = select_time(
        load_netcdf_data(resolve_path(data_cfg["era5_file"], config_dir), variable, "ERA5"),
        args.date,
        args.time_index,
    )
    era5_raw = crop_domain(era5_raw, domain)
    era5_interp = era5_raw.interp(latitude=cerra.latitude, longitude=cerra.longitude, method="linear")

    land_mask = load_land_mask(
        config,
        config_dir,
        cerra,
        domain,
        file_override=args.land_mask_file,
        var_override=args.land_mask_variable,
    )
    era5_land_mask = load_land_mask(
        config,
        config_dir,
        era5_raw,
        domain,
        file_override=args.land_mask_file,
        var_override=args.land_mask_variable,
    )
    cerra = apply_land_mask(cerra, land_mask)
    era5_raw = apply_land_mask(era5_raw, era5_land_mask)
    era5_interp = era5_interp.fillna(float(era5_interp.mean(skipna=True))).astype(np.float32)

    predictions = {}
    errors = {}
    base_out = output_base(config, config_dir)
    for model in models:
        if args.use_saved_predictions or args.prediction_file is not None:
            pred_path = prediction_path_from_model(config, config_dir, model, variable, args.prediction_file)
            pred = select_time(load_netcdf_data(pred_path, variable, "prediction"), args.date, args.time_index)
            pred = crop_domain(pred, domain)
        else:
            print(f"{model}: calcul direct de la prediction depuis ERA5 et le checkpoint.")
            pred = direct_prediction_da(model, era5_interp, cerra, args, variable)
        pred = apply_land_mask(pred, land_mask)
        predictions[model] = pred
        model_out = base_out / model / time_text
        source_note = build_source_note(
            config,
            config_dir,
            variable,
            model=model,
            prediction_file=args.prediction_file,
            use_saved_predictions=args.use_saved_predictions,
        )
        error, _ = save_model_maps(
            cerra,
            era5_raw,
            pred,
            model,
            variable,
            time_text,
            model_out,
            land_mask,
            era5_land_mask,
            domain,
            source_note=source_note,
        )
        errors[model] = error

    if args.model is None:
        all_out = base_out / "AllModels" / time_text
        source_note = build_source_note(
            config,
            config_dir,
            variable,
            model=None,
            prediction_file=args.prediction_file,
            use_saved_predictions=args.use_saved_predictions,
        )
        save_all_predictions(
            cerra,
            era5_raw,
            predictions,
            variable,
            time_text,
            all_out,
            land_mask,
            era5_land_mask,
            domain,
            source_note=source_note,
        )
        save_all_mae_maps(errors, variable, time_text, all_out, land_mask, domain, source_note=source_note)


if __name__ == "__main__":
    main()
