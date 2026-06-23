
# Script pour predire et tracer les modeles pour une seule date du dataset original.
# python results_single_date.py --date 2025-08-18T12:00
# python results_single_date.py --seed 42 | 42 fix random date selection
# python results_single_date.py --time-index 1000
# python results_single_date.py --date 2025-08-18T12:00 --palette-mode uniforme
# python results_single_date.py --date 2025-08-18T12:00 --palette-mode adaptive
# python results_single_date.py --date 2025-08-18T12:00 --palette-mode ch
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

import results_90times
from results_90times import (
    MODELS,
    compute_prediction,
    plot_all_models,
    plot_model_pair,
    rmse,
)
from utils import CERRA_DATA_PATH, DEFAULT_ERA5, DEFAULT_VARIABLE, evaluate, variable_tag
from utils import plot_unit


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "comparisons" / "main" / "results_single_date_Adaptive"
PRED_DIR = OUT_DIR / "predictions"


def is_uniform_palette(mode):
    return str(mode).lower() in {"uniform", "uniforme", "unforme", "shared", "global"}


def normalize_palette_mode(mode):
    mode = str(mode).lower()
    if mode == "ch":
        return "ch"
    return "uniform" if is_uniform_palette(mode) else "adaptive"


def choose_time(common_times, date=None, time_index=None, seed=None):
    if date is not None:
        target = np.datetime64(date)
        idx = int(np.argmin(np.abs(common_times - target)))
        selected_time = common_times[idx]
        print(f"Date demandee: {date} -> date utilisee: {selected_time} (index {idx})")
        return selected_time, idx

    if time_index is not None:
        idx = max(0, min(int(time_index), len(common_times) - 1))
        selected_time = common_times[idx]
        print(f"Index demande: {time_index} -> date utilisee: {selected_time} (index {idx})")
        return selected_time, idx

    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(common_times)))
    selected_time = common_times[idx]
    print(f"Date aleatoire utilisee: {selected_time} (index {idx})")
    return selected_time, idx


def load_single_date_data(era5_path, cerra_path, variable, date=None, time_index=None, seed=None):
    era5_ds = xr.open_dataset(era5_path, decode_timedelta=False).sortby("time")
    cerra_ds = xr.open_dataset(cerra_path, decode_timedelta=False).sortby("time")
    try:
        if variable not in era5_ds or variable not in cerra_ds:
            raise KeyError(
                f"Variable '{variable}' introuvable. ERA5={list(era5_ds.data_vars)}, "
                f"CERRA={list(cerra_ds.data_vars)}"
            )

        common_times = np.intersect1d(era5_ds.time.values, cerra_ds.time.values)
        if len(common_times) == 0:
            raise ValueError("Aucune date commune entre ERA5 et CERRA.")

        selected_time, selected_index = choose_time(common_times, date, time_index, seed)
        era5 = era5_ds[variable].sel(time=[selected_time]).load()
        cerra = cerra_ds[variable].sel(time=[selected_time]).astype(np.float32).load()
        era5_interp = era5.interp(latitude=cerra.latitude, longitude=cerra.longitude, method="linear")
        era5_interp = era5_interp.fillna(float(era5_interp.mean())).astype(np.float32).load()
        cerra = cerra.fillna(float(cerra.mean())).astype(np.float32).load()
    finally:
        era5_ds.close()
        cerra_ds.close()

    print(f"ERA5 interp date : {era5_interp.shape}")
    print(f"CERRA target date: {cerra.shape}")
    return era5_interp, cerra, selected_time, selected_index


def save_prediction_nc(name, pred, template, variable, selected_time):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    tag = variable_tag(variable)
    date_tag = str(selected_time)[:10].replace("-", "")
    path = PRED_DIR / f"{name.lower().replace(' ', '_')}_pred_date_{date_tag}_{tag}.nc"
    da = xr.DataArray(
        pred,
        dims=template.dims,
        coords=template.coords,
        name=f"{variable}_prediction",
        attrs={
            "model": name,
            "variable": variable,
            "selected_time": str(selected_time),
            "bilateral_filter": str(name == "CorrDiff"),
        },
    )
    da.to_dataset().to_netcdf(path)
    print(f"Prediction saved: {path}")


def save_metrics(predictions, cerra, variable, selected_time):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = variable_tag(variable)
    date_tag = str(selected_time)[:10].replace("-", "")
    path = OUT_DIR / f"metrics_single_date_{date_tag}_{tag}.csv"
    with open(path, "w", encoding="utf-8") as f:
        f.write("model,RMSE,MAE,R2,Bias,date,bilateral_filter\n")
        for name, pred in predictions.items():
            metrics = evaluate(pred, cerra.values, name)
            f.write(
                f"{name},{metrics['RMSE']},{metrics['MAE']},{metrics['R2']},"
                f"{metrics['Bias']},{selected_time},{name == 'CorrDiff'}\n"
            )
    print(f"Metrics saved: {path}")


def plot_mae_maps(predictions, cerra, variable, selected_time):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = variable_tag(variable)
    date_tag = str(selected_time)[:10].replace("-", "")
    lon = cerra.longitude.values
    lat = cerra.latitude.values
    true_map = np.asarray(cerra.isel(time=0).values, dtype=np.float32)
    names = list(predictions)
    mae_maps = {
        name: np.abs(np.asarray(pred[0], dtype=np.float32) - true_map)
        for name, pred in predictions.items()
    }
    shared_vmax = None
    if results_90times.use_uniform_palette() or results_90times.use_ch_palette():
        shared_maps = [
            arr
            for name, arr in mae_maps.items()
            if not (results_90times.use_ch_palette() and results_90times.has_ch_palette(name))
        ]
        valid = [arr[np.isfinite(arr)].reshape(-1) for arr in shared_maps if np.isfinite(arr).any()]
        flat = np.concatenate(valid) if valid else np.array([], dtype=np.float32)
        shared_vmax = float(np.nanmax(flat)) if flat.size else None

    ncols = 3
    nplots = len(names)
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)
    axes = axes.reshape(-1)

    for ax, name in zip(axes, names):
        mae_map = mae_maps[name]
        mae_global = float(np.nanmean(mae_map))
        finite = mae_map[np.isfinite(mae_map)]
        local_vmax = float(np.nanmax(finite)) if finite.size else None
        if results_90times.use_uniform_palette():
            vmax = shared_vmax
        elif results_90times.use_ch_palette() and not results_90times.has_ch_palette(name):
            vmax = shared_vmax
        else:
            vmax = local_vmax
        im = ax.pcolormesh(lon, lat, mae_map, shading="auto", cmap="Reds", vmin=0.0, vmax=vmax)
        ax.set_title(f"{name}\nMAE carte={mae_global:.3f}")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_box_aspect(1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(plot_unit(variable))

    for ax in axes[nplots:]:
        ax.axis("off")
    fig.suptitle(f"Cartes MAE - {selected_time} - {variable}")
    fig.tight_layout()
    path = OUT_DIR / f"all_models_mae_maps_single_date_{date_tag}_{tag}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"MAE maps saved: {path}")


def make_plots(predictions, cerra, variable, selected_time):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = variable_tag(variable)
    date_tag = str(selected_time)[:10].replace("-", "")
    lon = cerra.longitude.values
    lat = cerra.latitude.values
    true_map = cerra.isel(time=0).values
    pred_maps = {name: pred[0] for name, pred in predictions.items()}

    plot_all_models(
        pred_maps,
        true_map,
        lon,
        lat,
        variable,
        f"Comparaison des modeles - {selected_time} - {variable}",
        OUT_DIR / f"all_models_single_date_{date_tag}_{tag}.png",
    )
    for name, pred_map in pred_maps.items():
        plot_model_pair(
            name,
            pred_map,
            true_map,
            lon,
            lat,
            variable,
            f"{name} - date {selected_time}",
            OUT_DIR / f"{name.lower().replace(' ', '_')}_single_date_{date_tag}_{tag}.png",
        )
    plot_mae_maps(predictions, cerra, variable, selected_time)


def main():
    parser = argparse.ArgumentParser(
        description="Predire et tracer les modeles pour une seule date du dataset original."
    )
    parser.add_argument("--era5", type=Path, default=DEFAULT_ERA5)
    parser.add_argument("--cerra", type=Path, default=CERRA_DATA_PATH)
    parser.add_argument("--variable", default=DEFAULT_VARIABLE)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--date", default=None, help="Date a tracer, ex: 2025-08-18 ou 2025-08-18T12:00")
    parser.add_argument(
        "--time-index",
        "--time_index",
        dest="time_index",
        type=int,
        default=None,
        help="Index global dans le dataset original. Si absent, une date aleatoire est choisie.",
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
        choices=["uniform", "uniforme", "unforme", "shared", "global", "adaptive", "adaptative", "ch"],
        default="adaptive",
        help=(
            "uniforme: meme echelle couleur pour toutes les cartes; "
            "adaptive/adaptative: echelle propre a chaque carte; "
            "ch: DDPM et CorrDiff ont une palette adaptee, les autres modeles partagent la meme palette."
        ),
    )
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    results_90times.PALETTE_MODE = normalize_palette_mode(args.palette_mode)
    print(f"Palette mode: {args.palette_mode}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    era5_interp, cerra, selected_time, _ = load_single_date_data(
        args.era5,
        args.cerra,
        args.variable,
        args.date,
        args.time_index,
        args.seed,
    )

    predictions = {}
    failures = {}
    for name in args.models:
        print(f"\n=== {name} ===")
        try:
            pred = compute_prediction(name, MODELS[name], era5_interp, cerra, args, device)
            predictions[name] = pred
            print(f"{name}: RMSE={rmse(pred, cerra.values):.4f}")
            if args.save_predictions:
                save_prediction_nc(name, pred, cerra, args.variable, selected_time)
        except Exception as exc:
            failures[name] = str(exc)
            print(f"SKIP {name}: {exc}")

    if not predictions:
        raise RuntimeError("Aucun modele n'a pu etre predit.")
    if failures:
        print("\nModeles non traces:")
        for name, reason in failures.items():
            print(f"  - {name}: {reason}")

    make_plots(predictions, cerra, args.variable, selected_time)
    save_metrics(predictions, cerra, args.variable, selected_time)


if __name__ == "__main__":
    main()
