# python results_test20.py --use-saved-predictions --palette-mode ch
import argparse
from pathlib import Path

import numpy as np
import torch
import xarray as xr

import results_90times
from results_90times import (
    MODELS,
    compute_prediction,
    plot_all_models,
    plot_mae_maps,
    plot_model_pair,
    mae,
    rmse,
)
from utils import (
    CERRA_DATA_PATH,
    DEFAULT_ERA5,
    DEFAULT_VARIABLE,
    TRAIN_RATIO,
    evaluate,
    split_index,
    variable_tag,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "comparisons" / "test20" / "results_test20"
PRED_DIR = OUT_DIR / "predictions"


def is_uniform_palette(mode):
    return str(mode).lower() in {"uniform", "uniforme", "unforme", "shared", "global"}


def normalize_palette_mode(mode):
    mode = str(mode).lower()
    if mode in {"ch", "ddpm"}:
        return mode
    return "uniform" if is_uniform_palette(mode) else "adaptive"


def load_test20_data(era5_path, cerra_path, variable, train_ratio, max_test_samples=None):
    era5_ds = xr.open_dataset(era5_path, decode_timedelta=False).sortby("time")
    cerra_ds = xr.open_dataset(cerra_path, decode_timedelta=False).sortby("time")
    if variable not in era5_ds or variable not in cerra_ds:
        raise KeyError(
            f"Variable '{variable}' introuvable. ERA5={list(era5_ds.data_vars)}, "
            f"CERRA={list(cerra_ds.data_vars)}"
        )

    era5 = era5_ds[variable]
    cerra = cerra_ds[variable]
    common_times = np.intersect1d(era5.time.values, cerra.time.values)
    n_train = split_index(len(common_times), train_ratio)
    test_times = common_times[n_train:]
    if max_test_samples is not None:
        test_times = test_times[: int(max_test_samples)]
    if len(test_times) == 0:
        raise ValueError("Aucune donnee dans le jeu de test 20 %.")

    era5 = era5.sel(time=test_times).load()
    cerra = cerra.sel(time=test_times).astype(np.float32).load()
    era5_interp = era5.interp(latitude=cerra.latitude, longitude=cerra.longitude, method="linear")
    era5_interp = era5_interp.fillna(float(era5_interp.mean())).astype(np.float32).load()
    cerra = cerra.fillna(float(cerra.mean())).astype(np.float32).load()
    era5_ds.close()
    cerra_ds.close()
    print(f"ERA5 test 20 % interp : {era5_interp.shape}")
    print(f"CERRA test 20 % target: {cerra.shape}")
    return era5_interp, cerra


def save_prediction_nc(name, pred, template, variable):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    tag = variable_tag(variable)
    path = PRED_DIR / f"{name.lower().replace(' ', '_')}_pred_test20_{tag}.nc"
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
    path = PRED_DIR / f"{name.lower().replace(' ', '_')}_pred_test20_{tag}.nc"
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


def selected_time_index(cerra, date, time_index, seed=None):
    if date is not None:
        target = np.datetime64(date)
        idx = int(np.argmin(np.abs(cerra.time.values - target)))
        print(f"Date demandee: {date} -> date utilisee: {cerra.time.values[idx]} (index test {idx})")
        return idx
    if time_index is not None:
        idx = max(0, min(int(time_index), len(cerra.time) - 1))
        print(f"Index demande: {time_index} -> date utilisee: {cerra.time.values[idx]} (index test {idx})")
        return idx
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(cerra.time)))
    print(f"Date aleatoire utilisee: {cerra.time.values[idx]} (index test {idx})")
    return idx


def make_plots(predictions, cerra, args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = variable_tag(args.variable)
    lon = cerra.longitude.values
    lat = cerra.latitude.values

    true_mean = cerra.mean("time").values
    mean_pred_maps = {name: pred.mean(axis=0) for name, pred in predictions.items()}
    true_values = np.asarray(cerra.values, dtype=np.float32)
    scores = {name: mae(pred, true_values) for name, pred in predictions.items()}
    mae_maps = {
        name: np.nanmean(np.abs(np.asarray(pred, dtype=np.float32) - true_values), axis=0)
        for name, pred in predictions.items()
    }
    plot_all_models(
        mean_pred_maps,
        true_mean,
        lon,
        lat,
        args.variable,
        f"Moyenne du jeu de test 20 % - {args.variable}",
        OUT_DIR / f"all_models_mean_test20_{tag}.png",
    )
    for name, pred_map in mean_pred_maps.items():
        plot_model_pair(
            name,
            pred_map,
            true_mean,
            lon,
            lat,
            args.variable,
            f"{name} - moyenne du jeu de test 20 %",
            OUT_DIR / f"{name.lower().replace(' ', '_')}_mean_test20_{tag}.png",
            score=scores[name],
            mae_map=mae_maps[name],
        )

    plot_mae_maps(
        predictions,
        cerra,
        args.variable,
        f"Cartes MAE sur le jeu de test 20 % - {args.variable}",
        OUT_DIR / f"all_models_mae_maps_test20_{tag}.png",
        scores=scores,
    )

    idx = selected_time_index(cerra, args.date, args.time_index, args.seed)
    selected_time = str(cerra.time.values[idx])
    date_suffix = selected_time[:10].replace("-", "")
    true_date = cerra.isel(time=idx).values
    date_pred_maps = {name: pred[idx] for name, pred in predictions.items()}
    plot_all_models(
        date_pred_maps,
        true_date,
        lon,
        lat,
        args.variable,
        f"Carte test du {selected_time} - {args.variable}",
        OUT_DIR / f"all_models_date_test20_{date_suffix}_{tag}.png",
    )
    for name, pred_map in date_pred_maps.items():
        plot_model_pair(
            name,
            pred_map,
            true_date,
            lon,
            lat,
            args.variable,
            f"{name} - carte test du {selected_time}",
            OUT_DIR / f"{name.lower().replace(' ', '_')}_date_test20_{date_suffix}_{tag}.png",
        )


def save_metrics(predictions, cerra, variable):
    path = OUT_DIR / f"metrics_test20_{variable_tag(variable)}.csv"
    with open(path, "w", encoding="utf-8") as f:
        f.write("model,RMSE,MAE,R2,Bias,n_maps,bilateral_filter\n")
        for name, pred in predictions.items():
            metrics = evaluate(pred, cerra.values, name)
            f.write(
                f"{name},{metrics['RMSE']},{metrics['MAE']},{metrics['R2']},"
                f"{metrics['Bias']},{len(cerra.time)},{name == 'CorrDiff'}\n"
            )
    print(f"Metrics saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Predire et tracer les resultats sur le jeu de test 20 %.")
    parser.add_argument("--era5", type=Path, default=DEFAULT_ERA5)
    parser.add_argument("--cerra", type=Path, default=CERRA_DATA_PATH)
    parser.add_argument("--variable", default=DEFAULT_VARIABLE)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--date", default=None, help="Date a tracer, ex: 2025-01-03 ou 2025-01-03T12:00")
    parser.add_argument(
        "--time-index",
        "--time_index",
        dest="time_index",
        type=int,
        default=None,
        help="Index dans le jeu de test si --date n'est pas fourni. Si absent, une date aleatoire est choisie.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Graine optionnelle pour reproduire la date aleatoire.")
    parser.add_argument("--max-test-samples", type=int, default=None)
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
        default=results_90times.PALETTE_MODE,
        help=(
            "uniform: meme echelle couleur pour toutes les cartes; "
            "adaptive/adaptative: echelle propre a chaque carte; "
            "ch: DDPM et CorrDiff ont une palette adaptee, les autres modeles partagent la meme palette; "
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
    results_90times.PALETTE_MODE = normalize_palette_mode(args.palette_mode)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    era5_interp, cerra = load_test20_data(
        args.era5,
        args.cerra,
        args.variable,
        args.train_ratio,
        args.max_test_samples,
    )

    predictions = {}
    failures = {}
    for name in args.models:
        print(f"\n=== {name} ===")
        try:
            if args.use_saved_predictions:
                pred = load_prediction_nc(name, cerra, args.variable)
            else:
                pred = compute_prediction(name, MODELS[name], era5_interp, cerra, args, device)
            pred = results_90times.align_prediction_units(pred, cerra.values, args.variable, name)
            predictions[name] = pred
            print(f"{name}: RMSE={rmse(pred, cerra.values):.4f}")
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
