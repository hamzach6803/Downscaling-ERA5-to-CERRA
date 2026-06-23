import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from utils import (
    CERRA_DATA_PATH,
    DEFAULT_VARIABLE,
    evaluate,
    plot_cmap,
    plot_limits,
    plot_unit,
    values_for_plot,
    variable_tag,
)


ROOT = Path(__file__).resolve().parent
TAG = variable_tag(DEFAULT_VARIABLE)
METHODS = {
    "PyESD": ROOT / "PyESD" / "predictions" / f"pyesd_pred_{TAG}.nc",
    "DeepSD": ROOT / "DeepSD" / "predictions" / f"deepsd_pred_{TAG}.nc",
    "CAE": ROOT / "CAE" / "predictions" / f"cae_pred_{TAG}.nc",
    "ESRGAN": ROOT / "ESRGAN" / "predictions" / f"esrgan_pred_{TAG}.nc",
    "DDPM Cosine": ROOT / "DDPM" / "predictions" / f"ddpm_cosine_pred_{TAG}.nc",
    "CorrDiff": ROOT / "CorrDiff" / "predictions" / f"corrdiff_pred_{TAG}.nc",
}
PREDICT_COMMANDS = {
    "PyESD": "python PyESD/predict.py",
    "DeepSD": "python DeepSD/predict.py",
    "CAE": "python CAE/predict.py",
    "ESRGAN": "python ESRGAN/predict.py",
    "DDPM Cosine": "python DDPM/predict.py",
    "CorrDiff": "python CorrDiff/predict.py",
}
OUT_DIR = ROOT / "outputs" / "comparisons" / "main"
OUT_DIR.mkdir(exist_ok=True)
GRID_NOTES = {}

# Mets une date ici si tu veux comparer une date precise.
# Si None: une carte aleatoire commune est choisie parmi les temps disponibles.
COMPARE_DATE = None


def color_settings(variable, true_da, predictions):
    values = [values_for_plot(true_da.values, variable)]
    values.extend(values_for_plot(pred.values, variable) for pred in predictions.values())
    data = np.concatenate([np.asarray(v).reshape(-1) for v in values])
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return {"cmap": "viridis", "vmin": None, "vmax": None}
    vmin, vmax = plot_limits(data, variable)
    return {"cmap": plot_cmap(variable), "vmin": vmin, "vmax": vmax}


def error_vmax(true_frame, prediction_frames):
    errors = [np.abs(pred - true_frame).reshape(-1) for pred in prediction_frames]
    data = np.concatenate(errors)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return None
    return float(np.nanpercentile(data, 95))


def load_prediction(path):
    ds = xr.open_dataset(path, decode_timedelta=False)
    var_name = f"{DEFAULT_VARIABLE}_prediction"
    if var_name not in ds:
        var_name = list(ds.data_vars)[0]
    return ds[var_name]


def load_cerra_reference(path=CERRA_DATA_PATH):
    ds = xr.open_dataset(path, decode_timedelta=False).sortby("time")
    if DEFAULT_VARIABLE not in ds:
        raise KeyError(f"Variable '{DEFAULT_VARIABLE}' introuvable dans CERRA={list(ds.data_vars)}")
    cerra = ds[DEFAULT_VARIABLE].astype(np.float32)
    print(f"CERRA target  : {cerra.shape}")
    return cerra


def spatial_grid_issues(pred, target):
    issues = []
    spatial_dims = ("latitude", "longitude")
    for dim in spatial_dims:
        if dim not in pred.dims:
            issues.append(f"dimension '{dim}' absente")
            continue
        if dim not in pred.coords:
            issues.append(f"coordonnees '{dim}' absentes")
            continue
        if pred.sizes[dim] != target.sizes[dim]:
            issues.append(f"{dim}: {pred.sizes[dim]} au lieu de {target.sizes[dim]}")
            continue
        if not np.allclose(pred[dim].values, target[dim].values, rtol=0.0, atol=1e-6):
            pred_min = float(np.nanmin(pred[dim].values))
            pred_max = float(np.nanmax(pred[dim].values))
            target_min = float(np.nanmin(target[dim].values))
            target_max = float(np.nanmax(target[dim].values))
            issues.append(
                f"{dim}: domaine [{pred_min:g}, {pred_max:g}] au lieu de [{target_min:g}, {target_max:g}]"
            )
    return issues


def usable_predictions(true_da, predictions):
    usable = {}
    skipped = {}
    for name, pred in predictions.items():
        issues = spatial_grid_issues(pred, true_da)
        if "time" not in pred.dims:
            skipped[name] = ["dimension 'time' absente"]
            continue
        if "latitude" not in pred.coords or "longitude" not in pred.coords:
            skipped[name] = issues or ["coordonnees latitude/longitude absentes"]
            continue
        GRID_NOTES[name] = "complet" if not issues else "partiel: " + "; ".join(issues)
        usable[name] = pred
    partial = {name: note for name, note in GRID_NOTES.items() if note != "complet"}
    if partial:
        print("Predictions sur domaine partiel (incluses, mais metrics non strictement comparables au domaine complet):")
        for name, note in partial.items():
            print(f"  - {name}: {note}")
            print(f"    pour refaire en grille complete: {PREDICT_COMMANDS[name]}")
    if skipped:
        print("Predictions ignorees:")
        for name, issues in skipped.items():
            print(f"  - {name}: " + "; ".join(issues))
            print(f"    relance: {PREDICT_COMMANDS[name]}")
    return usable


def true_on_prediction_grid(true_da, pred):
    return (
        true_da.sel(time=pred.time.values)
        .interp(latitude=pred.latitude, longitude=pred.longitude)
        .transpose(*pred.dims)
    )


def compare_metrics(true_da, predictions):
    rows = []
    for name, pred in predictions.items():
        true_match = true_on_prediction_grid(true_da, pred)
        metrics = evaluate(pred.values, true_match.values, name)
        rows.append({"model": name, **metrics})
    return rows


def quality_label(r2):
    if not np.isfinite(r2):
        return "non defini"
    if r2 >= 0.90:
        return "excellent"
    if r2 >= 0.75:
        return "bon"
    if r2 >= 0.50:
        return "moyen"
    return "faible"


def add_quality_columns(rows):
    ranked = sorted(rows, key=lambda row: row["RMSE"])
    ranks = {row["model"]: rank for rank, row in enumerate(ranked, start=1)}
    for row in rows:
        row["Rank"] = ranks[row["model"]]
        row["Qualite"] = quality_label(row["R2"])
    return rows


def save_metrics_csv(rows, path):
    keys = ["Rank", "model", "RMSE", "MAE", "R2", "Bias", "Qualite", "Grille"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["Rank"]):
            writer.writerow(row)
    print(f"Metrics saved: {path}")


def plot_bar_metrics(rows, path):
    rows = sorted(rows, key=lambda item: item["Rank"])
    models = [f"{r['Rank']}. {r['model']}" for r in rows]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2", "Bias"]):
        values = [r[metric] for r in rows]
        ax.bar(models, values, color="#4777b3")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Comparaison des modeles - {DEFAULT_VARIABLE}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Metric plot saved: {path}")


def plot_quality_table(rows, path):
    rows = sorted(rows, key=lambda item: item["Rank"])
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rows) + 1.4))
    ax.axis("off")
    columns = ["Rang", "Modele", "RMSE", "MAE", "R2", "Bias", "Qualite", "Grille"]
    table_data = [
        [
            row["Rank"],
            row["model"],
            f"{row['RMSE']:.3f}",
            f"{row['MAE']:.3f}",
            f"{row['R2']:.3f}",
            f"{row['Bias']:.3f}",
            row["Qualite"],
            row["Grille"],
        ]
        for row in rows
    ]
    table = ax.table(cellText=table_data, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row_idx, _), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dfe8f3")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f5f7fa")
    ax.set_title(f"Qualite des modeles - {DEFAULT_VARIABLE}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Quality table saved: {path}")


def select_comparison_time(predictions):
    if COMPARE_DATE:
        return np.datetime64(COMPARE_DATE)

    common_times = None
    for pred in predictions.values():
        times = set(pred.time.values)
        common_times = times if common_times is None else common_times & times
    if common_times:
        chosen = np.random.choice(np.array(sorted(common_times)))
        print(f"Random common comparison map: {chosen}")
        return chosen

    first = next(iter(predictions.values()))
    chosen = first.time.values[0]
    print(f"No common prediction date. Spatial plot uses first available date: {chosen}")
    return chosen


def row_title(name, metrics_by_model):
    metrics = metrics_by_model.get(name, {})
    if not metrics:
        return name
    return (
        f"{metrics['Rank']}. {name}\n"
        f"RMSE={metrics['RMSE']:.3f}  MAE={metrics['MAE']:.3f}  "
        f"R2={metrics['R2']:.3f}  Bias={metrics['Bias']:.3f}  {metrics['Qualite']}\n"
        f"grille: {metrics['Grille']}"
    )


def plot_spatial_comparison(true_da, predictions, rows, path):
    selected_time = select_comparison_time(predictions)
    selected_predictions = {}
    selected_times = {}
    missing_common_date = []
    for name, pred in predictions.items():
        if selected_time in set(pred.time.values):
            model_time = selected_time
        else:
            model_time = pred.time.values[0]
            missing_common_date.append(name)
        selected_predictions[name] = pred.sel(time=[model_time])
        selected_times[name] = model_time
    if missing_common_date:
        print(
            "Modeles traces avec leur premiere date disponible: "
            + ", ".join(sorted(missing_common_date))
        )
    if not selected_predictions:
        raise ValueError("Aucune prediction disponible pour la date de comparaison spatiale.")

    true_full = true_da.sel(time=selected_time)
    true_frame = values_for_plot(true_full.values, DEFAULT_VARIABLE)
    lon = true_da.longitude.values
    lat = true_da.latitude.values
    colors = color_settings(DEFAULT_VARIABLE, true_da.sel(time=[selected_time]), selected_predictions)
    pred_frames = [values_for_plot(pred.isel(time=0).values, DEFAULT_VARIABLE) for pred in selected_predictions.values()]
    true_frames = [
        values_for_plot(true_on_prediction_grid(true_da, pred).isel(time=0).values, DEFAULT_VARIABLE)
        for pred in selected_predictions.values()
    ]
    errors = [np.abs(pred_frame - true_model_frame) for pred_frame, true_model_frame in zip(pred_frames, true_frames)]
    err_vmax = float(np.nanpercentile(np.concatenate([err.reshape(-1) for err in errors]), 95))
    signed_errors = [pred_frame - true_model_frame for pred_frame, true_model_frame in zip(pred_frames, true_frames)]
    signed_limit = np.nanpercentile(np.abs(np.concatenate([err.reshape(-1) for err in signed_errors])), 98)
    if not np.isfinite(signed_limit) or signed_limit <= 0:
        signed_limit = None
    unit = plot_unit(DEFAULT_VARIABLE)
    metrics_by_model = {row["model"]: row for row in rows}

    nrows = len(selected_predictions) + 1
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 3.2 * nrows), squeeze=False)

    im = axes[0, 0].pcolormesh(lon, lat, true_frame, shading="auto", **colors)
    axes[0, 0].set_title("Reference CERRA")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04).set_label(unit)
    axes[0, 1].axis("off")
    axes[0, 1].text(0.5, 0.5, f"Date comparee\n{selected_time}", ha="center", va="center", fontsize=12)
    axes[0, 2].axis("off")

    for row_idx, (name, pred) in enumerate(selected_predictions.items(), start=1):
        pred_frame = values_for_plot(pred.isel(time=0).values, DEFAULT_VARIABLE)
        pred_lon = pred.longitude.values
        pred_lat = pred.latitude.values
        true_model_frame = values_for_plot(true_on_prediction_grid(true_da, pred).isel(time=0).values, DEFAULT_VARIABLE)
        err = np.abs(pred_frame - true_model_frame)
        signed_err = pred_frame - true_model_frame
        im = axes[row_idx, 0].pcolormesh(pred_lon, pred_lat, pred_frame, shading="auto", **colors)
        axes[row_idx, 0].set_title(
            f"{row_title(name, metrics_by_model)}\ndate: {selected_times[name]}",
            fontsize=9,
        )
        fig.colorbar(im, ax=axes[row_idx, 0], fraction=0.046, pad=0.04).set_label(unit)
        im = axes[row_idx, 1].pcolormesh(pred_lon, pred_lat, err, shading="auto", cmap="Reds", vmin=0.0, vmax=err_vmax)
        axes[row_idx, 1].set_title("Erreur absolue")
        fig.colorbar(im, ax=axes[row_idx, 1], fraction=0.046, pad=0.04).set_label(unit)
        signed_kwargs = {"vmin": -signed_limit, "vmax": signed_limit} if signed_limit is not None else {}
        im = axes[row_idx, 2].pcolormesh(pred_lon, pred_lat, signed_err, shading="auto", cmap="coolwarm", **signed_kwargs)
        axes[row_idx, 2].set_title("Erreur signee (prediction - CERRA)")
        fig.colorbar(im, ax=axes[row_idx, 2], fraction=0.046, pad=0.04).set_label(unit)

    for ax in axes.reshape(-1):
        if ax.has_data():
            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
            ax.set_xlim(float(np.nanmin(lon)), float(np.nanmax(lon)))
            ax.set_ylim(float(np.nanmin(lat)), float(np.nanmax(lat)))

    fig.suptitle(f"Comparaison spatiale - {DEFAULT_VARIABLE} ({unit}) - {selected_time}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Spatial comparison saved: {path}")


def main():
    available_paths = {name: path for name, path in METHODS.items() if path.exists()}
    missing = {name: path for name, path in METHODS.items() if not path.exists()}
    if missing:
        print("Predictions manquantes:")
        for name, path in missing.items():
            print(f"  - {name}: {path}")
            print(f"    relance: {PREDICT_COMMANDS[name]}")
    if not available_paths:
        raise FileNotFoundError("Aucune prediction disponible. Lance d'abord les predict.py des modeles.")

    cerra = load_cerra_reference(CERRA_DATA_PATH)
    predictions = {}
    unreadable = []
    for name, path in available_paths.items():
        try:
            predictions[name] = load_prediction(path).sortby("time")
        except Exception as exc:
            print(f"Prediction illisible ignoree - {name}: {path} ({exc})")
            unreadable.append(name)
    if unreadable:
        print("Predictions a regenerer:")
        for name in unreadable:
            print(f"  - {PREDICT_COMMANDS[name]}")
    if not predictions:
        raise FileNotFoundError("Aucune prediction lisible. Relance les predict.py des modeles.")
    predictions = usable_predictions(cerra, predictions)
    if not predictions:
        raise ValueError("Aucune prediction utilisable. Relance les predict.py indiques.")
    rows = add_quality_columns(compare_metrics(cerra, predictions))
    for row in rows:
        row["Grille"] = GRID_NOTES.get(row["model"], "inconnue")
    save_metrics_csv(rows, OUT_DIR / f"metrics_{TAG}.csv")
    plot_bar_metrics(rows, OUT_DIR / f"metrics_{TAG}.png")
    plot_quality_table(rows, OUT_DIR / f"quality_table_{TAG}.png")
    plot_spatial_comparison(cerra, predictions, rows, OUT_DIR / f"spatial_comparison_{TAG}.png")


if __name__ == "__main__":
    main()
