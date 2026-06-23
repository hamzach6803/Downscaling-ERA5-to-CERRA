import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from utils import CERRA_DATA_PATH, DEFAULT_ERA5, DEFAULT_VARIABLE


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"


def spatial_dims(da):
    dims = [dim for dim in ("latitude", "longitude", "lat", "lon") if dim in da.dims]
    if len(dims) < 2:
        raise ValueError(f"Dimensions spatiales introuvables dans {da.name}: dims={da.dims}")
    return dims


def rank_extremes(cerra_da, n_hot, n_cold):
    dims = spatial_dims(cerra_da)
    score = cerra_da.mean(dim=dims, skipna=True)
    score_values = np.asarray(score.values, dtype=np.float64)
    valid = np.isfinite(score_values)
    if valid.sum() < n_hot + n_cold:
        raise ValueError(
            f"Pas assez de dates valides: {valid.sum()} disponibles, "
            f"{n_hot + n_cold} demandees."
        )

    valid_indices = np.where(valid)[0]
    ordered = valid_indices[np.argsort(score_values[valid])]
    cold_indices = ordered[:n_cold]
    hot_indices = ordered[-n_hot:][::-1]
    selected_indices = np.concatenate([cold_indices, hot_indices])

    labels = np.array(["cold"] * len(cold_indices) + ["hot"] * len(hot_indices), dtype=object)
    ranks = np.concatenate([np.arange(1, len(cold_indices) + 1), np.arange(1, len(hot_indices) + 1)])
    return selected_indices, labels, ranks, score


def save_metadata_csv(path, times, labels, ranks, scores):
    with open(path, "w", encoding="utf-8") as f:
        f.write("time,type,rank,spatial_mean\n")
        for time, label, rank, score in zip(times, labels, ranks, scores):
            f.write(f"{time},{label},{rank},{float(score)}\n")
    print(f"Metadata saved: {path}")


def select_extreme_cases(args):
    era5_ds = xr.open_dataset(args.era5, decode_timedelta=False).sortby("time")
    cerra_ds = xr.open_dataset(args.cerra, decode_timedelta=False).sortby("time")
    try:
        if args.variable not in era5_ds or args.variable not in cerra_ds:
            raise KeyError(
                f"Variable '{args.variable}' introuvable. "
                f"ERA5={list(era5_ds.data_vars)}, CERRA={list(cerra_ds.data_vars)}"
            )

        common_times = np.intersect1d(era5_ds.time.values, cerra_ds.time.values)
        if len(common_times) == 0:
            raise ValueError("Aucune date commune entre ERA5 et CERRA.")

        era5_common = era5_ds.sel(time=common_times)
        cerra_common = cerra_ds.sel(time=common_times)
        selected_indices, labels, ranks, score = rank_extremes(
            cerra_common[args.variable],
            args.n_hot,
            args.n_cold,
        )
        selected_times = common_times[selected_indices]
        selected_scores = np.asarray(score.values, dtype=np.float64)[selected_indices]

        era5_selected = era5_common.sel(time=selected_times).copy()
        cerra_selected = cerra_common.sel(time=selected_times).copy()

        attrs = {
            "source": "extreme temperature case selection",
            "variable": args.variable,
            "selection_score": "CERRA spatial mean over latitude/longitude",
            "n_hot": int(args.n_hot),
            "n_cold": int(args.n_cold),
            "case_order": "cold cases first, then hot cases",
        }
        era5_selected.attrs.update(attrs)
        cerra_selected.attrs.update(attrs)
        era5_selected = era5_selected.assign_coords(case_type=("time", labels), extreme_rank=("time", ranks))
        cerra_selected = cerra_selected.assign_coords(case_type=("time", labels), extreme_rank=("time", ranks))

        args.out_era5.parent.mkdir(parents=True, exist_ok=True)
        args.out_cerra.parent.mkdir(parents=True, exist_ok=True)
        era5_selected.to_netcdf(args.out_era5)
        cerra_selected.to_netcdf(args.out_cerra)
        print(f"ERA5 extremes saved : {args.out_era5}")
        print(f"CERRA extremes saved: {args.out_cerra}")

        if args.out_csv is not None:
            args.out_csv.parent.mkdir(parents=True, exist_ok=True)
            save_metadata_csv(args.out_csv, selected_times, labels, ranks, selected_scores)

        print(f"Cold cases: {args.n_cold}, hot cases: {args.n_hot}, total: {len(selected_times)}")
        print(f"Coldest mean: {float(np.nanmin(selected_scores[labels == 'cold'])):.4f}")
        print(f"Hottest mean: {float(np.nanmax(selected_scores[labels == 'hot'])):.4f}")
    finally:
        era5_ds.close()
        cerra_ds.close()


def main():
    parser = argparse.ArgumentParser(
        description="Selectionner 45 cas extremes chauds et 45 cas extremes froids depuis CERRA."
    )
    parser.add_argument("--era5", type=Path, default=DEFAULT_ERA5)
    parser.add_argument("--cerra", type=Path, default=CERRA_DATA_PATH)
    parser.add_argument("--variable", default=DEFAULT_VARIABLE)
    parser.add_argument("--n-hot", type=int, default=45)
    parser.add_argument("--n-cold", type=int, default=45)
    parser.add_argument("--out-era5", type=Path, default=DATA_DIR / "era5_2025_90times.nc")
    parser.add_argument("--out-cerra", type=Path, default=DATA_DIR / "cerra_2025_90times.nc")
    parser.add_argument("--out-csv", type=Path, default=DATA_DIR / "extreme_2025_90times_metadata.csv")
    args = parser.parse_args()
    select_extreme_cases(args)


if __name__ == "__main__":
    main()
