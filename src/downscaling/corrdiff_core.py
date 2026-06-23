import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from .utils import (
    ALLOW_MNT_COORD_FALLBACK,
    MNT_PATH,
    MNT_VARIABLE_CANDIDATES,
    add_common_args,
    add_predict_args,
    cleanup_memory,
    evaluate,
    load_and_preprocess,
    load_prediction_inputs,
    normalize_from_train,
    plot_prediction,
    save_prediction_nc,
    save_hyperparams,
    split_index,
    variable_tag,
)


def time_slice_array(data, start, end):
    if hasattr(data, "isel"):
        return np.asarray(data.isel({data.dims[0]: slice(start, end)}).values, dtype=np.float32)
    return np.asarray(data[start:end], dtype=np.float32)


def fit_linear_regression_baseline(x_train, y_train, chunk_size=64):
    n_train = len(x_train)
    if n_train <= 0:
        raise ValueError("Aucun echantillon d'entrainement pour la regression lineaire.")

    spatial_shape = x_train.shape[-2:]
    sum_x = np.zeros(spatial_shape, dtype=np.float64)
    sum_y = np.zeros(spatial_shape, dtype=np.float64)
    sum_xy = np.zeros(spatial_shape, dtype=np.float64)
    sum_x2 = np.zeros(spatial_shape, dtype=np.float64)

    for start in range(0, n_train, chunk_size):
        end = min(start + chunk_size, n_train)
        x_chunk = time_slice_array(x_train, start, end)
        y_chunk = time_slice_array(y_train, start, end)
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


def linear_baseline_predict(x, slope, intercept):
    return (x * slope + intercept).astype(np.float32)


class CorrDiffPatchDataset(Dataset):
    def __init__(self, x_hr, y_hr, slope, intercept, mnt=None, patch_size=64):
        self.x_hr = x_hr
        self.y_hr = y_hr
        self.slope = slope.astype(np.float32, copy=False)
        self.intercept = intercept.astype(np.float32, copy=False)
        self.mnt = None if mnt is None else mnt.astype(np.float32, copy=False)
        self.patch_size = min(patch_size, x_hr.shape[-2], x_hr.shape[-1])

    def __len__(self):
        return len(self.x_hr)

    def _patch(self, data, idx, i, j, p):
        if hasattr(data, "isel"):
            indexers = {
                data.dims[0]: idx,
                data.dims[-2]: slice(i, i + p),
                data.dims[-1]: slice(j, j + p),
            }
            return np.asarray(data.isel(indexers).values, dtype=np.float32)
        return np.asarray(data[idx, i:i + p, j:j + p], dtype=np.float32)

    def __getitem__(self, idx):
        h, w = self.x_hr.shape[-2:]
        p = self.patch_size
        i = torch.randint(0, h - p + 1, (1,)).item()
        j = torch.randint(0, w - p + 1, (1,)).item()
        x_patch = self._patch(self.x_hr, idx, i, j, p)
        y_patch = self._patch(self.y_hr, idx, i, j, p)
        baseline_patch = x_patch * self.slope[i:i + p, j:j + p] + self.intercept[i:i + p, j:j + p]
        residual_patch = y_patch - baseline_patch
        tensors = [
            torch.from_numpy(x_patch[None]),
            torch.from_numpy(baseline_patch[None]),
            torch.from_numpy(residual_patch[None]),
        ]
        if self.mnt is not None:
            tensors.append(torch.from_numpy(self.mnt[i:i + p, j:j + p][None]))
        return tuple(tensors)


class ScoreBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class ScoreUNet(nn.Module):
    def __init__(self, in_channels=4, channels=32):
        super().__init__()
        self.enc1 = ScoreBlock(in_channels, channels)
        self.enc2 = ScoreBlock(channels, channels * 2)
        self.mid = ScoreBlock(channels * 2, channels * 2)
        self.dec1 = ScoreBlock(channels * 3, channels)
        self.out = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x_hr, baseline, noisy_residual, sigma, mnt=None):
        sigma_map = torch.ones_like(noisy_residual) * torch.log(sigma).view(-1, 1, 1, 1)
        parts = [x_hr, baseline, noisy_residual, sigma_map]
        if mnt is not None:
            parts.append(mnt)
        x = torch.cat(parts, dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        mid = self.mid(e2)
        up = F.interpolate(mid, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        dec = self.dec1(torch.cat([up, e1], dim=1))
        return self.out(dec)


def coordinate_mnt_fallback(cerra_da):
    lat = cerra_da.latitude.values.astype(np.float32)
    lon = cerra_da.longitude.values.astype(np.float32)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    lat_norm = (lat2d - float(lat2d.mean())) / max(float(lat2d.std()), 1e-8)
    lon_norm = (lon2d - float(lon2d.mean())) / max(float(lon2d.std()), 1e-8)
    proxy = 0.5 * lat_norm + 0.5 * lon_norm
    print(
        "WARNING: MNT fallback actif. Aucune MNT lisible n'a ete chargee; "
        "utilisation d'un proxy latitude/longitude normalise. "
        "Pour une vraie MNT, remplace utils.MNT_PATH par un NetCDF lisible."
    )
    return proxy.astype(np.float32)


def open_mnt_dataset(path):
    errors = []
    for engine in [None, "netcdf4", "h5netcdf", "scipy"]:
        try:
            kwargs = {"decode_timedelta": False}
            if engine is not None:
                kwargs["engine"] = engine
            return xr.open_dataset(path, **kwargs)
        except Exception as exc:
            label = "auto" if engine is None else engine
            errors.append(f"{label}: {exc}")
    raise ValueError(
        f"Impossible d'ouvrir la MNT: {path}. Le fichier existe mais n'est pas un NetCDF lisible "
        "avec les moteurs installes. Remplace-le par un vrai NetCDF ou mets ALLOW_MNT_COORD_FALLBACK=True "
        f"dans utils.py. Details: {' | '.join(errors)}"
    )


def load_array_mnt(path, cerra_da):
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path).astype(np.float32)
    elif suffix == ".npz":
        data = np.load(path)
        key = next((k for k in MNT_VARIABLE_CANDIDATES if k in data), data.files[0])
        arr = data[key].astype(np.float32)
    elif suffix in {".tif", ".tiff"}:
        arr = load_tif_mnt(path, cerra_da)
    else:
        return None

    target_shape = (len(cerra_da.latitude), len(cerra_da.longitude))
    if arr.shape != target_shape:
        arr_t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        arr_t = F.interpolate(arr_t, size=target_shape, mode="bilinear", align_corners=False)
        arr = arr_t.squeeze().numpy()
    return arr


def load_tif_mnt(path, cerra_da):
    try:
        import rasterio
    except Exception:
        rasterio = None

    if rasterio is not None:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr = np.where(arr == src.nodata, np.nan, arr)
            bounds = src.bounds
            lon = np.linspace(bounds.left, bounds.right, src.width, dtype=np.float32)
            lat = np.linspace(bounds.top, bounds.bottom, src.height, dtype=np.float32)
        da = xr.DataArray(arr, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon})
        da = da.sortby("latitude").interp(latitude=cerra_da.latitude, longitude=cerra_da.longitude, method="linear")
        return da.values.astype(np.float32)

    try:
        from PIL import Image
    except Exception as exc:
        raise ImportError(
            f"Impossible de lire {path}. Installe rasterio ou pillow, ou convertis la MNT en .npy/.npz."
        ) from exc

    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def load_mnt_like(cerra_da, mnt_path=MNT_PATH):
    path = Path(mnt_path)
    if not path.exists():
        if ALLOW_MNT_COORD_FALLBACK:
            return coordinate_mnt_fallback(cerra_da)
        raise FileNotFoundError(
            f"MNT introuvable: {path}. Mets ton fichier MNT dans utils.MNT_PATH "
            "ou change MNT_PATH dans utils.py."
        )

    try:
        array_mnt = load_array_mnt(path, cerra_da)
    except Exception as exc:
        if ALLOW_MNT_COORD_FALLBACK:
            print(f"WARNING: Impossible de lire la MNT {path}: {exc}")
            return coordinate_mnt_fallback(cerra_da)
        raise
    if array_mnt is not None:
        arr = array_mnt
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr))
        std = std if std > 1e-8 else 1.0
        arr = np.where(np.isfinite(arr), arr, mean)
        return ((arr - mean) / std).astype(np.float32)

    try:
        ds = open_mnt_dataset(path)
    except Exception as exc:
        if ALLOW_MNT_COORD_FALLBACK:
            print(f"WARNING: {exc}")
            return coordinate_mnt_fallback(cerra_da)
        raise

    var_name = next((name for name in MNT_VARIABLE_CANDIDATES if name in ds.data_vars), None)
    if var_name is None:
        if ALLOW_MNT_COORD_FALLBACK:
            print(
                f"WARNING: Aucune variable MNT trouvee dans {path}. "
                f"Variables disponibles: {list(ds.data_vars)}"
            )
            return coordinate_mnt_fallback(cerra_da)
        raise KeyError(
            f"Aucune variable MNT trouvee dans {path}. Variables disponibles: {list(ds.data_vars)}. "
            f"Noms acceptes: {MNT_VARIABLE_CANDIDATES}"
        )
    da = ds[var_name]
    if "time" in da.dims:
        da = da.isel(time=0)
    if "band" in da.dims and da.sizes.get("band", 1) == 1:
        da = da.isel(band=0)
    rename = {}
    if "lat" in da.coords and "latitude" not in da.coords:
        rename["lat"] = "latitude"
    if "lon" in da.coords and "longitude" not in da.coords:
        rename["lon"] = "longitude"
    if "y" in da.coords and "latitude" not in da.coords:
        rename["y"] = "latitude"
    if "x" in da.coords and "longitude" not in da.coords:
        rename["x"] = "longitude"
    if rename:
        da = da.rename(rename)
    if "latitude" not in da.coords or "longitude" not in da.coords:
        if ALLOW_MNT_COORD_FALLBACK:
            print("WARNING: La MNT n'a pas de coordonnees latitude/longitude ou lat/lon.")
            return coordinate_mnt_fallback(cerra_da)
        raise ValueError("La MNT doit avoir des coordonnees latitude/longitude ou lat/lon.")

    da = da.sortby("latitude").sortby("longitude")
    da_linear = da.interp(latitude=cerra_da.latitude, longitude=cerra_da.longitude, method="linear")
    da_nearest = da.sel(latitude=cerra_da.latitude, longitude=cerra_da.longitude, method="nearest")
    arr = da_linear.values.astype(np.float32)
    nearest = da_nearest.values.astype(np.float32)
    arr = np.where(np.isfinite(arr), arr, nearest)
    if not np.isfinite(arr).any():
        if ALLOW_MNT_COORD_FALLBACK:
            print("WARNING: La MNT interpolee ne contient aucune valeur finie.")
            return coordinate_mnt_fallback(cerra_da)
        raise ValueError("La MNT interpolee ne contient aucune valeur finie.")
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    std = std if std > 1e-8 else 1.0
    arr = np.where(np.isfinite(arr), arr, mean)
    return ((arr - mean) / std).astype(np.float32)


def load_torch_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def make_corrdiff_loader(
    x_hr,
    y_hr,
    slope,
    intercept,
    mnt,
    train_ratio,
    batch_size,
    patch_size,
    max_train_samples=None,
    num_workers=0,
):
    n_train = split_index(len(x_hr), train_ratio)
    if max_train_samples is not None:
        n_train = min(n_train, int(max_train_samples))
    if hasattr(x_hr, "isel"):
        x_subset = x_hr.isel({x_hr.dims[0]: slice(0, n_train)})
        y_subset = y_hr.isel({y_hr.dims[0]: slice(0, n_train)})
    else:
        x_subset = x_hr[:n_train]
        y_subset = y_hr[:n_train]
    dataset = CorrDiffPatchDataset(x_subset, y_subset, slope, intercept, mnt, patch_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    ), n_train


def train_score_model(
    model,
    loader,
    epochs,
    device,
    lr,
    sigma_min,
    sigma_max,
    use_mnt=False,
    use_amp=True,
    lambda_consistency=0.0,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    losses = []
    log_min = np.log(sigma_min)
    log_max = np.log(sigma_max)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            if use_mnt:
                x_hr, baseline, residual, mnt = [b.to(device) for b in batch]
            else:
                x_hr, baseline, residual = [b.to(device) for b in batch]
                mnt = None
            bsz = residual.size(0)
            sigma = torch.exp(
                torch.empty(bsz, device=device).uniform_(log_min, log_max)
            ).view(bsz, 1, 1, 1)
            noise = torch.randn_like(residual)
            noisy = residual + sigma * noise
            target_score = -noise / sigma

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                score = model(x_hr, baseline, noisy, sigma, mnt)
                score_loss = ((score - target_score) ** 2 * sigma ** 2).mean()
                if lambda_consistency > 0:
                    denoised_residual = noisy + sigma ** 2 * score
                    y_hat = baseline + denoised_residual
                    lr_size = (max(1, x_hr.shape[-2] // 5), max(1, x_hr.shape[-1] // 5))
                    y_lr = F.interpolate(y_hat, size=lr_size, mode="bilinear", align_corners=False)
                    x_lr = F.interpolate(x_hr, size=lr_size, mode="bilinear", align_corners=False)
                    consistency_loss = F.mse_loss(y_lr, x_lr)
                    loss = score_loss + lambda_consistency * consistency_loss
                else:
                    consistency_loss = torch.zeros((), device=device)
                    loss = score_loss
            last_score_loss = score_loss.item()
            last_consistency_loss = consistency_loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            if lambda_consistency > 0:
                del denoised_residual, y_hat, y_lr, x_lr
            del sigma, noise, noisy, target_score, score, score_loss, loss
            del consistency_loss

        total /= max(1, len(loader))
        losses.append(total)
        cleanup_memory(device)
        if lambda_consistency > 0:
            print(
                f"Epoch {epoch + 1}/{epochs} total_loss={total:.5f} "
                f"score_loss={last_score_loss:.5f} consistency={last_consistency_loss:.5f}"
            )
        else:
            print(f"Epoch {epoch + 1}/{epochs} score_loss={total:.5f}")
    return losses


def sample_residual(model, x_hr, baseline, mnt, device, sigma_min, sigma_max, sample_steps=20,
                    langevin_steps=1, snr=0.15, batch_size=4):
    model.eval()
    xt = torch.tensor(x_hr, dtype=torch.float32).unsqueeze(1)
    bt = torch.tensor(baseline, dtype=torch.float32).unsqueeze(1)
    mt = None if mnt is None else torch.tensor(mnt, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sigmas = torch.exp(torch.linspace(np.log(sigma_max), np.log(sigma_min), sample_steps, device=device))
    out = []

    with torch.no_grad():
        for start in range(0, len(xt), batch_size):
            xb = xt[start:start + batch_size].to(device)
            bb = bt[start:start + batch_size].to(device)
            mb = None
            if mt is not None:
                mb = mt.repeat(len(xb), 1, 1, 1).to(device)
            residual = torch.randn_like(bb) * sigma_max
            for sigma_value in sigmas:
                sigma = torch.full((len(xb), 1, 1, 1), float(sigma_value), device=device)
                for _ in range(langevin_steps):
                    score = model(xb, bb, residual, sigma, mb)
                    step = (snr * float(sigma_value)) ** 2
                    noise = torch.randn_like(residual) if float(sigma_value) > sigma_min else 0.0
                    residual = residual + step * score + np.sqrt(2 * step) * noise
                    residual = residual.clamp(-6.0, 6.0)
            out.append(residual.cpu().numpy())
    return np.concatenate(out, axis=0).squeeze(1)


def apply_backprojection(pred, era5_raw, p=0.5, steps=3):
    pred_t = torch.tensor(pred, dtype=torch.float32).unsqueeze(1)
    lr_t = torch.tensor(era5_raw.values, dtype=torch.float32).unsqueeze(1)
    for _ in range(steps):
        down = F.interpolate(pred_t, size=lr_t.shape[-2:], mode="bilinear", align_corners=False)
        err = lr_t - down
        up = F.interpolate(err, size=pred_t.shape[-2:], mode="bilinear", align_corners=False)
        pred_t = pred_t + p * up
    return pred_t.squeeze(1).numpy()


def apply_bilateral_filter(pred, window=5, sigma_spatial=2.0, sigma_value=None):
    radius = window // 2
    filtered = np.empty_like(pred, dtype=np.float32)
    for t in range(len(pred)):
        img = pred[t].astype(np.float32)
        value_sigma = sigma_value
        if value_sigma is None:
            value_sigma = max(float(np.nanstd(img)) * 0.1, 1e-6)
        padded = np.pad(img, radius, mode="reflect")
        center = img
        total = np.zeros_like(img, dtype=np.float32)
        weights = np.zeros_like(img, dtype=np.float32)
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                neigh = padded[
                    radius + di:radius + di + img.shape[0],
                    radius + dj:radius + dj + img.shape[1],
                ]
                spatial_w = np.exp(-(di * di + dj * dj) / (2 * sigma_spatial ** 2))
                value_w = np.exp(-((neigh - center) ** 2) / (2 * value_sigma ** 2))
                w = spatial_w * value_w
                total += w * neigh
                weights += w
        filtered[t] = total / np.maximum(weights, 1e-8)
    return filtered


def add_corrdiff_train_args(parser):
    add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--lambda-consistency", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. Keep 0 on low-RAM machines.")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def add_corrdiff_predict_args(parser):
    add_predict_args(parser)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--langevin-steps", type=int, default=1)
    parser.add_argument("--snr", type=float, default=0.15)
    return parser


def train_corrdiff(model_dir, model_name, prefix, use_mnt=False, default_lambda_consistency=0.0):
    parser = add_corrdiff_train_args(argparse.ArgumentParser(description=f"Train {model_name}."))
    args = parser.parse_args()
    if args.lambda_consistency == 0.0 and default_lambda_consistency > 0:
        args.lambda_consistency = default_lambda_consistency
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device}")

    _, era5_interp, cerra = load_and_preprocess(args.era5, args.cerra, args.variable)
    mnt = load_mnt_like(cerra) if use_mnt else None
    x_norm, x_mu, x_std = normalize_from_train(era5_interp, args.train_ratio)
    y_norm, y_mu, y_std = normalize_from_train(cerra, args.train_ratio)

    n_train_full = split_index(len(x_norm), args.train_ratio)
    n_fit = n_train_full if args.max_train_samples is None else min(n_train_full, args.max_train_samples)
    target_shape = cerra.shape[1:]
    x_train = x_norm.isel({x_norm.dims[0]: slice(0, n_fit)})
    y_train = y_norm.isel({y_norm.dims[0]: slice(0, n_fit)})
    del era5_interp, x_norm, y_norm
    cleanup_memory(device)
    print(f"Building linear_regression baseline with train samples={n_fit} ...")
    slope, intercept = fit_linear_regression_baseline(x_train, y_train)

    loader, _ = make_corrdiff_loader(
        x_train,
        y_train,
        slope,
        intercept,
        mnt,
        train_ratio=1.0,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        max_train_samples=None,
        num_workers=args.num_workers,
    )
    del x_train, y_train
    cleanup_memory(device)

    model = ScoreUNet(in_channels=5 if use_mnt else 4, channels=args.channels).to(device)
    losses = train_score_model(
        model,
        loader,
        args.epochs,
        device,
        args.lr,
        args.sigma_min,
        args.sigma_max,
        use_mnt=use_mnt,
        use_amp=use_amp,
        lambda_consistency=args.lambda_consistency,
    )

    model_dir = Path(model_dir)
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag = variable_tag(args.variable)
    checkpoint_path = checkpoint_dir / f"{prefix}_score_{tag}.pth"
    torch.save(
        {
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "baseline_type": "linear_regression",
            "slope": slope,
            "intercept": intercept,
            "target_shape": target_shape,
            "x_mu": x_mu,
            "x_std": x_std,
            "y_mu": y_mu,
            "y_std": y_std,
            "train_ratio": args.train_ratio,
            "variable": args.variable,
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
            "lambda_consistency": args.lambda_consistency,
            "channels": args.channels,
            "use_mnt": use_mnt,
            "mnt_path": str(MNT_PATH),
        },
        checkpoint_path,
    )
    print(f"{model_name} checkpoint saved: {checkpoint_path}")
    save_hyperparams(
        model_dir,
        model_name,
        {
            "model_type": "CorrDiff score-based residual diffusion",
            "variable": args.variable,
            "era5": args.era5,
            "cerra": args.cerra,
            "device": device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_ratio": args.train_ratio,
            "train_samples_full": n_train_full,
            "train_samples_used": n_fit,
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": 1e-5,
            "patch_size": args.patch_size,
            "channels": args.channels,
            "in_channels": 5 if use_mnt else 4,
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
            "lambda_consistency": args.lambda_consistency,
            "max_train_samples": args.max_train_samples,
            "num_workers": args.num_workers,
            "amp": use_amp,
            "use_mnt": use_mnt,
            "mnt_path": MNT_PATH if use_mnt else "",
            "baseline_type": "linear_regression",
            "checkpoint": checkpoint_path,
        },
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, label="score loss")
    ax.set(title=f"{model_name} score loss", xlabel="epoch", ylabel="loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(checkpoint_dir / f"{prefix}_loss_{tag}.png", dpi=150)
    plt.close(fig)
    del fig, ax, model, loader, losses
    cleanup_memory(device)


def predict_corrdiff(model_dir, model_name, prefix, use_mnt=False, backprojection=False, bilateral=False):
    parser = add_corrdiff_predict_args(argparse.ArgumentParser(description=f"Predict with {model_name}."))
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model_dir = Path(model_dir)
    tag = variable_tag(args.variable)
    checkpoint_path = model_dir / "checkpoints" / f"{prefix}_score_{tag}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Run first: python {model_dir.name}/train.py")

    ckpt = load_torch_checkpoint(checkpoint_path, device)
    era5_sel, era5_interp_sel, cerra_sel = load_prediction_inputs(args, float(ckpt["train_ratio"]))
    mnt = load_mnt_like(cerra_sel) if use_mnt else None

    x_norm = ((era5_interp_sel - ckpt["x_mu"]) / ckpt["x_std"]).values.astype(np.float32)
    if "slope" not in ckpt or "intercept" not in ckpt:
        raise KeyError(
            "Checkpoint CorrDiff ancien format detecte (analog_regression). "
            "Relance train.py pour creer un checkpoint linear_regression."
        )
    baseline = linear_baseline_predict(x_norm, ckpt["slope"], ckpt["intercept"])

    model = ScoreUNet(in_channels=5 if use_mnt else 4, channels=int(ckpt["channels"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    residual = sample_residual(
        model,
        x_norm,
        baseline,
        mnt,
        device,
        float(ckpt["sigma_min"]),
        float(ckpt["sigma_max"]),
        sample_steps=args.sample_steps,
        langevin_steps=args.langevin_steps,
        snr=args.snr,
        batch_size=args.batch_size,
    )
    pred = (baseline + residual) * float(ckpt["y_std"]) + float(ckpt["y_mu"])

    if backprojection:
        pred = apply_backprojection(pred, era5_sel, p=0.5, steps=3)
    if bilateral:
        pred = apply_bilateral_filter(pred, window=5, sigma_spatial=2.0)

    pred_dir = model_dir / "predictions"
    metrics = evaluate(pred, cerra_sel.values, model_name)
    save_prediction_nc(pred, cerra_sel, pred_dir / f"{prefix}_pred_{tag}.nc", model_name, args.variable, metrics)
    plot_prediction(era5_sel, cerra_sel, pred, model_name, pred_dir, None, 0, args.variable)
