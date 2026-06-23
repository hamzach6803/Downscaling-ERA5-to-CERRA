import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from DDPM.model import CosineDDPM, DENOISING_STEPS, cosine_alpha_bar
from corrdiff_core import ScoreUNet, apply_bilateral_filter, linear_baseline_predict, load_torch_checkpoint
from utils import (
    DEFAULT_CERRA,
    DEFAULT_ERA5,
    DEFAULT_VARIABLE,
    TRAIN_RATIO,
    cleanup_memory,
    plot_cmap,
    plot_limits,
    plot_unit,
    split_index,
    values_for_plot,
    variable_tag,
)


ROOT = Path(__file__).resolve().parent
DDPM_DIR = ROOT / "DDPM"
CORRDIFF_DIR = ROOT / "CorrDiff"
OUT_DIR = ROOT / "outputs" / "comparisons" / "main" / "diffusion_steps"


def parse_args():
    def str_to_bool(value):
        if isinstance(value, bool):
            return value
        value = value.lower()
        if value in {"1", "true", "yes", "y", "oui"}:
            return True
        if value in {"0", "false", "no", "n", "non"}:
            return False
        raise argparse.ArgumentTypeError("Attendu: true/false, yes/no, oui/non, 1/0.")

    parser = argparse.ArgumentParser(
        description="Visualise les etapes de noising et denoising pour DDPM et CorrDiff."
    )
    parser.add_argument("--model", choices=["both", "ddpm", "corrdiff"], default="both")
    parser.add_argument("--date", default=None, help="Date a visualiser, ex: 2021-06-12. Aleatoire si absent.")
    parser.add_argument("--time-index", type=int, default=None, help="Index dans le split test si pas de date.")
    parser.add_argument("--era5", type=Path, default=DEFAULT_ERA5)
    parser.add_argument("--cerra", type=Path, default=DEFAULT_CERRA)
    parser.add_argument("--variable", default=DEFAULT_VARIABLE)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--ddpm-steps", type=int, default=None)
    parser.add_argument("--corrdiff-steps", type=int, default=20)
    parser.add_argument("--langevin-steps", type=int, default=1)
    parser.add_argument("--snr", type=float, default=0.15)
    parser.add_argument("--num-frames", type=int, default=6, help="Nombre d'images par ligne de processus.")
    parser.add_argument("--no-bilateral", action="store_true", help="Desactive le filtre bilateral final.")
    parser.add_argument(
        "--shared-palette",
        nargs="?",
        const=True,
        default=False,
        type=str_to_bool,
        help="Utilise une palette commune pour toute la figure. Par defaut, chaque carte est adaptative.",
    )
    return parser.parse_args()


def clean_slice(da):
    mean = da.mean(skipna=True)
    return da.fillna(mean).astype(np.float32).load()


def selected_single_date(args):
    era5_ds = xr.open_dataset(args.era5, decode_timedelta=False).sortby("time")
    cerra_ds = xr.open_dataset(args.cerra, decode_timedelta=False).sortby("time")
    try:
        if args.variable not in era5_ds or args.variable not in cerra_ds:
            raise KeyError(
                f"Variable '{args.variable}' introuvable. "
                f"ERA5={list(era5_ds.data_vars)}, CERRA={list(cerra_ds.data_vars)}"
            )

        era5 = era5_ds[args.variable]
        cerra = cerra_ds[args.variable]
        common_times = np.intersect1d(era5.time.values, cerra.time.values)
        if len(common_times) == 0:
            raise ValueError("Aucun timestep commun entre ERA5 et CERRA.")

        if args.date:
            target = np.datetime64(args.date)
            global_idx = int(np.argmin(np.abs(common_times - target)))
            test_idx = None
        else:
            n_train = split_index(len(common_times), args.train_ratio)
            n_test = len(common_times) - n_train
            if n_test <= 0:
                raise ValueError("Le split test est vide. Diminue --train-ratio.")
            test_idx = np.random.randint(0, n_test) if args.time_index is None else int(args.time_index)
            test_idx = max(0, min(test_idx, n_test - 1))
            global_idx = n_train + test_idx

        selected_time = common_times[global_idx]
        era5_raw = clean_slice(era5.sel(time=[selected_time]))
        cerra_sel = clean_slice(cerra.sel(time=[selected_time]))
        era5_interp = clean_slice(
            era5_raw.interp(latitude=cerra_sel.latitude, longitude=cerra_sel.longitude, method="linear")
        )
        print(
            f"Date selectionnee: {selected_time} "
            f"(global index {global_idx}, test index {'' if test_idx is None else test_idx})"
        )
        return era5_raw, era5_interp, cerra_sel, selected_time
    finally:
        era5_ds.close()
        cerra_ds.close()


def frame_indices(total_steps, num_frames):
    if num_frames <= 1:
        return [total_steps]
    return sorted(set(int(round(v)) for v in np.linspace(0, total_steps, num_frames)))


def denoise_capture_indices(total_steps, num_frames):
    return set(frame_indices(total_steps, num_frames))


def to_physical(arr_norm, mu, std):
    return np.asarray(arr_norm, dtype=np.float32) * float(std) + float(mu)


def plot_process_grid(rows, titles, path, variable, selected_time, model_name, shared_palette=False):
    plot_rows = [[values_for_plot(item, variable) for item in row] for row in rows]
    shared_limits = None
    if shared_palette:
        all_values = np.concatenate([np.asarray(item).reshape(-1) for row in plot_rows for item in row])
        finite = all_values[np.isfinite(all_values)]
        shared_limits = plot_limits(finite, variable) if finite.size else (None, None)

    cmap = plot_cmap(variable)
    unit = plot_unit(variable)
    n_rows = len(rows)
    n_cols = max(len(row) for row in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows), squeeze=False)
    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c >= len(row):
                ax.axis("off")
                continue
            vmin, vmax = shared_limits if shared_limits is not None else plot_limits(plot_rows[r][c], variable)
            kwargs = {}
            if vmin is not None and vmax is not None:
                kwargs = {"vmin": vmin, "vmax": vmax}
            im = ax.imshow(plot_rows[r][c], cmap=cmap, origin="lower", **kwargs)
            ax.set_title(titles[r][c], fontsize=9)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(unit)
    fig.suptitle(f"{model_name} - {variable} - {selected_time}", fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardee: {path}")


def visualize_ddpm(args, era5_interp, cerra_sel, selected_time, device):
    tag = variable_tag(args.variable)
    ckpt_path = DDPM_DIR / "checkpoints" / f"ddpm_cosine_best_{tag}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint DDPM introuvable: {ckpt_path}")
    ckpt = load_torch_checkpoint(ckpt_path, device)
    steps = int(args.ddpm_steps or ckpt.get("steps", DENOISING_STEPS))

    x_norm = ((era5_interp - ckpt["x_mu"]) / ckpt["x_std"]).values.astype(np.float32)
    y_norm = ((cerra_sel - ckpt["y_mu"]) / ckpt["y_std"]).values.astype(np.float32)
    h, w = cerra_sel.shape[-2:]
    cond = torch.tensor(x_norm, dtype=torch.float32, device=device).unsqueeze(1)
    cond = F.interpolate(cond, size=(h, w), mode="bilinear", align_corners=False)
    y0 = torch.tensor(y_norm, dtype=torch.float32, device=device).unsqueeze(1)
    fixed_noise = torch.randn_like(y0)

    model = CosineDDPM().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    noising_frames = []
    noising_titles = []
    for t in frame_indices(steps - 1, args.num_frames):
        tt = torch.full((1,), float(t), device=device)
        alpha = cosine_alpha_bar(tt, steps).view(1, 1, 1, 1)
        noisy = alpha.sqrt() * y0 + (1 - alpha).sqrt() * fixed_noise
        noising_frames.append(to_physical(noisy.squeeze().detach().cpu().numpy(), ckpt["y_mu"], ckpt["y_std"]))
        noising_titles.append(f"Noising t={t}")

    denoising_frames = []
    denoising_titles = []
    capture = denoise_capture_indices(steps, args.num_frames)
    y = torch.randn_like(y0)
    with torch.no_grad():
        if 0 in capture:
            denoising_frames.append(to_physical(y.squeeze().cpu().numpy(), ckpt["y_mu"], ckpt["y_std"]))
            denoising_titles.append("Denoising init")
        for step in range(1, steps + 1):
            y = y - model(cond, y) / steps
            if step in capture:
                denoising_frames.append(to_physical(y.squeeze().cpu().numpy(), ckpt["y_mu"], ckpt["y_std"]))
                denoising_titles.append(f"Denoising {step}/{steps}")

    if not args.no_bilateral:
        final = apply_bilateral_filter(np.asarray([denoising_frames[-1]], dtype=np.float32), window=5, sigma_spatial=2.0)[0]
        denoising_frames[-1] = final
        denoising_titles[-1] += " + bilateral"

    date_tag = np.datetime_as_string(np.datetime64(selected_time), unit="D").replace("-", "")
    path = args.out_dir / f"ddpm_steps_{date_tag}_{tag}.png"
    plot_process_grid(
        [noising_frames, denoising_frames],
        [noising_titles, denoising_titles],
        path,
        args.variable,
        selected_time,
        "DDPM",
        shared_palette=args.shared_palette,
    )
    del model, cond, y0, fixed_noise, y
    cleanup_memory(device)


def visualize_corrdiff(args, era5_interp, cerra_sel, selected_time, device):
    tag = variable_tag(args.variable)
    ckpt_path = CORRDIFF_DIR / "checkpoints" / f"corrdiff_score_{tag}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint CorrDiff introuvable: {ckpt_path}")
    ckpt = load_torch_checkpoint(ckpt_path, device)

    x_norm = ((era5_interp - ckpt["x_mu"]) / ckpt["x_std"]).values.astype(np.float32)
    y_norm = ((cerra_sel - ckpt["y_mu"]) / ckpt["y_std"]).values.astype(np.float32)
    baseline = linear_baseline_predict(x_norm, ckpt["slope"], ckpt["intercept"])
    residual_true = y_norm - baseline

    x_hr = torch.tensor(x_norm, dtype=torch.float32, device=device).unsqueeze(1)
    baseline_t = torch.tensor(baseline, dtype=torch.float32, device=device).unsqueeze(1)
    residual_t = torch.tensor(residual_true, dtype=torch.float32, device=device).unsqueeze(1)
    fixed_noise = torch.randn_like(residual_t)
    sigma_min = float(ckpt["sigma_min"])
    sigma_max = float(ckpt["sigma_max"])
    sigmas = torch.exp(torch.linspace(np.log(sigma_max), np.log(sigma_min), args.corrdiff_steps, device=device))

    model = ScoreUNet(in_channels=4, channels=int(ckpt["channels"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    noising_frames = []
    noising_titles = []
    sigma_idx = frame_indices(len(sigmas) - 1, args.num_frames)
    for idx in sigma_idx:
        sigma = sigmas[idx].view(1, 1, 1, 1)
        noisy_residual = residual_t + sigma * fixed_noise
        field = baseline_t + noisy_residual
        noising_frames.append(to_physical(field.squeeze().detach().cpu().numpy(), ckpt["y_mu"], ckpt["y_std"]))
        noising_titles.append(f"Noising sigma={float(sigmas[idx]):.3f}")

    denoising_frames = []
    denoising_titles = []
    capture = denoise_capture_indices(len(sigmas), args.num_frames)
    residual = torch.randn_like(baseline_t) * sigma_max
    with torch.no_grad():
        if 0 in capture:
            field = baseline_t + residual
            denoising_frames.append(to_physical(field.squeeze().cpu().numpy(), ckpt["y_mu"], ckpt["y_std"]))
            denoising_titles.append("Denoising init")
        for step, sigma_value in enumerate(sigmas, start=1):
            sigma = torch.full((1, 1, 1, 1), float(sigma_value), device=device)
            for _ in range(args.langevin_steps):
                score = model(x_hr, baseline_t, residual, sigma, None)
                step_size = (args.snr * float(sigma_value)) ** 2
                noise = torch.randn_like(residual) if float(sigma_value) > sigma_min else 0.0
                residual = residual + step_size * score + np.sqrt(2 * step_size) * noise
                residual = residual.clamp(-6.0, 6.0)
            if step in capture:
                field = baseline_t + residual
                denoising_frames.append(to_physical(field.squeeze().cpu().numpy(), ckpt["y_mu"], ckpt["y_std"]))
                denoising_titles.append(f"Denoising {step}/{len(sigmas)}")

    if not args.no_bilateral:
        final = apply_bilateral_filter(np.asarray([denoising_frames[-1]], dtype=np.float32), window=5, sigma_spatial=2.0)[0]
        denoising_frames[-1] = final
        denoising_titles[-1] += " + bilateral"

    date_tag = np.datetime_as_string(np.datetime64(selected_time), unit="D").replace("-", "")
    path = args.out_dir / f"corrdiff_steps_{date_tag}_{tag}.png"
    plot_process_grid(
        [noising_frames, denoising_frames],
        [noising_titles, denoising_titles],
        path,
        args.variable,
        selected_time,
        "CorrDiff",
        shared_palette=args.shared_palette,
    )
    del model, x_hr, baseline_t, residual_t, fixed_noise, residual
    cleanup_memory(device)


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    era5_raw, era5_interp, cerra_sel, selected_time = selected_single_date(args)
    del era5_raw

    if args.model in {"both", "ddpm"}:
        visualize_ddpm(args, era5_interp, cerra_sel, selected_time, device)
    if args.model in {"both", "corrdiff"}:
        visualize_corrdiff(args, era5_interp, cerra_sel, selected_time, device)


if __name__ == "__main__":
    main()
