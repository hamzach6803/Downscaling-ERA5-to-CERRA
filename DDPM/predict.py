import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from model import DENOISING_STEPS, CosineDDPM
from corrdiff_core import apply_bilateral_filter
from utils import BATCH_SIZE, add_predict_args, evaluate, load_prediction_inputs, plot_prediction, save_prediction_nc, variable_tag


MODEL_DIR = Path(__file__).resolve().parent
PRED_DIR = MODEL_DIR / "predictions"


def predict_ddpm(model, x, h, w, y_mu, y_std, steps, device):
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    out = []
    with torch.no_grad():
        for i in range(0, len(xt), BATCH_SIZE):
            xb = xt[i:i + BATCH_SIZE].to(device)
            cond = F.interpolate(xb, size=(h, w), mode="bilinear", align_corners=False)
            y = torch.randn(len(xb), 1, h, w, device=device)
            for _ in range(steps):
                y = y - model(cond, y) / steps
            out.append(y.cpu().numpy())
    pred = np.concatenate(out, axis=0).squeeze(1)
    return pred * y_std + y_mu


def main():
    parser = add_predict_args(argparse.ArgumentParser(description="Predict with DDPM cosine."))
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    tag = variable_tag(args.variable)
    checkpoint = MODEL_DIR / "checkpoints" / f"ddpm_cosine_best_{tag}.pth"
    if not checkpoint.exists():
        raise FileNotFoundError("Run first: python DDPM/train.py")

    ckpt = torch.load(checkpoint, map_location=device)
    era5_sel, era5_interp_sel, cerra_sel = load_prediction_inputs(args, float(ckpt["train_ratio"]))
    x = ((era5_interp_sel - ckpt["x_mu"]) / ckpt["x_std"]).values

    model = CosineDDPM().to(device)
    model.load_state_dict(ckpt["model_state"])
    steps = args.steps or int(ckpt.get("steps", DENOISING_STEPS))
    pred = predict_ddpm(model, x, cerra_sel.shape[-2], cerra_sel.shape[-1], ckpt["y_mu"], ckpt["y_std"], steps, device)
    pred = apply_bilateral_filter(pred, window=5, sigma_spatial=2.0)
    metrics = evaluate(pred, cerra_sel.values, "DDPM Cosine")
    save_prediction_nc(pred, cerra_sel, PRED_DIR / f"ddpm_cosine_pred_{tag}.nc", "DDPM Cosine", args.variable, metrics)
    plot_prediction(era5_sel, cerra_sel, pred, "DDPM Cosine", PRED_DIR, None, 0, args.variable)


if __name__ == "__main__":
    main()
