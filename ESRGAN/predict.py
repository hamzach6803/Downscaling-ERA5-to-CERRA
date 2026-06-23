import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from model import ESRGenerator
from utils import (
    add_predict_args,
    evaluate,
    load_prediction_inputs,
    plot_prediction,
    predict_full,
    save_prediction_nc,
    variable_tag,
)


MODEL_DIR = Path(__file__).resolve().parent
PRED_DIR = MODEL_DIR / "predictions"


def main():
    parser = add_predict_args(argparse.ArgumentParser(description="Predict with ESRGAN."))
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    tag = variable_tag(args.variable)
    checkpoint = MODEL_DIR / "checkpoints" / f"esrgan_generator_{tag}.pth"
    if not checkpoint.exists():
        raise FileNotFoundError("Run first: python ESRGAN/train.py")

    ckpt = torch.load(checkpoint, map_location=device)
    era5_sel, era5_interp_sel, cerra_sel = load_prediction_inputs(args, float(ckpt["train_ratio"]))
    x = ((era5_interp_sel - ckpt["x_mu"]) / ckpt["x_std"]).values

    model = ESRGenerator(
        channels=int(ckpt.get("channels", 64)),
        num_rrdb=int(ckpt.get("rrdb_blocks", 4)),
    ).to(device)
    model.load_state_dict(ckpt["generator_state"])
    pred = predict_full(model, x, cerra_sel.shape, ckpt["y_mu"], ckpt["y_std"], device, args.batch_size)
    metrics = evaluate(pred, cerra_sel.values, "ESRGAN")
    save_prediction_nc(pred, cerra_sel, PRED_DIR / f"esrgan_pred_{tag}.nc", "ESRGAN", args.variable, metrics)
    plot_prediction(era5_sel, cerra_sel, pred, "ESRGAN", PRED_DIR, None, 0, args.variable)


if __name__ == "__main__":
    main()
