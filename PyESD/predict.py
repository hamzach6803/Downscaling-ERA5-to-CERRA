import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils import add_predict_args, evaluate, load_prediction_inputs, plot_prediction, save_prediction_nc, variable_tag


MODEL_DIR = Path(__file__).resolve().parent
PRED_DIR = MODEL_DIR / "predictions"


def main():
    parser = add_predict_args(argparse.ArgumentParser(description="Predict with PyESD."))
    args = parser.parse_args()
    tag = variable_tag(args.variable)
    checkpoint = MODEL_DIR / "checkpoints" / f"pyesd_linear_{tag}.npz"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Run first: python PyESD/train.py")

    ckpt = np.load(checkpoint)
    era5_sel, era5_interp_sel, cerra_sel = load_prediction_inputs(args, float(ckpt["train_ratio"]))
    x = ((era5_interp_sel - float(ckpt["x_mu"])) / float(ckpt["x_std"])).values
    pred_norm = x * ckpt["slope"] + ckpt["intercept"]
    pred = pred_norm * float(ckpt["y_std"]) + float(ckpt["y_mu"])

    metrics = evaluate(pred, cerra_sel.values, "PyESD")
    save_prediction_nc(pred, cerra_sel, PRED_DIR / f"pyesd_pred_{tag}.nc", "PyESD", args.variable, metrics)
    plot_prediction(era5_sel, cerra_sel, pred, "PyESD", PRED_DIR, None, 0, args.variable)


if __name__ == "__main__":
    main()
