import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DeepSD
from utils import (
    add_common_args,
    add_training_memory_args,
    load_and_preprocess,
    make_loaders,
    normalize_from_train,
    save_hyperparams,
    variable_tag,
)


MODEL_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Train DeepSD deep CNN."))
    add_training_memory_args(parser)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, era5_interp, cerra = load_and_preprocess(args.era5, args.cerra, args.variable)
    x_norm, x_mu, x_std = normalize_from_train(era5_interp, args.train_ratio)
    y_norm, y_mu, y_std = normalize_from_train(cerra, args.train_ratio)
    train_loader, test_loader = make_loaders(
        x_norm.values,
        y_norm.values,
        args.train_ratio,
        args.batch_size,
        max_train_samples=args.max_train_samples,
        num_workers=args.num_workers,
    )

    scale_h = round(cerra.shape[-2] / era5_interp.shape[-2])
    model = DeepSD(scale=scale_h).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    best_loss = np.inf
    best_state = None
    train_losses, test_losses = [], []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                test_loss += criterion(model(xb), yb).item()
        test_loss /= max(1, len(test_loader))
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch + 1}/{args.epochs} train={train_loss:.5f} test={test_loss:.5f}")

    tag = variable_tag(args.variable)
    path = CHECKPOINT_DIR / f"deepsd_best_{tag}.pth"
    torch.save(
        {
            "model_state": best_state,
            "scale": scale_h,
            "x_mu": x_mu,
            "x_std": x_std,
            "y_mu": y_mu,
            "y_std": y_std,
            "train_ratio": args.train_ratio,
            "variable": args.variable,
        },
        path,
    )
    print(f"DeepSD checkpoint saved: {path}")
    save_hyperparams(
        MODEL_DIR,
        "DeepSD",
        {
            "model_type": "deep convolutional neural network",
            "variable": args.variable,
            "era5": args.era5,
            "cerra": args.cerra,
            "device": device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_ratio": args.train_ratio,
            "max_train_samples": args.max_train_samples,
            "num_workers": args.num_workers,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "loss": "MSELoss",
            "scale": scale_h,
            "best_test_loss": best_loss,
            "checkpoint": path,
        },
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train")
    ax.plot(test_losses, label="test")
    ax.set(title="DeepSD loss", xlabel="epoch", ylabel="MSE")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHECKPOINT_DIR / f"deepsd_loss_{tag}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
