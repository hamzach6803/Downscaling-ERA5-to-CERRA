import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import COSINE_S, DENOISING_STEPS, CosineDDPM, add_cosine_noise
from utils import (
    add_common_args,
    add_training_memory_args,
    cleanup_memory,
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
    parser = add_common_args(argparse.ArgumentParser(description="Train conditional DDPM with cosine noise schedule."))
    add_training_memory_args(parser)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=DENOISING_STEPS)
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, cosine_s={COSINE_S}, steps={args.steps}")

    _, era5_interp, cerra = load_and_preprocess(args.era5, args.cerra, args.variable)
    x_norm, x_mu, x_std = normalize_from_train(era5_interp, args.train_ratio)
    y_norm, y_mu, y_std = normalize_from_train(cerra, args.train_ratio)
    train_loader, _ = make_loaders(
        x_norm.values,
        y_norm.values,
        args.train_ratio,
        args.batch_size,
        max_train_samples=args.max_train_samples,
        num_workers=args.num_workers,
    )

    h, w = cerra.shape[-2:]
    del era5_interp, cerra, x_norm, y_norm
    cleanup_memory(device)

    model = CosineDDPM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    losses = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            cond = F.interpolate(xb, size=(h, w), mode="bilinear", align_corners=False)
            noisy, noise, _ = add_cosine_noise(yb, args.steps)
            optimizer.zero_grad()
            loss = criterion(model(cond, noisy), noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(1, len(train_loader))
        losses.append(epoch_loss)
        del xb, yb, cond, noisy, noise, loss
        cleanup_memory(device)
        print(f"Epoch {epoch + 1}/{args.epochs} loss={epoch_loss:.5f}")

    tag = variable_tag(args.variable)
    path = CHECKPOINT_DIR / f"ddpm_cosine_best_{tag}.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "x_mu": x_mu,
            "x_std": x_std,
            "y_mu": y_mu,
            "y_std": y_std,
            "train_ratio": args.train_ratio,
            "variable": args.variable,
            "steps": args.steps,
            "cosine_s": COSINE_S,
        },
        path,
    )
    print(f"DDPM checkpoint saved: {path}")
    save_hyperparams(
        MODEL_DIR,
        "DDPM Cosine",
        {
            "model_type": "conditional denoising diffusion",
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
            "loss": "MSELoss",
            "steps": args.steps,
            "cosine_s": COSINE_S,
            "checkpoint": path,
        },
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, label="train")
    ax.set(title="DDPM cosine loss", xlabel="epoch", ylabel="noise MSE")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHECKPOINT_DIR / f"ddpm_loss_{tag}.png", dpi=150)
    plt.close(fig)
    del fig, ax, model, optimizer, criterion, train_loader
    cleanup_memory(device)


if __name__ == "__main__":
    main()
