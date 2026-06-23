import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import Discriminator, ESRGenerator
from utils import (
    add_common_args,
    add_training_memory_args,
    as_float32_array,
    cleanup_memory,
    load_and_preprocess,
    normalize_from_train,
    save_hyperparams,
    split_index,
    variable_tag,
)


MODEL_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


class RandomPatchDataset(Dataset):
    def __init__(self, x, y, patch_size=64):
        self.x = as_float32_array(x)
        self.y = as_float32_array(y)
        self.patch_size = min(patch_size, x.shape[-2], x.shape[-1])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        h, w = x.shape
        p = self.patch_size
        i = torch.randint(0, h - p + 1, (1,)).item()
        j = torch.randint(0, w - p + 1, (1,)).item()
        return (
            torch.tensor(x[None, i:i + p, j:j + p], dtype=torch.float32),
            torch.tensor(y[None, i:i + p, j:j + p], dtype=torch.float32),
        )


def make_patch_loaders(x, y, train_ratio, batch_size, patch_size, max_train_samples=None, num_workers=0):
    n_train = split_index(len(x), train_ratio)
    if max_train_samples is not None:
        n_train = min(n_train, int(max_train_samples))
    n_test_start = split_index(len(x), train_ratio)
    train = DataLoader(
        RandomPatchDataset(x[:n_train], y[:n_train], patch_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test = DataLoader(
        RandomPatchDataset(x[n_test_start:], y[n_test_start:], patch_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train, test


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Train ESRGAN for ERA5 to CERRA downscaling."))
    add_training_memory_args(parser)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--adv-weight", type=float, default=1e-3)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--rrdb-blocks", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, era5_interp, cerra = load_and_preprocess(args.era5, args.cerra, args.variable)
    x_norm, x_mu, x_std = normalize_from_train(era5_interp, args.train_ratio)
    y_norm, y_mu, y_std = normalize_from_train(cerra, args.train_ratio)
    train_loader, test_loader = make_patch_loaders(
        x_norm.values,
        y_norm.values,
        args.train_ratio,
        args.batch_size,
        args.patch_size,
        max_train_samples=args.max_train_samples,
        num_workers=args.num_workers,
    )
    del era5_interp, cerra, x_norm, y_norm
    cleanup_memory(device)

    generator = ESRGenerator(channels=args.channels, num_rrdb=args.rrdb_blocks).to(device)
    discriminator = Discriminator(channels=args.channels).to(device)
    opt_g = optim.AdamW(generator.parameters(), lr=1e-4, betas=(0.9, 0.99))
    opt_d = optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.99))
    l1 = nn.L1Loss()
    bce = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    train_g_losses, train_d_losses, test_l1_losses = [], [], []
    best_state, best_test = None, float("inf")

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        g_loss_sum = 0.0
        d_loss_sum = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            real = torch.ones((xb.size(0), 1), device=device)
            fake_label = torch.zeros((xb.size(0), 1), device=device)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                fake = generator(xb)
            opt_d.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                d_real = bce(discriminator(yb), real)
                d_fake = bce(discriminator(fake), fake_label)
                d_loss = 0.5 * (d_real + d_fake)
            scaler.scale(d_loss).backward()
            scaler.step(opt_d)

            opt_g.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                fake = generator(xb)
                pixel_loss = l1(fake, yb)
                adv_loss = bce(discriminator(fake), real)
                g_loss = args.l1_weight * pixel_loss + args.adv_weight * adv_loss
            scaler.scale(g_loss).backward()
            scaler.step(opt_g)
            scaler.update()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()

        generator.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    test_loss += l1(generator(xb), yb).item()
        test_loss /= max(1, len(test_loader))
        g_loss_sum /= max(1, len(train_loader))
        d_loss_sum /= max(1, len(train_loader))
        train_g_losses.append(g_loss_sum)
        train_d_losses.append(d_loss_sum)
        test_l1_losses.append(test_loss)

        if test_loss < best_test:
            best_test = test_loss
            best_state = {k: v.detach().cpu() for k, v in generator.state_dict().items()}

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"G={g_loss_sum:.5f} D={d_loss_sum:.5f} test_l1={test_loss:.5f}"
        )
        del xb, yb, real, fake_label, fake
        cleanup_memory(device)

    tag = variable_tag(args.variable)
    path = CHECKPOINT_DIR / f"esrgan_generator_{tag}.pth"
    torch.save(
        {
            "generator_state": best_state,
            "x_mu": x_mu,
            "x_std": x_std,
            "y_mu": y_mu,
            "y_std": y_std,
            "train_ratio": args.train_ratio,
            "variable": args.variable,
            "channels": args.channels,
            "rrdb_blocks": args.rrdb_blocks,
            "patch_size": args.patch_size,
        },
        path,
    )
    print(f"ESRGAN checkpoint saved: {path}")
    save_hyperparams(
        MODEL_DIR,
        "ESRGAN",
        {
            "model_type": "GAN with RRDB generator",
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
            "generator_betas": "(0.9, 0.99)",
            "discriminator_betas": "(0.9, 0.99)",
            "adv_weight": args.adv_weight,
            "l1_weight": args.l1_weight,
            "patch_size": args.patch_size,
            "channels": args.channels,
            "rrdb_blocks": args.rrdb_blocks,
            "amp": use_amp,
            "best_test_l1": best_test,
            "checkpoint": path,
        },
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_g_losses, label="G train")
    ax.plot(train_d_losses, label="D train")
    ax.plot(test_l1_losses, label="test L1")
    ax.set(title="ESRGAN loss", xlabel="epoch", ylabel="loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHECKPOINT_DIR / f"esrgan_loss_{tag}.png", dpi=150)
    plt.close(fig)
    del fig, ax, generator, discriminator, opt_g, opt_d, l1, bce, scaler
    del train_loader, test_loader, best_state
    cleanup_memory(device)


if __name__ == "__main__":
    main()
