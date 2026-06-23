import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DENOISING_STEPS = 20
COSINE_S = 0.008


def cosine_alpha_bar(t, steps=DENOISING_STEPS, s=COSINE_S):
    frac = (t / steps + s) / (1 + s)
    return torch.cos(frac * math.pi / 2).pow(2).clamp(1e-5, 0.999)


def add_cosine_noise(x, steps=DENOISING_STEPS):
    b = x.size(0)
    t = torch.randint(0, steps, (b,), device=x.device).float()
    alpha = cosine_alpha_bar(t, steps).view(b, 1, 1, 1)
    noise = torch.randn_like(x)
    noisy = alpha.sqrt() * x + (1 - alpha).sqrt() * noise
    return noisy, noise, t


class UNetBlock(nn.Module):
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


class CosineDDPM(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.enc1 = UNetBlock(2, channels)
        self.enc2 = UNetBlock(channels, channels * 2)
        self.mid = UNetBlock(channels * 2, channels * 2)
        self.dec1 = UNetBlock(channels * 4, channels)
        self.out = nn.Conv2d(channels * 2, 1, 3, padding=1)

    def forward(self, cond, noisy_target):
        x = torch.cat([cond, noisy_target], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        mid = self.mid(e2)
        d1 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(torch.cat([d1, e1], dim=1))
