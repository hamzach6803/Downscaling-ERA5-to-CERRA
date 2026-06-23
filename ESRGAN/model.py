import torch
import torch.nn as nn


class DenseResidualBlock(nn.Module):
    def __init__(self, channels=64, growth=32, beta=0.2):
        super().__init__()
        self.beta = beta
        self.c1 = nn.Conv2d(channels, growth, 3, padding=1)
        self.c2 = nn.Conv2d(channels + growth, growth, 3, padding=1)
        self.c3 = nn.Conv2d(channels + growth * 2, growth, 3, padding=1)
        self.c4 = nn.Conv2d(channels + growth * 3, growth, 3, padding=1)
        self.c5 = nn.Conv2d(channels + growth * 4, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.c3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.c4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.c5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + self.beta * x5


class RRDB(nn.Module):
    def __init__(self, channels=64, beta=0.2):
        super().__init__()
        self.beta = beta
        self.blocks = nn.Sequential(
            DenseResidualBlock(channels),
            DenseResidualBlock(channels),
            DenseResidualBlock(channels),
        )

    def forward(self, x):
        return x + self.beta * self.blocks(x)


class ESRGenerator(nn.Module):
    def __init__(self, channels=64, num_rrdb=4):
        super().__init__()
        self.head = nn.Conv2d(1, channels, 3, padding=1)
        self.body = nn.Sequential(*[RRDB(channels) for _ in range(num_rrdb)])
        self.trunk = nn.Conv2d(channels, channels, 3, padding=1)
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, x):
        feat = self.head(x)
        body = self.trunk(self.body(feat))
        return self.tail(feat + body)


class Discriminator(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 2, channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 2, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
