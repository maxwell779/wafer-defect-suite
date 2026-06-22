"""웨이퍼맵 멀티라벨 분류용 소형 CNN 베이스라인.

52x52 → 3 conv block (각 ÷2) → GAP → FC → 8 logits.
"""
import torch.nn as nn


def _block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class WaferCNN(nn.Module):
    def __init__(self, in_ch=3, n_classes=8, width=32, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_ch, width),
            _block(width, width * 2),
            _block(width * 2, width * 4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))  # raw logits (loss가 sigmoid 처리)


# ── 깊은 백본 + SE attention (난클래스 개선 시도) ──────────────────────
class SEBlock(nn.Module):
    """Squeeze-Excitation 채널 attention."""
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                nn.Linear(c, max(c // r, 4)), nn.ReLU(inplace=True),
                                nn.Linear(max(c // r, 4), c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x).view(x.size(0), -1, 1, 1)


class ResBlock(nn.Module):
    def __init__(self, i, o, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(i, o, 3, stride, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o))
        self.se = SEBlock(o)
        self.short = nn.Sequential() if (stride == 1 and i == o) else \
            nn.Sequential(nn.Conv2d(i, o, 1, stride), nn.BatchNorm2d(o))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.se(self.conv(x)) + self.short(x))


class WaferResNet(nn.Module):
    """SE-ResNet 류 — CNN 베이스라인보다 깊고 attention 포함."""
    def __init__(self, in_ch=3, n_classes=8, width=32, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, width, 3, 1, 1), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.layers = nn.Sequential(
            ResBlock(width, width), ResBlock(width, width * 2, 2),
            ResBlock(width * 2, width * 2), ResBlock(width * 2, width * 4, 2),
            ResBlock(width * 4, width * 4))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(width * 4, n_classes))

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))


def build_model(arch, in_ch=3, n_classes=8, width=32):
    return WaferResNet(in_ch, n_classes, width) if arch == "resnet" else WaferCNN(in_ch, n_classes, width)
