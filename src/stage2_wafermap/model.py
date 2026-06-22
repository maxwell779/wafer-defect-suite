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
