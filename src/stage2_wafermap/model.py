"""웨이퍼맵 멀티라벨 분류용 모델들 (CNN / SE-ResNet / CBAM-ResNet).

52x52 → conv → GAP → FC → 8 logits.
"""
import torch
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


class SpatialAttn(nn.Module):
    """CBAM 공간 attention (avg+max 채널 풀링 → conv)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx = x.max(1, keepdim=True)[0]
        return x * self.conv(torch.cat([avg, mx], 1))


class ResBlock(nn.Module):
    """se: 채널 attention만(하위호환 키 self.se) / cbam: + 공간 attention(self.sp)."""
    def __init__(self, i, o, stride=1, attn="se"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(i, o, 3, stride, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o))
        self.se = SEBlock(o)
        self.sp = SpatialAttn() if attn == "cbam" else None
        self.short = nn.Sequential() if (stride == 1 and i == o) else \
            nn.Sequential(nn.Conv2d(i, o, 1, stride), nn.BatchNorm2d(o))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.se(self.conv(x))
        if self.sp is not None:
            h = self.sp(h)
        return self.act(h + self.short(x))


class WaferResNet(nn.Module):
    """SE/CBAM-ResNet 류 — CNN 베이스라인보다 깊고 attention 포함."""
    def __init__(self, in_ch=3, n_classes=8, width=32, dropout=0.3, attn="se"):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, width, 3, 1, 1), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.layers = nn.Sequential(
            ResBlock(width, width, attn=attn), ResBlock(width, width * 2, 2, attn=attn),
            ResBlock(width * 2, width * 2, attn=attn), ResBlock(width * 2, width * 4, 2, attn=attn),
            ResBlock(width * 4, width * 4, attn=attn))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(width * 4, n_classes))

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))


def _tv_resnet(in_ch, n_classes, depth=18):
    """torchvision ResNet(이질 패밀리) — 52x52 소입력용으로 stem 수정."""
    import torchvision.models as tvm
    net = {18: tvm.resnet18, 34: tvm.resnet34}[depth](weights=None)
    net.conv1 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False)   # 7x7s2 → 3x3s1(소입력)
    net.maxpool = nn.Identity()                              # 해상도 보존
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    return net


def build_model(arch, in_ch=3, n_classes=8, width=32):
    if arch == "resnet":
        return WaferResNet(in_ch, n_classes, width, attn="se")
    if arch == "resnet_cbam":
        return WaferResNet(in_ch, n_classes, width, attn="cbam")
    if arch == "tvresnet18":
        return _tv_resnet(in_ch, n_classes, 18)
    if arch == "tvresnet34":
        return _tv_resnet(in_ch, n_classes, 34)
    return WaferCNN(in_ch, n_classes, width)
