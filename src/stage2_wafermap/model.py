"""웨이퍼맵 멀티라벨 분류용 모델들 (CNN / SE-ResNet / CBAM-ResNet / ViT-lite).

52x52 → conv → GAP → FC → 8 logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPool(nn.Module):
    """Generalized Mean pooling — GAP(p=1)과 GMP(p=∞) 사이 학습형 풀링(산발 결함 표적)."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)
        return x


def make_pool(kind):
    """gap | gem | maxavg(=avg+max concat, 채널 2배)."""
    if kind == "gem":
        return GeMPool(), 1
    if kind == "maxavg":
        class _MA(nn.Module):
            def forward(self, x):
                return torch.cat([F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)], 1)
        return _MA(), 2
    return nn.AdaptiveAvgPool2d(1), 1


def _block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class WaferCNN(nn.Module):
    def __init__(self, in_ch=3, n_classes=8, width=32, dropout=0.3, pool="gap"):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_ch, width),
            _block(width, width * 2),
            _block(width * 2, width * 4),
        )
        pl, mul = make_pool(pool)
        self.head = nn.Sequential(
            pl, nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(width * 4 * mul, n_classes),
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
    def __init__(self, in_ch=3, n_classes=8, width=32, dropout=0.3, attn="se", pool="gap"):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, width, 3, 1, 1), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.layers = nn.Sequential(
            ResBlock(width, width, attn=attn), ResBlock(width, width * 2, 2, attn=attn),
            ResBlock(width * 2, width * 2, attn=attn), ResBlock(width * 2, width * 4, 2, attn=attn),
            ResBlock(width * 4, width * 4, attn=attn))
        pl, mul = make_pool(pool)
        self.head = nn.Sequential(pl, nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(width * 4 * mul, n_classes))

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))


class WaferViT(nn.Module):
    """ViT-lite — conv patch-embed(4x4 patch=13x13 토큰) + Transformer encoder.
    소데이터라 작게(depth/heads 절제), 산발·전역 패턴(Scratch/Random) 표적."""
    def __init__(self, in_ch=3, n_classes=8, dim=128, depth=4, heads=4, patch=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Conv2d(in_ch, dim, patch, patch)         # 52/4=13 → 169 토큰
        ntok = (52 // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, ntok + 1, dim))
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, n_classes)
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        z = self.embed(x).flatten(2).transpose(1, 2)             # B,N,dim
        z = torch.cat([self.cls.expand(z.size(0), -1, -1), z], 1) + self.pos
        z = self.tr(z)
        return self.fc(self.norm(z[:, 0]))


class WaferDilatedCNN(nn.Module):
    """넓은 수용영역 — dilated conv(다운샘플 없이 receptive field↑), 전역 패턴 표적."""
    def __init__(self, in_ch=3, n_classes=8, width=48, dropout=0.3, pool="gap"):
        super().__init__()
        def blk(i, o, d):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=d, dilation=d), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.features = nn.Sequential(
            blk(in_ch, width, 1), blk(width, width, 2),
            blk(width, width * 2, 4), blk(width * 2, width * 2, 8),
            blk(width * 2, width * 4, 1))
        pl, mul = make_pool(pool)
        self.head = nn.Sequential(pl, nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(width * 4 * mul, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def _tv_resnet(in_ch, n_classes, depth=18):
    """torchvision ResNet(이질 패밀리) — 52x52 소입력용으로 stem 수정."""
    import torchvision.models as tvm
    net = {18: tvm.resnet18, 34: tvm.resnet34}[depth](weights=None)
    net.conv1 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False)   # 7x7s2 → 3x3s1(소입력)
    net.maxpool = nn.Identity()                              # 해상도 보존
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    return net


def build_model(arch, in_ch=3, n_classes=8, width=32, pool="gap", dropout=0.3):
    if arch == "resnet":
        return WaferResNet(in_ch, n_classes, width, dropout=dropout, attn="se", pool=pool)
    if arch == "resnet_cbam":
        return WaferResNet(in_ch, n_classes, width, dropout=dropout, attn="cbam", pool=pool)
    if arch == "tvresnet18":
        return _tv_resnet(in_ch, n_classes, 18)
    if arch == "tvresnet34":
        return _tv_resnet(in_ch, n_classes, 34)
    if arch == "vit":
        return WaferViT(in_ch, n_classes, dim=max(width * 2, 96), dropout=dropout)
    if arch == "dilated":
        return WaferDilatedCNN(in_ch, n_classes, width, dropout=dropout, pool=pool)
    return WaferCNN(in_ch, n_classes, width, dropout=dropout, pool=pool)
