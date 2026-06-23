"""멀티라벨 불균형 손실.

- bce : BCEWithLogits + pos_weight (간단·강건)
- asl : Asymmetric Loss (Ridnik et al. 2021) — 멀티라벨 불균형 정석.
        음성(negative)을 더 강하게 down-weight + 확률 시프트로 쉬운 음성 무시.
"""
import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        p_pos = p
        p_neg = 1 - p
        if self.clip and self.clip > 0:  # probability shift: 쉬운 음성 버림
            p_neg = (p_neg + self.clip).clamp(max=1)
        loss_pos = y * torch.log(p_pos.clamp(min=self.eps))
        loss_neg = (1 - y) * torch.log(p_neg.clamp(min=self.eps))
        # focusing
        w_pos = torch.pow(1 - p_pos, self.gamma_pos)
        w_neg = torch.pow(p, self.gamma_neg)
        loss = loss_pos * w_pos + loss_neg * w_neg
        return -loss.sum(dim=1).mean()


class FocalLoss(nn.Module):
    """멀티라벨 focal — 쉬운 샘플 down-weight (희귀클래스 표적)."""
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, y):
        import torch.nn.functional as F
        ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=self.pos_weight)
        p = torch.sigmoid(logits)
        pt = p * y + (1 - p) * (1 - y)
        return (((1 - pt) ** self.gamma) * ce).sum(dim=1).mean()


class TverskyLoss(nn.Module):
    """멀티라벨 Tversky — FN/FP 비대칭 가중(α=FP, β=FN). β↑ → recall 표적(희귀클래스)."""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        tp = (p * y).sum(0)
        fp = (p * (1 - y)).sum(0)
        fn = ((1 - p) * y).sum(0)
        tv = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tv).mean()


class LDAMLoss(nn.Module):
    """LDAM(Cao et al. 2019) 멀티라벨 적응 — 희귀클래스에 큰 마진(1/n^0.25). BCE 기반."""
    def __init__(self, cls_count, max_m=0.5, pos_weight=None):
        super().__init__()
        m = 1.0 / torch.sqrt(torch.sqrt(cls_count.float().clamp(min=1)))
        self.m = (m / m.max() * max_m)
        self.pos_weight = pos_weight

    def forward(self, logits, y):
        import torch.nn.functional as F
        m = self.m.to(logits.device)
        adj = logits - m * y           # 양성 로짓에서 마진만큼 깎음 → 더 강한 분리 요구
        return F.binary_cross_entropy_with_logits(adj, y, pos_weight=self.pos_weight)


class SmoothBCE(nn.Module):
    """label smoothing BCE (eps): 과신 억제."""
    def __init__(self, eps=0.05, pos_weight=None):
        super().__init__()
        self.eps, self.pos_weight = eps, pos_weight

    def forward(self, logits, y):
        import torch.nn.functional as F
        yt = y * (1 - self.eps) + 0.5 * self.eps
        return F.binary_cross_entropy_with_logits(logits, yt, pos_weight=self.pos_weight)


def build_loss(name, pos_weight=None, cls_count=None):
    if name == "asl":
        return AsymmetricLoss()
    if name == "focal":
        return FocalLoss(gamma=2.0, pos_weight=pos_weight)
    if name == "tversky":
        return TverskyLoss()
    if name == "ldam":
        return LDAMLoss(cls_count if cls_count is not None else torch.ones(8), pos_weight=pos_weight)
    if name == "smoothbce":
        return SmoothBCE(eps=0.05, pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
