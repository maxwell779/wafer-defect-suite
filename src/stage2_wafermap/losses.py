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


def build_loss(name, pos_weight=None):
    if name == "asl":
        return AsymmetricLoss()
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
