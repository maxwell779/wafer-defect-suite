"""스모크 테스트 — 핵심 함수/모델이 깨지지 않는지(데이터·GPU 불필요)."""
import numpy as np
import torch

from src.common.metrics import multilabel_report, format_report
from src.stage2_wafermap.model import build_model
from src.stage2_wafermap.losses import build_loss


def test_multilabel_report():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (30, 8))
    p = rng.random((30, 8))
    summ, per = multilabel_report(y, p, [str(i) for i in range(8)])
    assert 0.0 <= summ["macro_f1"] <= 1.0
    assert 0.0 <= summ["mAP"] <= 1.0
    assert len(per) == 8
    assert isinstance(format_report(summ, per), str)


def test_model_forward_shapes():
    x = torch.randn(2, 3, 52, 52)
    for arch in ["cnn", "resnet", "resnet_cbam", "vit", "dilated"]:
        out = build_model(arch, in_ch=3, n_classes=8, width=16)(x)
        assert out.shape == (2, 8), (arch, out.shape)


def test_model_pool_and_channels():
    for pool in ["gap", "gem", "maxavg"]:
        out = build_model("resnet", in_ch=5, n_classes=8, width=16, pool=pool)(torch.randn(2, 5, 52, 52))
        assert out.shape == (2, 8)


def test_losses_finite_and_backward():
    y = torch.randint(0, 2, (4, 8)).float()
    logits = torch.randn(4, 8, requires_grad=True)
    for name in ["bce", "asl", "focal", "tversky", "ldam", "smoothbce"]:
        loss = build_loss(name, pos_weight=torch.ones(8), cls_count=torch.arange(1, 9).float())
        v = loss(logits, y)
        assert torch.isfinite(v), name
        v.backward()
        assert logits.grad is not None
        logits.grad = None
