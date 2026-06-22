"""멀티라벨 평가 지표.

busbar 교훈: 전체 정확도 하나로 뭉개지 말고 per-class 로 본다(희귀클래스 추적).
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, hamming_loss,
)


def multilabel_report(y_true, y_prob, class_names, thresh=0.5):
    """y_true,(N,C) 0/1 ; y_prob,(N,C) [0,1]. returns (summary dict, per-class list)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= thresh).astype(int)

    per = []
    for i, name in enumerate(class_names):
        sup = int(y_true[:, i].sum())
        ap = average_precision_score(y_true[:, i], y_prob[:, i]) if sup > 0 else float("nan")
        per.append(dict(
            cls=name, support=sup,
            precision=precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            recall=recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            f1=f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            ap=ap,
        ))

    summary = dict(
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        micro_f1=f1_score(y_true, y_pred, average="micro", zero_division=0),
        mAP=float(np.nanmean([p["ap"] for p in per])),
        exact_match=float((y_pred == y_true).all(axis=1).mean()),  # subset accuracy
        hamming=hamming_loss(y_true, y_pred),
    )
    return summary, per


def format_report(summary, per) -> str:
    lines = [f"{'class':12s}{'sup':>7}{'prec':>8}{'rec':>8}{'f1':>8}{'AP':>8}"]
    for p in per:
        ap = p["ap"]
        lines.append(
            f"{p['cls']:12s}{p['support']:7d}{p['precision']:8.3f}"
            f"{p['recall']:8.3f}{p['f1']:8.3f}{ap:8.3f}"
        )
    lines.append("-" * 51)
    lines.append(
        f"macro-F1 {summary['macro_f1']:.4f} | micro-F1 {summary['micro_f1']:.4f} | "
        f"mAP {summary['mAP']:.4f} | exact-match {summary['exact_match']:.4f} | "
        f"hamming {summary['hamming']:.4f}"
    )
    return "\n".join(lines)
