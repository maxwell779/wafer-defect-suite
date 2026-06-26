"""ONNX export + 지연(p50/p95/p99) + int8 동적양자화 — Stage2 웨이퍼맵 분류기(WaferCNN 32, 8클래스).
turbofan/steel MLOps 층 포팅. 입력 3×52×52. parity로 PyTorch↔ONNX 동치 검증.
*int8 동적양자화는 conv 비중 모델에 부적합(가중치만 양자화, conv 연산 fp 유지) → 지연 이득 적음을
   정직히 측정·기록. conv는 정적양자화(캘리브레이션)가 정석.* 출처: pytorch onnx, onnxruntime quantization.

사용: python -m src.mlops.export_onnx
"""
from __future__ import annotations
import json, time, os
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from src.stage2_wafermap.model import build_model
import config

CLS = config.WM_CLASSES
CKPT = config.EXPERIMENTS / "stage2_real_asl" / "best.pt"
OUT = config.EXPERIMENTS / "mlops"; OUT.mkdir(parents=True, exist_ok=True)
ONNX = config.EXPERIMENTS / "onnx"; ONNX.mkdir(parents=True, exist_ok=True)


def _bench(fn, x, warmup=5, iters=60):
    for _ in range(warmup): fn(x)
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); fn(x); ts.append((time.perf_counter() - t) * 1000)
    a = np.array(ts)
    return {"p50": round(float(np.percentile(a, 50)), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "p99": round(float(np.percentile(a, 99)), 3), "mean": round(float(a.mean()), 3)}


def run():
    net = build_model("cnn", 3, len(CLS), width=32).eval()
    sd = torch.load(CKPT, map_location="cpu"); sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    net.load_state_dict(sd)
    dummy = torch.randn(1, 3, 52, 52)
    fp = str(ONNX / "stage2_wafercnn.onnx"); q8 = str(ONNX / "stage2_wafercnn_int8.onnx")
    torch.onnx.export(net, dummy, fp, input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "b"}}, opset_version=17, dynamo=False)
    quantize_dynamic(fp, q8, weight_type=QuantType.QInt8)

    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s = ort.InferenceSession(fp, so, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(q8, so, providers=["CPUExecutionProvider"])
    xnp = dummy.numpy()
    with torch.no_grad():
        y_pt = net(dummy).numpy()
    y_on = s.run(None, {"x": xnp})[0]

    res = {"model": "Stage2 WaferCNN(width32, 8클래스)", "input": [3, 52, 52],
           "size_mb": {"onnx": round(os.path.getsize(fp) / 1e6, 2), "int8": round(os.path.getsize(q8) / 1e6, 2)},
           "parity_max_abs_err": round(float(np.abs(y_pt - y_on).max()), 6),
           "latency_ms_cpu_bs1": {
               "pytorch": _bench(lambda x: net(torch.tensor(x)).detach(), xnp),
               "onnx": _bench(lambda x: s.run(None, {"x": x}), xnp),
               "onnx_int8": _bench(lambda x: s8.run(None, {"x": x}), xnp)},
           "note": "int8 동적양자화는 conv 모델 지연 이득이 작음(가중치만 양자화). 정석은 정적양자화."}
    json.dump(res, open(OUT / "latency.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    L = res["latency_ms_cpu_bs1"]
    print(f"[onnx] {res['model']} | parity={res['parity_max_abs_err']} | "
          f"p50(ms) pt={L['pytorch']['p50']} onnx={L['onnx']['p50']} int8={L['onnx_int8']['p50']} | "
          f"{res['size_mb']['onnx']}→{res['size_mb']['int8']}MB", flush=True)


if __name__ == "__main__":
    run()
