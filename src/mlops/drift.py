"""공정 드리프트 모니터 — Meruva 센서값 분포 변화(PSI/KS). 생산순서 기준 감시.
reference=전반부 생산분, current=후반부 생산분(wafer_id 순) → 6개 공정변수 PSI/KS.
양산 중 장비 상태/레시피 이동 감지(예: etch_rate 하향, 압력 상승). 출처: fiddler PSI, deepchecks KS.
*주의: 분포 변화 ≠ 결함 증가. 결함률은 별도 추적.*

사용: python -m src.mlops.drift
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import config

FEATS = ["temperature_c", "pressure_torr", "gas_flow_sccm",
         "etch_rate_nm_min", "voltage_v", "current_ma"]
OUT = config.EXPERIMENTS / "mlops"; OUT.mkdir(parents=True, exist_ok=True)


def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    a = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def run():
    df = pd.read_csv(config.MERUVA_CSV).sort_values("wafer_id").reset_index(drop=True)
    half = len(df) // 2
    ref, cur = df.iloc[:half], df.iloc[half:]
    rows = []
    for f in FEATS:
        p = psi(ref[f].to_numpy(), cur[f].to_numpy())
        ks = ks_2samp(ref[f], cur[f])
        rows.append({"feature": f, "psi": round(p, 4), "ks_stat": round(float(ks.statistic), 4),
                     "ks_p": round(float(ks.pvalue), 4),
                     "level": "alert" if p > 0.25 else ("watch" if p > 0.1 else "stable")})
    # 결함률 드리프트(전반부 vs 후반부)
    dr_ref = float(ref["defect_label"].mean()); dr_cur = float(cur["defect_label"].mean())
    share = float(np.mean([r["psi"] > 0.25 for r in rows]))
    out = {"reference": f"wafer_id 1-{half}", "current": f"wafer_id {half+1}-{len(df)}",
           "n_ref": half, "n_cur": len(df) - half,
           "process_drift": share >= 0.5, "drift_share": round(share, 3),
           "defect_rate": {"ref": round(dr_ref, 4), "cur": round(dr_cur, 4)},
           "features": rows}
    json.dump(out, open(OUT / "drift.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    for r in rows:
        print(f"[drift] {r['feature']:18s} PSI={r['psi']:.3f} KS={r['ks_stat']:.3f} → {r['level']}", flush=True)
    print(f"process_drift={out['process_drift']} (share {share:.2f}) | "
          f"결함률 {dr_ref:.2%}→{dr_cur:.2%}", flush=True)


if __name__ == "__main__":
    run()
