import React from "react";
import { Card } from "../ui.jsx";
import runs from "../appdata/process_runs.json";
import wmaps from "../appdata/wafermaps.json";
import metrics from "../appdata/stage2_metrics.json";

export default function Dashboard({ go }) {
  const total = runs.length;
  const defects = runs.filter((r) => r.defect === 1).length;
  const fail = runs.filter((r) => r.status === "FAIL").length;
  const warn = runs.filter((r) => r.status === "WARN").length;
  const realF1 = Math.max(...metrics.comparison.filter((c) => c.name.includes("실데이터") || c.name.includes("보정")).map((c) => c.macro_f1));

  const kpis = [
    { v: total, label: "누적 공정 run", cls: "" },
    { v: ((defects / total) * 100).toFixed(2) + "%", label: `${defects}/${total} 결함`, cls: "red" },
    { v: fail + warn, label: `FAIL ${fail} · WARN ${warn}`, cls: "amber" },
    { v: wmaps.length, label: "웨이퍼맵", cls: "violet" },
  ];
  const stages = [
    { id: "stage1", n: "01", t: "공정 모니터링", d: "비지도 이상탐지", metric: total + " runs", sub: "6 파라미터 · 5 모델", c: "var(--blue)" },
    { id: "stage2", n: "02", t: "웨이퍼맵 분석", d: "멀티라벨 분류 + 위치", metric: realF1.toFixed(3) + " macro-F1", sub: "8 클래스 · 실데이터", c: "var(--violet)" },
    { id: "stage3", n: "03", t: "결함 검출/위치", d: "Grad-CAM(실) + YOLO(합성)", metric: "mAP 0.753", sub: "실데이터 위치탐지 포함", c: "var(--green)" },
  ];

  return (
    <div className="grid">
      <div>
        <h1 className="page">3-스테이지 결함 분석 파이프라인</h1>
        <div className="sub">반도체 웨이퍼 제조 — 공정 모니터링 → 웨이퍼 패턴 → 결함 위치까지 통합 콘솔 (왜 → 무엇 → 어디)</div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "repeat(4,1fr)" }}>
        {kpis.map((k, i) => (
          <Card key={i}>
            <div className="klabel">{["lot", "결함율", "경보", "maps"][i]}</div>
            <div className={"kpi " + k.cls}>{k.v}</div>
            <div className="sub" style={{ margin: 0 }}>{k.label}</div>
          </Card>
        ))}
      </div>

      <div className="grid" style={{ gridTemplateColumns: "repeat(3,1fr)" }}>
        {stages.map((s) => (
          <Card key={s.id} style={{ cursor: "pointer", borderTop: `3px solid ${s.c}` }}>
            <div onClick={() => go(s.id)}>
              <div className="klabel" style={{ color: s.c }}>STAGE {s.n} →</div>
              <h3 style={{ fontSize: 18, margin: "8px 0 2px" }}>{s.t}</h3>
              <div className="sub" style={{ margin: 0 }}>{s.d}</div>
              <div style={{ marginTop: 18, display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <span className="mono" style={{ fontWeight: 700, fontSize: 17 }}>{s.metric}</span>
                <span style={{ fontSize: 12, color: "var(--muted)" }}>{s.sub}</span>
              </div>
            </div>
          </Card>
        ))}
      </div>

      <Card title="파이프라인 개요">
        {[
          ["Stage 1", "공정 run 센서 6종을 비지도 이상탐지(LOF·OCSVM 등)로 스크리닝 → 의심 lot 플래그 (실데이터 Meruva)"],
          ["Stage 2", "웨이퍼맵 8클래스 멀티라벨 분류 + 결함 위치 추정. 합성→실제 전이 한계까지 정직하게 분석"],
          ["Stage 3", "실데이터 Grad-CAM 위치탐지(메인) + ELLIMAC YOLO11m 객체검출(부록·합성, mAP 0.753)"],
        ].map(([s, d], i) => (
          <div key={i} style={{ display: "flex", gap: 12, fontSize: 13, lineHeight: 1.6 }}>
            <span className="link" style={{ minWidth: 56, color: "var(--blue)" }}>{s}</span><span>{d}</span>
          </div>
        ))}
      </Card>
    </div>
  );
}
