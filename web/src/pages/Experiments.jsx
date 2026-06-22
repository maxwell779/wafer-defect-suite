import React from "react";
import { Card, HBars, LineChart, Confusion } from "../ui.jsx";
import m from "../appdata/stage2_metrics.json";

export default function Experiments() {
  const cmp = m.comparison;
  return (
    <div className="grid">
      <div><h1 className="page">Experiments — 학습 전략 비교</h1>
        <div className="sub">합성·전이·증강·실데이터·SSL 결과 (negative result 포함, 과장 없음)</div></div>

      <Card title="학습 전략별 성능 추이 (macro-F1 / mAP)" sub={cmp.map((c) => c.name + " " + c.macro_f1).join(" → ")}>
        <LineChart
          xlabels={cmp.map((c) => c.name)}
          series={[
            { name: "macro-F1", color: "var(--blue)", points: cmp.map((c) => c.macro_f1) },
            { name: "mAP", color: "var(--muted)", dash: "5 4", points: cmp.map((c) => c.mAP) },
          ]}
        />
        <div style={{ display: "flex", gap: 18, fontSize: 12, color: "var(--muted)" }}>
          <span><span style={{ display: "inline-block", width: 14, borderTop: "2px solid var(--blue)", verticalAlign: "middle" }} /> macro-F1</span>
          <span><span style={{ display: "inline-block", width: 14, borderTop: "2px dashed var(--muted)", verticalAlign: "middle" }} /> mAP</span>
        </div>
      </Card>

      <div className="grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <Card title="클래스별 F1 (실데이터 모델)" sub="Loc(0.72)·Scratch(0.78) 최저 — 그대로 노출">
          <HBars rows={m.per_class.map((p) => ({ label: p.cls, value: p.f1 }))} max={1} fmt={(v) => v.toFixed(3)}
            colorFn={(r) => (r.value < 0.8 ? "var(--red)" : r.value < 0.9 ? "var(--amber)" : "var(--blue)")} />
        </Card>
        <Card title="정직성 노트">
          <div style={{ display: "flex", flexDirection: "column", gap: 10, fontSize: 13.5, lineHeight: 1.5 }}>
            <div>• A(증강)·C(자기지도)의 <b>negative result</b>를 보정 없이 그대로 보고.</div>
            <div>• 합성→실제 <span style={{ color: "var(--red)" }}>전이 실패(0.36)</span>가 핵심 발견 — 합성 단독 학습의 한계.</div>
            <div>• 실데이터 lot-split 직접 학습이 <span style={{ color: "var(--green)" }}>0.86</span>으로 회복, SSL은 +0.01 수준.</div>
            <div>• 평가는 <b>leak-free</b>(lot 그룹 분할, 임계 val-only).</div>
          </div>
        </Card>
      </div>

      <Card title="혼동행렬 — 합성모델로 실제맵 예측" sub="행=실제 라벨, 열=예측 · 셀 hover로 값 확인 · Random 열에 대량 오분류(전이 실패)">
        <Confusion labels={m.confusion_synth_on_real.labels} matrix={m.confusion_synth_on_real.matrix} />
      </Card>
    </div>
  );
}
