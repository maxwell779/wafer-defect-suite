import React, { useState } from "react";
import { Card } from "../ui.jsx";
import runs from "../appdata/process_runs.json";

// 웨이퍼 패턴 → 공정 연관(도메인 규칙) — Stage4 통합 추론
const PATTERN = {
  Center: ["중앙부 집중", "척/가스 분포 비대칭·중앙 압력 편차"],
  Donut: ["환형", "회전 불균일·온도 링 편차"],
  "Edge-Ring": ["가장자리 링", "에지 클램프/식각 균일도"],
  "Edge-Loc": ["가장자리 국소", "핸들링/엣지 결함"],
  Loc: ["국소", "입자 낙하·국부 오염"],
  Scratch: ["선형", "기계적 긁힘(핸들링)"],
  Random: ["산발", "전반적 공정 불안정/오염"],
  "Near-full": ["전면", "심각한 레시피 이탈"],
};
const FEAT = { temperature_c: ["온도", 450], pressure_torr: ["압력", 760], etch_rate_nm_min: ["식각률", 95] };

export default function Stage4Report() {
  const defects = runs.filter((r) => r.defect === 1).sort((a, b) => b.anomaly_score - a.anomaly_score);
  const [sel, setSel] = useState(defects[0].wafer_id);
  const [pattern, setPattern] = useState("Center");
  const w = runs.find((r) => r.wafer_id === sel);

  // 공정 편차(정상 대비) 자동 추출
  const devs = Object.entries(FEAT).map(([k, [label, mean]]) => {
    const d = w[k] - mean;
    return { label, k, val: w[k], dev: d, pct: (d / mean) * 100 };
  }).filter((d) => Math.abs(d.pct) > 8);
  const [pName, pCause] = PATTERN[pattern];

  return (
    <div className="grid">
      <div><h1 className="page">Stage 4 — 통합 원인추론 리포트</h1>
        <div className="sub">공정(왜) + 웨이퍼 패턴(무엇) + 위치(어디) → 추정 원인·조치 (3-스테이지 종합)</div></div>

      <div className="grid" style={{ gridTemplateColumns: "300px 1fr" }}>
        <Card title="대상 선택">
          <div className="sub" style={{ margin: "0 0 6px" }}>결함 웨이퍼 (이상점수순)</div>
          <select value={sel} onChange={(e) => setSel(+e.target.value)} style={{ width: "100%", marginBottom: 12 }}>
            {defects.map((d) => <option key={d.wafer_id} value={d.wafer_id}>Wafer {d.wafer_id} · {d.process_step} · {d.anomaly_score.toFixed(2)}</option>)}
          </select>
          <div className="sub" style={{ margin: "0 0 6px" }}>웨이퍼 패턴 (Stage2 판정)</div>
          <select value={pattern} onChange={(e) => setPattern(e.target.value)} style={{ width: "100%" }}>
            {Object.keys(PATTERN).map((p) => <option key={p}>{p}</option>)}
          </select>
          <div className="note" style={{ marginTop: 12, fontSize: 12 }}>실제 운영 시: Stage1 이상점수·Stage2 패턴·Stage3 위치가 자동 연결됩니다. (LLM 연결 시 자연어 리포트)</div>
        </Card>

        <Card title={`📋 Wafer ${w.wafer_id} 결함 분석 리포트`} sub={`공정단계 ${w.process_step} · 이상점수 ${w.anomaly_score.toFixed(2)}`}>
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div>
              <b style={{ color: "var(--blue)" }}>1. 공정 (왜)</b>
              <ul style={{ margin: "4px 0" }}>
                {devs.length ? devs.map((d) => (
                  <li key={d.k} style={{ color: "var(--red)" }}>{d.label} {d.val.toFixed(1)} — 정상 대비 {d.dev > 0 ? "높음" : "낮음"} ({d.pct > 0 ? "+" : ""}{d.pct.toFixed(0)}%) → {d.dev > 0 ? "하향" : "상향"} 권장</li>
                )) : <li>공정 파라미터 정상범위</li>}
              </ul>
            </div>
            <div><b style={{ color: "var(--violet)" }}>2. 웨이퍼 패턴 (무엇)</b>
              <div style={{ margin: "4px 0" }}>판정: <b>{pattern}</b> ({pName}) — {pCause}</div></div>
            <div><b style={{ color: "var(--green)" }}>3. 위치 (어디)</b>
              <div style={{ margin: "4px 0" }}>맵 {pName} 영역 (Grad-CAM 빨강 집중)</div></div>
            <div className="note warn">
              <b>4. 종합 추정 원인 &amp; 조치</b><br />
              · 추정: <b>{pattern}</b> 패턴({pCause}) + 공정 편차({devs.map((d) => d.label).join("·") || "정상"}) 연계<br />
              · 조치: {devs[0] ? `${devs[0].label} ${devs[0].dev > 0 ? "하향" : "상향"} 보정` : "공정 모니터링 유지"}; 해당 lot 격리·재검사
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
