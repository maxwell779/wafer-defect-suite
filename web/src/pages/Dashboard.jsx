import React, { useState } from "react";
import { Card, downloadCSV } from "../ui.jsx";
import { useI18n } from "../i18n.jsx";
import runs from "../appdata/process_runs.json";
import wmaps from "../appdata/wafermaps.json";
import metrics from "../appdata/stage2_metrics.json";

// 웨이퍼 패턴 → 공정 연관(도메인 규칙)
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
const DISP_BADGE = { 격리: "b-fail", 재검사: "b-warn", 보정: "b-tag", 완료: "b-ok" };

export default function Dashboard({ go }) {
  const total = runs.length;
  const realF1 = Math.max(...metrics.comparison.filter((c) => c.name.includes("실데이터") || c.name.includes("보정")).map((c) => c.macro_f1));
  const realMaps = wmaps.filter((m) => m.source === "real").length;
  const defects = runs.filter((r) => r.defect === 1).sort((a, b) => b.anomaly_score - a.anomaly_score);

  const { t } = useI18n();
  const [disp, setDisp] = useState({});          // wafer_id → 처리상태
  const [log, setLog] = useState([]);            // 처리 이력
  const [sel, setSel] = useState(defects[0].wafer_id);
  const [pattern, setPattern] = useState("Center");
  const [qFilter, setQFilter] = useState("ALL"); // 상태 필터
  const [q, setQ] = useState("");                // wafer 검색
  const w = runs.find((r) => r.wafer_id === sel);

  const queue = defects.filter((d) =>
    (qFilter === "ALL" || (qFilter === "미처리" ? !disp[d.wafer_id] : disp[d.wafer_id] === qFilter)) &&
    (!q || String(d.wafer_id).includes(q)));

  function exportCSV() {
    const rows = [["wafer_id", "process_step", "anomaly_score", "status"],
      ...defects.map((d) => [d.wafer_id, d.process_step, d.anomaly_score, disp[d.wafer_id] || "미처리"])];
    downloadCSV("defect_queue.csv", rows);
  }
  function exportLogCSV() {
    downloadCSV("action_log.csv", [["log"], ...log.map((l) => [l])]);
  }

  const devs = Object.entries(FEAT).map(([k, [label, mean]]) => {
    const d = w[k] - mean; return { label, k, val: w[k], dev: d, pct: (d / mean) * 100 };
  }).filter((d) => Math.abs(d.pct) > 8);
  const [pName, pCause] = PATTERN[pattern];

  const pending = defects.filter((d) => d.anomaly_score >= 0.5 && !disp[d.wafer_id]).length;
  const handled = Object.keys(disp).length;
  const yieldPct = (((total - defects.length) / total) * 100).toFixed(1);

  function act(status, msg) {
    setDisp((p) => ({ ...p, [sel]: status }));
    const t = new Date().toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    setLog((L) => [`[${t}] Wafer ${sel} · ${msg}`, ...L].slice(0, 8));
  }

  const stages = [
    { id: "stage1", n: "01", t: "공정 모니터링", q: "왜", metric: "PR-AUC 0.81", c: "var(--blue)" },
    { id: "stage2", n: "02", t: "웨이퍼 패턴", q: "무엇", metric: realF1.toFixed(3) + " F1", c: "var(--violet)" },
    { id: "stage3", n: "03", t: "결함 위치/검출", q: "어디", metric: "mAP 0.753", c: "var(--green)" },
  ];

  const kpis = [
    ["검사 완료", `${total}`, "공정 run (lot)", "var(--blue)", "stage1"],
    ["결함 검출", `${defects.length}건`, `${((defects.length / total) * 100).toFixed(2)}% 결함율`, "var(--red)", "stage1"],
    ["격리 대기", `${pending}건`, `고위험 미처리 (처리완료 ${handled})`, "var(--amber)", "stage3"],
    ["양품률", `${yieldPct}%`, `정상 ${total - defects.length} / ${total}`, "var(--green)", "stage2"],
  ];

  return (
    <div className="grid">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 12 }}>
        <div>
          <h1 className="page">{t("통합 검사 콘솔")}</h1>
          <div className="sub" style={{ marginBottom: 0 }}>{t("공정(왜) → 패턴(무엇) → 위치(어디) 분석을 종합해 결함 lot을 판정·조치하는 작업자 화면")}</div>
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button className="btn" onClick={() => go("stage1")}>Stage 1 공정</button>
          <button className="btn" onClick={() => go("stage2")}>Stage 2 패턴</button>
          <button className="btn" onClick={() => go("stage3")}>Stage 3 위치</button>
          <button className="btn" onClick={() => go("experiments")}>Experiments</button>
        </div>
      </div>

      {/* 운영 KPI (클릭 시 해당 단계로 이동) */}
      <div className="grid" style={{ gridTemplateColumns: "repeat(4,1fr)" }}>
        {kpis.map(([label, v, d, c, to], i) => (
          <div key={i} className="card kpicard clickable" style={{ "--accent": c }} onClick={() => go(to)} role="button" tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && go(to)}>
            <div className="klabel">{t(label)}</div>
            <div className="kpi" style={{ color: c }}>{v}</div>
            <div className="sub" style={{ margin: 0, fontSize: 13.5 }}>{d}</div>
          </div>
        ))}
      </div>

      {/* 단계 바로가기 카드 */}
      <div className="grid" style={{ gridTemplateColumns: "repeat(3,1fr)" }}>
        {stages.map((s) => (
          <div key={s.id} className="card clickable" style={{ borderTop: `4px solid ${s.c}` }} onClick={() => go(s.id)}>
            <div className="klabel" style={{ color: s.c }}>STAGE {s.n} · {s.q} · 클릭 →</div>
            <h3 style={{ fontSize: 19, margin: "8px 0 0" }}>{s.t}</h3>
            <div className="mono" style={{ fontWeight: 800, fontSize: 19, color: s.c, marginTop: 10 }}>{s.metric}</div>
          </div>
        ))}
      </div>

      {/* 통합 의사결정 콘솔 (Stage4 통합) */}
      <div className="grid" style={{ gridTemplateColumns: "320px 1fr" }}>
        <Card title={t("결함 큐")} sub="이상점수순 · 행 클릭 → 우측 리포트">
          <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
            <input type="text" placeholder="Wafer 검색" value={q} onChange={(e) => setQ(e.target.value)} style={{ flex: 1 }} aria-label="Wafer 검색" />
            <select value={qFilter} onChange={(e) => setQFilter(e.target.value)} aria-label="상태 필터">
              <option value="ALL">{t("상태 전체")}</option><option value="미처리">{t("미처리")}</option>
              <option value="격리">격리</option><option value="재검사">재검사</option><option value="보정">보정</option><option value="완료">완료</option>
            </select>
          </div>
          <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
            <button className="btn" style={{ flex: 1 }} onClick={exportCSV} aria-label="결함 큐 CSV 내보내기">⬇ {t("CSV 내보내기")}</button>
            <button className="btn" style={{ flex: 1 }} onClick={() => window.print()} aria-label="PDF 인쇄">🖨 {t("PDF/인쇄")}</button>
          </div>
          <div style={{ maxHeight: 360, overflow: "auto" }}>
            <table>
              <thead><tr><th>Wafer</th><th>점수</th><th>상태</th></tr></thead>
              <tbody>
                {queue.map((d) => (
                  <tr key={d.wafer_id} className={d.wafer_id === sel ? "sel" : ""} onClick={() => setSel(d.wafer_id)}>
                    <td className="mono">{d.wafer_id}</td>
                    <td className="mono" style={{ color: d.anomaly_score >= 0.5 ? "var(--red)" : "inherit" }}>{d.anomaly_score.toFixed(2)}</td>
                    <td>{disp[d.wafer_id]
                      ? <span className={"badge " + DISP_BADGE[disp[d.wafer_id]]}>{disp[d.wafer_id]}</span>
                      : <span className="sub" style={{ margin: 0, fontSize: 12 }}>{t("미처리")}</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {queue.length === 0 && <div className="empty">조건에 맞는 wafer가 없습니다.</div>}
          </div>
          <div className="sub" style={{ margin: "8px 0 4px" }}>웨이퍼 패턴 (Stage 2 판정)</div>
          <select value={pattern} onChange={(e) => setPattern(e.target.value)} style={{ width: "100%" }}>
            {Object.keys(PATTERN).map((p) => <option key={p}>{p}</option>)}
          </select>
        </Card>

        <Card title={`📋 Wafer ${w.wafer_id} 통합 분석 리포트`} sub={`공정 ${w.process_step} · 이상점수 ${w.anomaly_score.toFixed(2)} · 상태 ${disp[sel] || "미처리"}`}>
          <div style={{ display: "flex", flexDirection: "column", gap: 13 }}>
            <div><b style={{ color: "var(--blue)" }}>1. 공정 (왜)</b>
              <ul style={{ margin: "4px 0" }}>
                {devs.length ? devs.map((d) => (
                  <li key={d.k} style={{ color: "var(--red)" }}>{d.label} {d.val.toFixed(1)} — 정상 대비 {d.dev > 0 ? "높음" : "낮음"} ({d.pct > 0 ? "+" : ""}{d.pct.toFixed(0)}%) → {d.dev > 0 ? "하향" : "상향"} 권장</li>
                )) : <li>공정 파라미터 정상범위</li>}
              </ul></div>
            <div><b style={{ color: "var(--violet)" }}>2. 웨이퍼 패턴 (무엇)</b>
              <div style={{ margin: "4px 0" }}>판정 <b>{pattern}</b> ({pName}) — {pCause}</div></div>
            <div><b style={{ color: "var(--green)" }}>3. 위치 (어디)</b>
              <div style={{ margin: "4px 0" }}>맵 {pName} 영역 집중 (Grad-CAM)</div></div>
            <div className="note warn"><b>4. 종합 추정 &amp; 조치</b><br />
              · 추정: <b>{pattern}</b> 패턴 + 공정 편차({devs.map((d) => d.label).join("·") || "정상"}) 연계<br />
              · 권고: {devs[0] ? `${devs[0].label} ${devs[0].dev > 0 ? "하향" : "상향"} 보정` : "모니터링 유지"}; 해당 lot 격리·재검사
            </div>
            <div>
              <div className="klabel" style={{ marginBottom: 8 }}>조치 실행</div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <button className="btn" style={{ borderColor: "var(--red)", color: "var(--red)" }} onClick={() => act("격리", "lot 격리 지시")}>🚫 lot 격리</button>
                <button className="btn" style={{ borderColor: "var(--amber)", color: "var(--amber)" }} onClick={() => act("재검사", "재검사 요청")}>🔁 재검사</button>
                <button className="btn" onClick={() => act("보정", `${devs[0] ? devs[0].label + " 보정" : "공정 보정"} 티켓 생성`)}>🛠 공정 보정</button>
                <button className="btn" style={{ borderColor: "var(--green)", color: "var(--green)" }} onClick={() => act("완료", "검토 완료 처리")}>✓ 완료</button>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* 처리 이력 + 파이프라인 설명 */}
      <div className="grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <Card title={t("처리 이력")} sub={`이번 세션 조치 ${log.length}건`}>
          {log.length === 0
            ? <div className="empty">아직 조치 없음 — 우측 리포트에서 격리/재검사/보정/완료를 실행하면 기록됩니다.</div>
            : <>
                <button className="btn" style={{ marginBottom: 10 }} onClick={exportLogCSV}>⬇ {t("CSV 내보내기")}</button>
                <div style={{ display: "flex", flexDirection: "column", gap: 6, fontSize: 13.5 }}>
                  {log.map((l, i) => <div key={i} className="mono" style={{ color: i === 0 ? "var(--ink)" : "var(--muted)" }}>{l}</div>)}
                </div>
              </>}
        </Card>
        <Card title="파이프라인 동작" sub="제조 검사 흐름 그대로 — 왜 → 무엇 → 어디 → 통합">
          <div style={{ display: "flex", flexDirection: "column", gap: 9, fontSize: 13.5, lineHeight: 1.55 }}>
            <div><b style={{ color: "var(--blue)" }}>1 공정</b> Meruva 센서 6종 비지도 이상탐지 → 의심 lot 플래그 (PR-AUC 0.31→0.81)</div>
            <div><b style={{ color: "var(--violet)" }}>2 패턴</b> WM-811K 8클래스 분류, 합성→실 전이실패(0.36) 규명 후 앙상블 0.935</div>
            <div><b style={{ color: "var(--green)" }}>3 위치</b> Grad-CAM 위치 히트맵 + ELLIMAC 칩표면 YOLO11m 검출(0.753)</div>
            <div><b style={{ color: "var(--amber)" }}>통합</b> 위 결과를 합쳐 이 화면에서 원인 추정·조치 결정</div>
          </div>
          <div className="note" style={{ marginTop: 12 }}>핵심 — "합성만으론 실전에서 무너진다(0.99→0.36)"를 규명하고 실데이터+구조개선으로 0.935까지. 효과 없던 방법도 Experiments에 공개.</div>
        </Card>
      </div>
    </div>
  );
}
