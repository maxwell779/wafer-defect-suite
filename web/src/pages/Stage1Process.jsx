import React, { useMemo, useState, useEffect } from "react";
import { Card, HBars, ControlChart, Scatter, STATUS_BADGE } from "../ui.jsx";
import { stage1Score } from "../api.js";
import runs from "../appdata/process_runs.json";
import s1 from "../appdata/stage1_results.json";

const FEATS = [
  ["temperature_c", "온도", "°C"], ["pressure_torr", "압력", "Torr"],
  ["gas_flow_sccm", "가스유량", "sccm"], ["etch_rate_nm_min", "식각률", "nm/min"],
  ["voltage_v", "전압", "V"], ["current_ma", "전류", "mA"],
];

export default function Stage1Process({ live }) {
  const normal = runs.filter((r) => r.defect === 0);
  const stats = useMemo(() => {
    const s = {};
    for (const [k] of FEATS) {
      const v = normal.map((r) => r[k]);
      const m = v.reduce((a, b) => a + b, 0) / v.length;
      const sd = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length);
      s[k] = { m, sd };
    }
    return s;
  }, []);
  const maxScore = useMemo(() =>
    Math.max(...runs.map((r) => FEATS.reduce((a, [k]) => a + ((r[k] - stats[k].m) / stats[k].sd) ** 2, 0))), []);

  const [sort, setSort] = useState("anomaly_score");
  const [stat, setStat] = useState("ALL");
  const [step, setStep] = useState("ALL");
  const [q, setQ] = useState("");
  const [selId, setSelId] = useState(null);
  const [param, setParam] = useState("pressure_torr");
  const [sliders, setSliders] = useState(null);

  const steps = [...new Set(runs.map((r) => r.process_step))];
  const rows = runs
    .filter((r) => (stat === "ALL" || r.status === stat) && (step === "ALL" || r.process_step === step) && (!q || String(r.wafer_id).includes(q)))
    .sort((a, b) => (typeof a[sort] === "number" ? b[sort] - a[sort] : String(a[sort]).localeCompare(b[sort])));

  const sel = runs.find((r) => r.wafer_id === selId);
  function pick(r) {
    setSelId(r.wafer_id);
    setSliders(Object.fromEntries(FEATS.map(([k]) => [k, r[k]])));
  }
  const liveScore = sliders
    ? FEATS.reduce((a, [k]) => a + ((sliders[k] - stats[k].m) / stats[k].sd) ** 2, 0) / maxScore
    : 0;

  // LIVE: 슬라이더 변경 → 백엔드 LOF 실시간 채점 (디바운스, 폴백=휴리스틱)
  const [api, setApi] = useState(null);
  useEffect(() => {
    if (!live || !sliders) { setApi(null); return; }
    const t = setTimeout(() => stage1Score(sliders).then(setApi).catch(() => setApi(null)), 250);
    return () => clearTimeout(t);
  }, [live, sliders]);
  const effScore = live && api ? api.anomaly_score : liveScore;

  return (
    <div className="grid">
      <div><h1 className="page">Stage 1 — Process Monitor</h1>
        <div className="sub">공정 센서 6종 비지도 이상탐지 · {runs.length} run 중 {runs.filter(r=>r.defect).length} 결함(소표본)</div></div>

      <div className="grid" style={{ gridTemplateColumns: "1.3fr 1fr" }}>
        {/* Run 테이블 */}
        <Card title="공정 Run 테이블" sub="행 클릭 → 우측 추천 패널. 헤더 클릭 정렬">
          <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
            <input type="text" placeholder="Wafer ID 검색" value={q} onChange={(e) => setQ(e.target.value)} style={{ flex: 1 }} />
            <select value={step} onChange={(e) => setStep(e.target.value)}><option value="ALL">전체 공정</option>{steps.map((s) => <option key={s}>{s}</option>)}</select>
            <select value={stat} onChange={(e) => setStat(e.target.value)}><option value="ALL">전체 상태</option><option>FAIL</option><option>WARN</option><option>OK</option></select>
          </div>
          <div style={{ maxHeight: 360, overflow: "auto" }}>
            <table>
              <thead><tr>
                {[["wafer_id", "Wafer"], ["process_step", "공정"], ["temperature_c", "온도"], ["pressure_torr", "압력"], ["etch_rate_nm_min", "식각률"], ["anomaly_score", "이상점수"], ["status", "상태"]].map(([k, l]) =>
                  <th key={k} onClick={() => setSort(k)}>{l}{sort === k ? " ▼" : ""}</th>)}
              </tr></thead>
              <tbody>
                {rows.slice(0, 120).map((r) => (
                  <tr key={r.wafer_id} className={r.wafer_id === selId ? "sel" : ""} onClick={() => pick(r)}>
                    <td className="mono">{r.wafer_id}</td><td>{r.process_step}</td>
                    <td className="mono">{r.temperature_c}</td><td className="mono">{r.pressure_torr}</td><td className="mono">{r.etch_rate_nm_min}</td>
                    <td className="mono" style={{ color: r.anomaly_score > 0.5 ? "var(--red)" : "inherit" }}>{r.anomaly_score.toFixed(3)}</td>
                    <td><span className={"badge " + STATUS_BADGE[r.status]}>{r.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {/* 추천 패널 */}
        <Card title="파라미터 조정 추천" sub={sel ? `Wafer ${sel.wafer_id} · 정상 평균 대비 편차` : "좌측에서 run을 선택하세요"}>
          {sel && sliders && (
            <div>
              <div className="bar-track" style={{ height: 20, marginBottom: 4 }}>
                <div className="bar-fill" style={{ width: Math.min(effScore * 100, 100) + "%", background: effScore > 0.5 ? "var(--red)" : "var(--amber)" }} />
              </div>
              <div style={{ textAlign: "right", marginBottom: 10 }} className="mono">
                {live && api ? "실시간 LOF 점수" : "재계산 이상점수(휴리스틱)"} {effScore.toFixed(3)}
              </div>
              {FEATS.map(([k, label, unit]) => {
                const z = (sliders[k] - stats[k].m) / stats[k].sd;
                const mn = stats[k].m, lo = mn - 4 * stats[k].sd, hi = mn + 4 * stats[k].sd;
                const rec = Math.abs(z) > 1.2;
                return (
                  <div key={k} style={{ marginBottom: 8 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5 }}>
                      <span>{label} <span style={{ color: "var(--muted)" }}>{unit}</span></span>
                      <span className="mono" style={{ color: rec ? "var(--red)" : "inherit" }}>{sliders[k].toFixed(1)}</span>
                    </div>
                    <input type="range" min={lo} max={hi} step={(hi - lo) / 200} value={sliders[k]}
                      onChange={(e) => setSliders({ ...sliders, [k]: +e.target.value })} />
                    {rec && <div style={{ fontSize: 11.5, color: "var(--red)" }}>
                      {z > 0 ? "▲" : "▼"} 정상 평균 {mn.toFixed(0)} 대비 {z > 0 ? "높음" : "낮음"} → {(mn - sliders[k] > 0 ? "+" : "") + (mn - sliders[k]).toFixed(0)} {z > 0 ? "하향" : "상향"} 권장
                    </div>}
                  </div>
                );
              })}
              <button className="btn" onClick={() => pick(sel)} style={{ marginTop: 6 }}>원래 값으로 초기화</button>
            </div>
          )}
        </Card>
      </div>

      {/* 관리도 */}
      <Card title="관리도 (Control Chart)" sub="파라미터별 시계열 + 관리한계선 UCL/LCL = mean±3σ · 한계 벗어난 run 빨강">
        <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
          {FEATS.map(([k, l]) => <button key={k} className={"btn" + (param === k ? " on" : "")} onClick={() => setParam(k)}>{l}</button>)}
        </div>
        <ControlChart values={runs.slice(0, 120).map((r) => r[param])} />
      </Card>

      <div className="grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <Card title="수율–파라미터 관계" sub="x=압력 vs 이상점수 (빨강=결함)">
          <Scatter points={runs.map((r) => ({ x: r.pressure_torr, y: r.anomaly_score, defect: r.defect === 1 }))} xlab="압력(Torr)" ylab="이상점수" />
        </Card>
        <Card title="변수 중요도 (수율 영향, z-gap)" sub="결함 시 +상승 / −하강 · |값| 클수록 위험신호">
          <HBars
            rows={Object.entries(s1.feature_contrib).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
              .map(([k, v]) => ({ label: { temperature_c: "온도", pressure_torr: "압력", gas_flow_sccm: "가스유량", etch_rate_nm_min: "식각률", voltage_v: "전압", current_ma: "전류" }[k], value: v }))}
            max={2.5} fmt={(v) => (v > 0 ? "+" : "") + v.toFixed(2)}
            colorFn={(r) => (r.value > 0 ? "var(--red)" : "var(--blue)")} />
          <div className="sub" style={{ marginTop: 10 }}>압력↓·식각률↑·온도↑ 가 결함과 가장 강하게 연관</div>
        </Card>
      </div>

      {/* ★ 피처 엔지니어링 성능 향상 (핵심 성과) */}
      <Card title="★ 피처 엔지니어링 성능 향상 (PR-AUC)" sub="도메인 메커니즘 인코딩으로 0.31 → 0.81 (leak-free, reps=40 CI)">
        <HBars
          rows={s1.progression.map((p) => ({ label: p.stage, value: p.pr_auc }))}
          max={1} fmt={(v) => v.toFixed(3)}
          colorFn={(r) => (r.value >= 0.78 ? "var(--green)" : r.value >= 0.44 ? "var(--blue)" : "#94a3b8")} />
        <div className="note" style={{ marginTop: 12 }}>
          <b style={{ color: "var(--green)" }}>★ {s1.feature_best.base_best} → {s1.feature_best.hybrid_best}</b> — 결함 7건이 전부 <b>pressure≤-2σ + 고온/식각</b> 조합 → "결함=센서 비정상 조합" 메커니즘을 표적 피처로 인코딩(0.78), 지도/비지도 하이브리드(LGB+Maha피처)로 reps40 <b>0.81</b>. 무차별 전수조합은 차원의 저주로 붕괴(0.07~0.22).
        </div>
        <div className="note warn" style={{ marginTop: 10 }}>
          <b>정직한 천장 규명(reps=40 CI±{s1.ceiling.ci})</b> — {s1.ceiling.note}
        </div>
      </Card>

      {/* ML vs DL + 신규기법 negative */}
      <div className="grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <Card title="ML vs DL (base 피처, 비지도)" sub="소표본(결함 7)에서 PR-AUC — 라벨은 평가에만(leak-free)">
          <table>
            <thead><tr><th>모델</th><th>PR-AUC</th><th>ROC</th><th>R@100</th></tr></thead>
            <tbody>
              {[...s1.comparison].sort((a, b) => b.pr_auc - a.pr_auc).map((m, i) => (
                <tr key={m.model} style={{ background: i === 0 ? "#f0fdf4" : "" }}>
                  <td style={{ fontWeight: i === 0 ? 700 : 400 }}>{m.model}{m.model.includes("DL") && <span className="badge b-tag" style={{ marginLeft: 6 }}>DL</span>}</td>
                  <td className="mono">{m.pr_auc}</td><td className="mono">{m.roc_auc}</td><td className="mono">{m.recall_at_100}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="sub" style={{ marginTop: 10 }}>고전 ML(~0.29) &gt; DL(AE 0.215, 불안정 ±0.095). 양성 7건엔 딥러닝 불필요.</div>
        </Card>
        <Card title="신규 기법 탐색 (정직한 negative)" sub="천장(~0.81) 못 넘음을 reps=40으로 규명 — '복잡 ≠ 좋음'">
          <table>
            <thead><tr><th>방법</th><th>PR-AUC</th><th>판정</th></tr></thead>
            <tbody>
              {s1.negatives.map((n) => (
                <tr key={n.method}>
                  <td style={{ fontSize: 12.5 }}>{n.method}</td>
                  <td className="mono">{n.pr_auc}</td>
                  <td><span className={"badge " + (n.pr_auc >= 0.79 ? "b-warn" : "b-fail")} style={{ fontSize: 10.5 }}>{n.verdict}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </div>
    </div>
  );
}
