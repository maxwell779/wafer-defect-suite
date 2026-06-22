import React, { useState } from "react";
import { Card, WM_CLASSES } from "../ui.jsx";

const url = (f) => "/assets/" + f;
const DET_COLORS = { Center: "#dc2626", Donut: "#d97706", "Edge-Loc": "#2563eb", "Edge-Ring": "#7c3aed", Loc: "#16a34a", Scratch: "#db2777" };
const DET = {
  0: [{ x: .40, y: .42, w: .20, h: .22, cls: "Edge-Ring", conf: .91 }, { x: .07, y: .07, w: .13, h: .13, cls: "Center", conf: .44 }, { x: .05, y: .55, w: .10, h: .10, cls: "Loc", conf: .33 }],
  1: [{ x: .45, y: .48, w: .12, h: .13, cls: "Center", conf: .95 }, { x: .86, y: .10, w: .10, h: .16, cls: "Edge-Loc", conf: .52 }, { x: .86, y: .62, w: .10, h: .14, cls: "Edge-Loc", conf: .41 }],
  2: [{ x: .40, y: .46, w: .07, h: .09, cls: "Loc", conf: .74 }, { x: .46, y: .52, w: .05, h: .07, cls: "Loc", conf: .58 }, { x: .20, y: .30, w: .06, h: .06, cls: "Scratch", conf: .29 }],
  3: [{ x: .46, y: .05, w: .16, h: .90, cls: "Scratch", conf: .88 }, { x: .30, y: .40, w: .12, h: .16, cls: "Edge-Loc", conf: .46 }],
};
const CAM_CLASSES = WM_CLASSES; // 8 클래스 Grad-CAM

export default function Stage3Detection() {
  const [view, setView] = useState("loc"); // loc(B,메인) / det(A,부록)
  const [imgIdx, setImgIdx] = useState(0);
  const [conf, setConf] = useState(0.3);

  const boxes = DET[imgIdx].filter((b) => b.conf >= conf);
  const counts = Object.keys(DET_COLORS).map((c) => [c, DET[imgIdx].filter((b) => b.cls === c && b.conf >= conf).length]);

  return (
    <div className="grid">
      <div><h1 className="page">Stage 3 — Defect Localization & Detection</h1>
        <div className="sub">결함이 <b>어디</b>에 있는지 — 실데이터 위치탐지(메인) + 합성 검출(부록)</div></div>

      <div style={{ display: "flex", gap: 8 }}>
        <button className={"btn" + (view === "loc" ? " on" : "")} onClick={() => setView("loc")}>위치탐지 (Grad-CAM · 실데이터) ★</button>
        <button className={"btn" + (view === "det" ? " on" : "")} onClick={() => setView("det")}>객체검출 (YOLO · ELLIMAC 합성)</button>
      </div>

      {view === "loc" && (
        <Card title="실데이터 결함 위치탐지 (Grad-CAM)" sub="Stage2 실데이터 모델의 Grad-CAM — WM-811K 실제 맵에서 결함 근거 영역(빨강). 합성 무관.">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
            {CAM_CLASSES.map((c) => (
              <div key={c} style={{ textAlign: "center" }}>
                <img src={url("cam_" + c + ".png")} style={{ width: "100%", borderRadius: 8, border: "1px solid var(--border)" }} />
                <div style={{ fontSize: 13, fontWeight: 600, marginTop: 4 }}>{c}</div>
              </div>
            ))}
          </div>
          <div className="note" style={{ marginTop: 14 }}>실제 WM-811K 맵 + Grad-CAM 오버레이(빨강=결함 근거). Center=중앙, Donut=링, Edge=가장자리 등 위치가 정확히 잡힘 → "어느 부분이 결함인지"를 <b>실데이터</b>로 제시(합성 미사용).</div>
        </Card>
      )}

      {view === "det" && (
        <div className="grid" style={{ gridTemplateColumns: "1fr 280px" }}>
          <Card title="검출 결과" sub={`ellimac_0${imgIdx} · bestV2 YOLO`}>
            <div style={{ position: "relative" }}>
              <img src={url("ellimac_0" + imgIdx + ".jpg")} style={{ width: "100%", borderRadius: 8 }} />
              <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
                {boxes.map((b, i) => (
                  <g key={i}>
                    <rect x={b.x * 100} y={b.y * 100} width={b.w * 100} height={b.h * 100} fill="none" stroke={DET_COLORS[b.cls]} strokeWidth="0.6" />
                    <rect x={b.x * 100} y={b.y * 100 - 4} width={b.cls.length * 2.2 + 8} height="4" fill={DET_COLORS[b.cls]} />
                    <text x={b.x * 100 + 0.6} y={b.y * 100 - 1} fontSize="2.6" fill="#fff" className="mono">{b.cls} {b.conf.toFixed(2)}</text>
                  </g>
                ))}
              </svg>
            </div>
            <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 10 }}>
              <span className="sub" style={{ margin: 0 }}>confidence ≥ {conf.toFixed(2)}</span>
              <input type="range" min={0.1} max={0.95} step={0.05} value={conf} onChange={(e) => setConf(+e.target.value)} style={{ flex: 1 }} />
              <span className="mono" style={{ fontWeight: 700 }}>{boxes.length} 검출</span>
            </div>
          </Card>
          <div className="grid">
            <Card title="이미지 선택">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                {[0, 1, 2, 3].map((i) => <img key={i} src={url("ellimac_0" + i + ".jpg")} onClick={() => setImgIdx(i)}
                  style={{ width: "100%", borderRadius: 6, cursor: "pointer", border: i === imgIdx ? "2px solid var(--blue)" : "1px solid var(--border)" }} />)}
              </div>
            </Card>
            <Card title="클래스 범례 & 집계">
              {counts.map(([c, n]) => (
                <div key={c} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, marginBottom: 6 }}>
                  <span style={{ width: 12, height: 12, borderRadius: 3, background: DET_COLORS[c] }} />
                  <span style={{ flex: 1 }}>{c}</span><span className="mono">{n}</span>
                </div>
              ))}
            </Card>
          </div>
        </div>
      )}

      {view === "det" && <div className="note warn">⚠ ELLIMAC은 Roboflow <b>합성</b> 데이터 — 폴리곤→bbox 정제 + cls6 18줄 제거 후 bestV2 test <b>mAP@0.5 0.739</b>. 합성이라 실전 일반화 보장 없음(스킬 데모·부록).</div>}
    </div>
  );
}
