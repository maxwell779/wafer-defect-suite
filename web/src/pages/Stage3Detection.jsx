import React, { useState, useEffect } from "react";
import { Card, WM_CLASSES } from "../ui.jsx";
import { stage3DetectUpload } from "../api.js";
import DET from "../appdata/ellimac_dets.json";   // 실제 yolo11m 추론 박스(8장)

const url = (f) => "/assets/" + f;
const ei = (i) => "ellimac_" + String(i).padStart(2, "0") + ".jpg";   // 2자리 zero-pad(10장+ 대응)
const DET_COLORS = { Center: "#dc2626", Donut: "#d97706", "Edge-Loc": "#2563eb", "Edge-Ring": "#7c3aed", Loc: "#16a34a", Scratch: "#db2777" };
const DET_IDX = Object.keys(DET).map(Number).sort((a, b) => a - b);  // [0..7]
const CAM_CLASSES = WM_CLASSES; // 8 클래스 Grad-CAM

export default function Stage3Detection({ live, go }) {
  const [view, setView] = useState("det"); // det(ELLIMAC 칩표면, 기본) / loc(Grad-CAM)
  const [imgIdx, setImgIdx] = useState(0);
  const [conf, setConf] = useState(0.3);
  const [liveBoxes, setLiveBoxes] = useState(null);  // 백엔드 실제 추론 결과(이미지별)
  const [liveErr, setLiveErr] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => { setLiveBoxes(null); setLiveErr(""); }, [imgIdx]);  // 이미지 바뀌면 초기화

  // LIVE 활성 상태에서 confidence 바꾸면 해당 임계로 재추론(debounce)
  useEffect(() => {
    if (!live || !liveBoxes) return;
    const tmr = setTimeout(() => {
      setBusy(true);
      stage3DetectUpload(url(ei(imgIdx)), conf)
        .then((r) => setLiveBoxes(r.boxes || []))
        .catch(() => {})
        .finally(() => setBusy(false));
    }, 400);
    return () => clearTimeout(tmr);
  }, [conf]); // eslint-disable-line

  // LIVE: 화면의 그 이미지를 업로드 추론(정확히 같은 이미지 → 박스 정합)
  function runLive() {
    setBusy(true); setLiveErr("");
    stage3DetectUpload(url(ei(imgIdx)), 0.05)
      .then((r) => setLiveBoxes(r.boxes || []))
      .catch(() => setLiveErr("백엔드 추론 실패 — 정적 예시로 폴백"))
      .finally(() => setBusy(false));
  }

  const srcBoxes = liveBoxes ?? (DET[imgIdx] || []);
  const boxes = srcBoxes.filter((b) => b.conf >= conf);
  const counts = Object.keys(DET_COLORS).map((c) => [c, srcBoxes.filter((b) => b.cls === c && b.conf >= conf).length]);

  return (
    <div className="grid">
      <div><h1 className="page">Stage 3 — Defect Localization & Detection</h1>
        <div className="sub">결함이 <b>어디</b>에 있는지 — ELLIMAC 칩표면 객체검출(YOLO) + WM-811K 실데이터 위치탐지(Grad-CAM)</div></div>

      <div style={{ display: "flex", gap: 8 }}>
        <button className={"btn" + (view === "det" ? " on" : "")} onClick={() => setView("det")}>객체검출 (YOLO11m · ELLIMAC 칩표면) ★</button>
        <button className={"btn" + (view === "loc" ? " on" : "")} onClick={() => setView("loc")}>위치탐지 (Grad-CAM · WM-811K)</button>
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
          <Card title="검출 결과" sub={`ellimac_${String(imgIdx).padStart(2,"0")} · YOLO11m · ${liveBoxes ? "⚡LIVE 재추론" : "추론 결과(저장본)"}`}>
            <div style={{ position: "relative" }}>
              <img src={url(ei(imgIdx))} style={{ width: "100%", borderRadius: 8 }} />
              <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
                {boxes.map((b, i) => {
                  const col = DET_COLORS[b.cls] || "#0ea5e9";
                  return (
                  <g key={i}>
                    <rect x={b.x * 100} y={b.y * 100} width={b.w * 100} height={b.h * 100} fill="none" stroke={col} strokeWidth="0.6" />
                    <rect x={b.x * 100} y={b.y * 100 - 4} width={b.cls.length * 2.2 + 8} height="4" fill={col} />
                    <text x={b.x * 100 + 0.6} y={b.y * 100 - 1} fontSize="2.6" fill="#fff" className="mono">{b.cls} {b.conf.toFixed(2)}</text>
                  </g>
                  );
                })}
              </svg>
            </div>
            <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 10 }}>
              <span className="sub" style={{ margin: 0 }}>confidence ≥ {conf.toFixed(2)}</span>
              <input type="range" min={0.1} max={0.95} step={0.05} value={conf} onChange={(e) => setConf(+e.target.value)} style={{ flex: 1 }} />
              <span className="mono" style={{ fontWeight: 700 }}>{boxes.length} 검출</span>
            </div>
            {live && (
              <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 10 }}>
                <button className="btn" style={{ borderColor: "var(--green)", color: "var(--green)" }} disabled={busy} onClick={runLive}>
                  {busy ? "추론 중…" : "⚡ LIVE 추론(이 이미지 업로드)"}
                </button>
                {liveBoxes && <span className="badge b-ok">백엔드 YOLO 실추론</span>}
                {liveErr && <span className="sub" style={{ margin: 0, color: "var(--red)" }}>{liveErr}</span>}
              </div>
            )}
            {!live && <div className="sub" style={{ marginTop: 8 }}>박스 = 실제 YOLO11m 추론 결과(저장본). LIVE 연결 시 이 이미지를 즉석 재추론.</div>}
          </Card>
          <div className="grid">
            <Card title="이미지 선택">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, maxHeight: 360, overflow: "auto" }}>
                {DET_IDX.map((i) => <img key={i} src={url(ei(i))} onClick={() => setImgIdx(i)}
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

      {view === "det" && <div className="note warn">ℹ ELLIMAC은 Roboflow의 <b>실제 칩/다이 표면 사진</b>(증강 적용) — 웨이퍼맵과 <b>다른 도메인</b>이라 우리 과제 일반화는 제한적(검출 스킬 데모). 폴리곤→bbox 정제 + cls6 18줄 제거 후 YOLO11m test <b>mAP@0.5 0.753</b>(bestV2 0.739↑, 11l 0.755 동률→11m 유지).</div>}

      <div className="note" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
        <span>결함 판정·조치는 통합 콘솔에서 종합됩니다.</span>
        <span style={{ display: "flex", gap: 8 }}>
          <button className="btn" onClick={() => go && go("experiments")}>Experiments</button>
          <button className="btn on" onClick={() => go && go("dashboard")}>통합 콘솔 →</button>
        </span>
      </div>
    </div>
  );
}
