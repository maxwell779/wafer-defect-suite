import React, { useState, useEffect } from "react";
import { Card, HBars, WM_CLASSES } from "../ui.jsx";
import { stage2Sample } from "../api.js";
import wmaps from "../appdata/wafermaps.json";

const url = (f) => "/" + f; // public/assets

export default function Stage2WaferMap({ live }) {
  const [cls, setCls] = useState("ALL");
  const realMaps = wmaps.filter((m) => m.source === "real");   // 합성 이미지는 갤러리에서 제외(실데이터만)
  const [selId, setSelId] = useState(realMaps[0]?.id || wmaps[0].id);
  const [model, setModel] = useState("real");
  const [thr, setThr] = useState(0.5);
  const [heat, setHeat] = useState(false);
  const [cmp, setCmp] = useState(false);
  const [liveRes, setLiveRes] = useState(null);
  const [liveErr, setLiveErr] = useState("");

  const gallery = realMaps.filter((m) => (cls === "ALL" || m.classes.includes(cls)));
  const sel = wmaps.find((m) => m.id === selId);
  useEffect(() => { setLiveRes(null); setLiveErr(""); }, [selId]);   // 맵 바뀌면 라이브 결과 초기화
  const staticPred = model === "real" ? sel.pred_real : sel.pred_synth;
  const pred = liveRes ? (model === "real" ? liveRes.pred_real : liveRes.pred_synth) : staticPred;
  const predRows = WM_CLASSES.map((c, i) => ({ label: c, value: pred[i] }));
  const ng = Math.max(...pred) >= thr;
  const hits = WM_CLASSES.filter((_, i) => pred[i] >= thr);
  const camAvail = sel.source === "real";

  return (
    <div className="grid">
      <div><h1 className="page">Stage 2 — Wafer Map Analyzer</h1>
        <div className="sub">8클래스 멀티라벨 분류 + 결함 위치(Grad-CAM) · 합성 vs 실데이터 모델 전이 비교</div></div>

      <div className="grid" style={{ gridTemplateColumns: "260px 1fr 1fr" }}>
        {/* 갤러리 */}
        <Card title="맵 갤러리" sub={gallery.length + " maps"}>
          <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
            <select value={cls} onChange={(e) => setCls(e.target.value)} style={{ flex: 1 }}><option value="ALL">전체 클래스</option>{WM_CLASSES.map((c) => <option key={c}>{c}</option>)}</select>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, maxHeight: 420, overflow: "auto" }}>
            {gallery.map((m) => (
              <div key={m.id} onClick={() => setSelId(m.id)} style={{ cursor: "pointer", border: m.id === selId ? "2px solid var(--blue)" : "1px solid var(--border)", borderRadius: 8, padding: 4, textAlign: "center" }}>
                <img src={url(m.file)} style={{ width: "100%", borderRadius: 4 }} />
                <div style={{ fontSize: 10.5 }}>{m.classes.join("+")}</div>
                <div style={{ fontSize: 9, color: "var(--muted)" }}>{m.source === "real" ? "실데이터" : "합성"}</div>
              </div>
            ))}
          </div>
        </Card>

        {/* 뷰어 */}
        <Card title="맵 뷰어" sub={`${sel.id} · ${sel.source === "real" ? "실데이터" : "합성"}`}>
          <div style={{ position: "relative" }}>
            <img src={url(sel.file)} style={{ width: "100%", borderRadius: 8 }} />
            {heat && camAvail && <img src={url("assets/cam_" + sel.classes[0] + ".png")} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", opacity: 0.55, borderRadius: 8 }} />}
          </div>
          <div style={{ marginTop: 10, display: "flex", gap: 8, alignItems: "center" }}>
            <button className={"btn" + (heat ? " on" : "")} disabled={!camAvail} onClick={() => setHeat(!heat)}>
              위치 히트맵 {heat ? "ON" : "OFF"}
            </button>
            <span className="sub" style={{ margin: 0 }}>{camAvail ? "Grad-CAM(실데이터) 결함 위치" : "합성맵은 위치탐지 미지원"}</span>
          </div>
        </Card>

        {/* 판정 */}
        <Card title="결함 판정" sub={`${model === "real" ? "실데이터" : "합성"} 모델 · 임계값 ${thr.toFixed(2)}`}>
          <div style={{ display: "flex", gap: 6, marginBottom: 10, flexWrap: "wrap" }}>
            <button className={"btn" + (model === "real" ? " on" : "")} onClick={() => setModel("real")}>실데이터 모델</button>
            <button className={"btn" + (model === "synth" ? " on" : "")} onClick={() => setModel("synth")}>합성 모델</button>
            {live && <button className="btn" style={{ borderColor: "var(--green)", color: "var(--green)" }}
              onClick={() => { setLiveErr(""); stage2Sample(sel.classes[0]).then(setLiveRes).catch(() => setLiveErr("백엔드 추론 실패 — 정적 예측 표시")); }}>⚡ LIVE 추론(실모델)</button>}
          </div>
          {liveRes && <div className="note" style={{ marginBottom: 10, fontSize: 12 }}>실시간 추론 결과 — 클래스 {liveRes.true_class} 실제 샘플 (백엔드 WaferCNN)</div>}
          {liveErr && <div className="note warn" style={{ marginBottom: 10, fontSize: 12 }}>{liveErr}</div>}
          <div style={{ fontSize: 22, fontWeight: 700, marginBottom: 8, color: ng ? "var(--red)" : "var(--green)" }}>
            판정 {ng ? "NG" : "OK"} {hits.map((h) => <span key={h} className="badge b-fail" style={{ marginLeft: 6, fontSize: 11 }}>{h}</span>)}
          </div>
          <HBars rows={predRows} max={1} fmt={(v) => v.toFixed(2)} colorFn={(r) => (r.value >= thr ? "var(--red)" : "#cbd5e1")} />
          <div style={{ marginTop: 12 }}>
            <div className="sub" style={{ margin: 0 }}>임계값 {thr.toFixed(2)}</div>
            <input type="range" min={0.1} max={0.9} step={0.05} value={thr} onChange={(e) => setThr(+e.target.value)} />
          </div>
        </Card>
      </div>

      <Card title="합성 vs 실데이터 모델 비교 (전이 실패 시각화)" sub="같은 맵에 두 모델 예측 — 합성모델은 실제맵을 Random으로 오인·정상 오탐 0.957">
        <button className={"btn" + (cmp ? " on" : "")} onClick={() => setCmp(!cmp)}>비교 {cmp ? "ON" : "OFF"}</button>
        {cmp && (
          <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 14 }}>
            {[["실데이터 모델", sel.pred_real], ["합성 모델", sel.pred_synth]].map(([name, p]) => (
              <div key={name}>
                <div style={{ fontWeight: 600, marginBottom: 8 }}>{name}</div>
                <HBars rows={WM_CLASSES.map((c, i) => ({ label: c, value: p[i] }))} max={1} fmt={(v) => v.toFixed(2)}
                  colorFn={(r) => (r.value >= 0.5 ? "var(--red)" : "#cbd5e1")} />
              </div>
            ))}
          </div>
        )}
        {!cmp && <div className="sub" style={{ marginTop: 10 }}>버튼을 켜면 동일 맵에 대한 두 모델 예측이 나란히 표시됩니다.</div>}
      </Card>
    </div>
  );
}
