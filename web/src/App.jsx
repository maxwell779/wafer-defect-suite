import React, { useState, useEffect } from "react";
import Dashboard from "./pages/Dashboard.jsx";
import Stage1Process from "./pages/Stage1Process.jsx";
import Stage2WaferMap from "./pages/Stage2WaferMap.jsx";
import Stage3Detection from "./pages/Stage3Detection.jsx";
import Stage4Report from "./pages/Stage4Report.jsx";
import Experiments from "./pages/Experiments.jsx";
import { apiHealth } from "./api.js";

const TABS = [
  ["dashboard", "Dashboard"],
  ["stage1", "Stage 1 · 공정 모니터"],
  ["stage2", "Stage 2 · 웨이퍼맵"],
  ["stage3", "Stage 3 · 결함 검출/위치"],
  ["stage4", "Stage 4 · 통합 리포트"],
  ["experiments", "Experiments"],
];

export default function App() {
  const [tab, setTab] = useState("dashboard");
  const [live, setLive] = useState(false);
  useEffect(() => { apiHealth().then(setLive); }, []);
  return (
    <div className="app">
      <div className="topbar">
        <div className="brand"><span className="dot" /> Wafer Defect Console</div>
        <div className="tabs">
          {TABS.map(([id, label]) => (
            <button key={id} className={"tab" + (tab === id ? " active" : "")} onClick={() => setTab(id)}>{label}</button>
          ))}
        </div>
        <span className="demo" style={live ? { color: "var(--green)", borderColor: "var(--green)" } : {}}>{live ? "LIVE" : "DEMO"}</span>
      </div>
      <div className="wrap">
        {tab === "dashboard" && <Dashboard go={setTab} />}
        {tab === "stage1" && <Stage1Process live={live} />}
        {tab === "stage2" && <Stage2WaferMap live={live} />}
        {tab === "stage3" && <Stage3Detection />}
        {tab === "stage4" && <Stage4Report />}
        {tab === "experiments" && <Experiments />}
      </div>
    </div>
  );
}
