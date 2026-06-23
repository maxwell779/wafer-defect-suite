import React, { useState, useEffect } from "react";
import Dashboard from "./pages/Dashboard.jsx";
import Stage1Process from "./pages/Stage1Process.jsx";
import Stage2WaferMap from "./pages/Stage2WaferMap.jsx";
import Stage3Detection from "./pages/Stage3Detection.jsx";
import Experiments from "./pages/Experiments.jsx";
import { apiHealth } from "./api.js";
import { useI18n, useTheme } from "./i18n.jsx";

const TABS = [
  ["dashboard", "통합 콘솔"],
  ["stage1", "Stage 1 · 공정 모니터"],
  ["stage2", "Stage 2 · 웨이퍼맵"],
  ["stage3", "Stage 3 · 결함 검출/위치"],
  ["experiments", "Experiments"],
];

export default function App() {
  const [tab, setTab] = useState("dashboard");
  const [live, setLive] = useState(false);
  const { lang, t, setLang } = useI18n();
  const [theme, setTheme] = useTheme();
  useEffect(() => { apiHealth().then(setLive); }, []);
  return (
    <div className="app">
      <div className="topbar">
        <div className="brand"><span className="dot" /> Wafer Defect Console</div>
        <div className="tabs">
          {TABS.map(([id, label]) => (
            <button key={id} className={"tab" + (tab === id ? " active" : "")} onClick={() => setTab(id)}>{t(label)}</button>
          ))}
        </div>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <button className="theme-btn" title="언어" aria-label="언어 전환" onClick={() => setLang(lang === "ko" ? "en" : "ko")}>{lang === "ko" ? "EN" : "한"}</button>
          <button className="theme-btn" title="테마" aria-label="다크모드 전환" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? "☀" : "🌙"}</button>
          <span className="demo" style={live ? { color: "var(--green)", borderColor: "var(--green)" } : {}}>{live ? "LIVE" : "DEMO"}</span>
        </div>
      </div>
      <div className="wrap">
        {tab === "dashboard" && <Dashboard go={setTab} />}
        {tab === "stage1" && <Stage1Process live={live} go={setTab} />}
        {tab === "stage2" && <Stage2WaferMap live={live} go={setTab} />}
        {tab === "stage3" && <Stage3Detection live={live} go={setTab} />}
        {tab === "experiments" && <Experiments go={setTab} />}
      </div>
    </div>
  );
}
