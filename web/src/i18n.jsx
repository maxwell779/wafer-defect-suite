import React, { createContext, useContext, useState, useEffect } from "react";

// 한→영 사전(보이는 핵심 문자열). 없는 키는 한국어 그대로 폴백.
const EN = {
  // nav
  "통합 콘솔": "Console", "Stage 1 · 공정 모니터": "Stage 1 · Process",
  "Stage 2 · 웨이퍼맵": "Stage 2 · Wafer Map", "Stage 3 · 결함 검출/위치": "Stage 3 · Detection",
  "Experiments": "Experiments",
  // dashboard
  "통합 검사 콘솔": "Integrated Inspection Console",
  "공정(왜) → 패턴(무엇) → 위치(어디) 분석을 종합해 결함 lot을 판정·조치하는 작업자 화면":
    "Operator console — combine process(why)→pattern(what)→location(where) to judge & act on defect lots",
  "검사 완료": "Inspected", "결함 검출": "Defects", "격리 대기": "Quarantine pending", "양품률": "Yield",
  "결함 큐": "Defect queue", "처리 이력": "Action log", "파이프라인 동작": "How it works",
  "조치 실행": "Actions", "CSV 내보내기": "Export CSV", "PDF/인쇄": "PDF / Print",
  "상태 전체": "All status", "미처리": "Pending",
  // stage titles
  "Stage 1 — Process Monitor": "Stage 1 — Process Monitor",
  "Stage 2 — Wafer Map Analyzer": "Stage 2 — Wafer Map Analyzer",
  "Stage 3 — Defect Localization & Detection": "Stage 3 — Defect Localization & Detection",
  "Experiments — 단계별 실험 & 성능 향상": "Experiments — Per-stage experiments & gains",
  // buttons
  "Stage 1 공정": "Stage 1", "Stage 2 패턴": "Stage 2", "Stage 3 위치": "Stage 3",
  "통합 콘솔 →": "Console →",
};

const Ctx = createContext({ lang: "ko", t: (s) => s, setLang: () => {} });

export function I18nProvider({ children }) {
  const [lang, setLang] = useState(() => localStorage.getItem("lang") || "ko");
  useEffect(() => { localStorage.setItem("lang", lang); }, [lang]);
  const t = (s) => (lang === "en" ? (EN[s] || s) : s);
  return <Ctx.Provider value={{ lang, t, setLang }}>{children}</Ctx.Provider>;
}
export const useI18n = () => useContext(Ctx);

// 테마(다크모드)
export function useTheme() {
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "light");
  useEffect(() => {
    localStorage.setItem("theme", theme);
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);
  return [theme, setTheme];
}
