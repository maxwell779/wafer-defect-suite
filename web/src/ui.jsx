import React from "react";

export const WM_CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Scratch", "Random"];

/* CSV 다운로드 — rows: 배열의 배열(첫 행 헤더 포함) */
export function downloadCSV(filename, rows) {
  const csv = rows.map((r) => r.map((c) => {
    const s = String(c ?? "");
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }).join(",")).join("\n");
  const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8" });   // BOM=엑셀 한글
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = filename; a.click();
  URL.revokeObjectURL(a.href);
}

export const STATUS_BADGE = { OK: "b-ok", WARN: "b-warn", FAIL: "b-fail" };

export function Card({ title, sub, children, style }) {
  return (
    <div className="card" style={style}>
      {title && <h3>{title}</h3>}
      {sub && <div className="ch-sub">{sub}</div>}
      {children}
    </div>
  );
}

/* 가로 막대 (라벨 + 값) */
export function HBars({ rows, max, fmt = (v) => v.toFixed(3), colorFn }) {
  const m = max ?? Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
      {rows.map((r, i) => (
        <div key={i} style={{ display: "grid", gridTemplateColumns: "110px 1fr 54px", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 12.5 }}>{r.label}</span>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: (Math.abs(r.value) / m) * 100 + "%", background: colorFn ? colorFn(r) : "var(--blue)" }} />
          </div>
          <span className="mono" style={{ fontSize: 12, textAlign: "right" }}>{fmt(r.value)}</span>
        </div>
      ))}
    </div>
  );
}

/* 라인차트 (다중 시리즈) — series: [{name,color,dash,points:[v..]}], xlabels */
export function LineChart({ series, xlabels, h = 240, yMin = 0, yMax = 1 }) {
  const W = 720, pad = 38, n = xlabels.length;
  const X = (i) => pad + (i * (W - pad * 2)) / Math.max(n - 1, 1);
  const Y = (v) => h - 30 - ((v - yMin) / (yMax - yMin)) * (h - 60);
  return (
    <svg viewBox={`0 0 ${W} ${h}`} style={{ width: "100%" }}>
      {[0, 0.25, 0.5, 0.75, 1].map((g, i) => {
        const v = yMin + g * (yMax - yMin);
        return (
          <g key={i}>
            <line x1={pad} x2={W - pad} y1={Y(v)} y2={Y(v)} stroke="var(--grid)" />
            <text x={6} y={Y(v) + 3} fontSize="10" fill="var(--axis)">{v.toFixed(2)}</text>
          </g>
        );
      })}
      {series.map((s, si) => (
        <g key={si}>
          <polyline points={s.points.map((v, i) => `${X(i)},${Y(v)}`).join(" ")} fill="none"
            stroke={s.color} strokeWidth="2" strokeDasharray={s.dash || "0"} />
          {s.points.map((v, i) => (
            <circle key={i} cx={X(i)} cy={Y(v)} r="3.2" fill={s.color}>
              <title>{s.name} · {xlabels[i]}: {v.toFixed(3)}</title>
            </circle>
          ))}
        </g>
      ))}
      {xlabels.map((l, i) => (
        <text key={i} x={X(i)} y={h - 10} fontSize="10.5" fill="var(--axis2)" textAnchor="middle">{l}</text>
      ))}
    </svg>
  );
}

/* 관리도: 값 시계열 + mean/UCL/LCL */
export function ControlChart({ values, h = 230 }) {
  const W = 720, pad = 40, n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
  const ucl = mean + 3 * sd, lcl = mean - 3 * sd;
  const lo = Math.min(lcl, ...values), hi = Math.max(ucl, ...values);
  const X = (i) => pad + (i * (W - pad * 2)) / Math.max(n - 1, 1);
  const Y = (v) => h - 24 - ((v - lo) / (hi - lo + 1e-9)) * (h - 48);
  return (
    <svg viewBox={`0 0 ${W} ${h}`} style={{ width: "100%" }}>
      <rect x={pad} y={Y(ucl)} width={W - pad * 2} height={Y(lcl) - Y(ucl)} fill="#16a34a08" />
      {[["UCL", ucl, "#dc2626"], ["mean", mean, "#0f172a"], ["LCL", lcl, "#dc2626"]].map(([t, v, c], i) => (
        <g key={i}>
          <line x1={pad} x2={W - pad} y1={Y(v)} y2={Y(v)} stroke={c} strokeDasharray={t === "mean" ? "0" : "5 4"} strokeWidth="1" />
          <text x={W - pad + 2} y={Y(v) + 3} fontSize="9.5" fill={c} className="mono">{t} {v.toFixed(1)}</text>
        </g>
      ))}
      <polyline points={values.map((v, i) => `${X(i)},${Y(v)}`).join(" ")} fill="none" stroke="#60a5fa" strokeWidth="1" />
      {values.map((v, i) => (
        <circle key={i} cx={X(i)} cy={Y(v)} r="1.8" fill={v > ucl || v < lcl ? "#dc2626" : "#3b82f6"}>
          <title>#{i + 1}: {v.toFixed(1)}{v > ucl || v < lcl ? " (관리한계 벗어남)" : ""}</title>
        </circle>
      ))}
    </svg>
  );
}

/* 산점도 */
export function Scatter({ points, h = 230, xlab = "x", ylab = "y" }) {
  const W = 560, pad = 36;
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const X = (v) => pad + ((v - xmin) / (xmax - xmin + 1e-9)) * (W - pad * 2);
  const Y = (v) => h - 26 - ((v - ymin) / (ymax - ymin + 1e-9)) * (h - 46);
  return (
    <svg viewBox={`0 0 ${W} ${h}`} style={{ width: "100%" }}>
      {points.map((p, i) => (
        <circle key={i} cx={X(p.x)} cy={Y(p.y)} r={p.defect ? 4 : 2.4}
          fill={p.defect ? "#dc2626" : "#93c5fd"} opacity={p.defect ? 0.95 : 0.6}>
          <title>{xlab} {p.x.toFixed(1)} · {ylab} {p.y.toFixed(2)}{p.defect ? " · 결함" : ""}</title>
        </circle>
      ))}
      <text x={W / 2} y={h - 6} fontSize="10" fill="var(--axis)" textAnchor="middle">{xlab}</text>
      <text x={10} y={14} fontSize="10" fill="var(--axis)">{ylab}</text>
    </svg>
  );
}

/* 혼동행렬 히트맵 */
export function Confusion({ labels, matrix }) {
  const max = Math.max(...matrix.flat());
  const diag = (i, j) => i === j;
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="mono" style={{ fontSize: 11, width: "auto" }}>
        <thead>
          <tr><th></th>{labels.map((l) => <th key={l} style={{ writingMode: "vertical-rl", padding: 4 }}>{l}</th>)}</tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td style={{ fontWeight: 600 }}>{labels[i]}</td>
              {row.map((v, j) => {
                const t = v / (max || 1);
                const bg = diag(i, j) ? `rgba(22,163,74,${0.15 + t * 0.7})` : `rgba(220,38,38,${t * 0.8})`;
                return <td key={j} title={`${labels[i]}→${labels[j]}: ${v}`} style={{ textAlign: "center", background: v ? bg : "var(--chartbg)", color: t > 0.5 ? "#fff" : "var(--ink)", borderColor: "var(--border)" }}>{v}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
