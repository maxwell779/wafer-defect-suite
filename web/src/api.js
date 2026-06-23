// 백엔드(FastAPI) 연결 — 미실행 시 graceful 폴백(정적 데모 유지)
// dev: 별도 백엔드(localhost:8000) / prod(Docker·배포): 같은 오리진(""→/api). 정적호스트(Pages)면 404→DEMO.
const BASE = (typeof window !== "undefined" && window.__API_BASE__) ||
  (import.meta.env.DEV ? "http://localhost:8000" : "");

async function j(path, opts) {
  const r = await fetch(BASE + path, opts);
  if (!r.ok) throw new Error("api " + r.status);
  return r.json();
}

export const API_BASE = BASE;

export async function apiHealth() {
  try {
    const c = new AbortController();
    const t = setTimeout(() => c.abort(), 1500);
    const r = await fetch(BASE + "/api/health", { signal: c.signal });
    clearTimeout(t);
    return r.ok;
  } catch {
    return false;
  }
}

export const stage1Score = (params) =>
  j("/api/stage1/score", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(params) });

export const stage2Sample = (cls) => j("/api/stage2/sample/" + encodeURIComponent(cls));

export const stage3DetectSample = (idx, conf = 0.25) => j(`/api/stage3/detect_sample/${idx}?conf=${conf}`);

// 화면에 보이는 바로 그 이미지(자산)를 업로드 추론 → idx 불일치 없이 정확한 박스
export async function stage3DetectUpload(assetUrl, conf = 0.1) {
  const blob = await (await fetch(assetUrl)).blob();
  const fd = new FormData();
  fd.append("file", blob, "img.jpg");
  return j(`/api/stage3/detect?conf=${conf}`, { method: "POST", body: fd });
}
