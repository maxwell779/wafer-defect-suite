import React, { useState } from "react";
import { Card, HBars, LineChart, Confusion } from "../ui.jsx";
import m from "../appdata/stage2_metrics.json";
import s1 from "../appdata/stage1_results.json";

const STAGE_META = [
  { n: "1", t: "공정 센서 이상탐지", c: "var(--blue)",
    data: "Meruva (실데이터) — 공정 run 5,000건, 센서 6종(온도·압력·가스·식각률·전압·전류), 결함 7건(0.14%)",
    goal: "결함이 극소수라 지도학습 불가 → 정상 패턴만 학습해 비정상 run을 가려내기",
    exp: "Mahalanobis·LOF·OCSVM·IsolationForest·AutoEncoder 비교 → 도메인 표적 피처(저압×고온/식각) 설계 → 지도/비지도 하이브리드·스태킹·1000조합 튜닝" },
  { n: "2", t: "웨이퍼 패턴 분류", c: "var(--violet)",
    data: "WM-811K (실, 81만장) + MixedWM38 (합성) — 52×52 웨이퍼맵, 8개 결함 패턴 멀티라벨",
    goal: "맵에 어떤 결함 패턴이 있는지 다중 분류 + 합성으로 학습하면 실제에서 통할지 검증",
    exp: "합성→실제 전이 실험 → 실데이터 직접 학습 → 증강·자기지도(SSL)·구조개선(SE-ResNet)·앙상블·손실함수·풀링·라벨정제 등 광범위 탐색" },
  { n: "3", t: "결함 위치/검출", c: "var(--green)",
    data: "WM-811K (실맵, Grad-CAM용) + ELLIMAC (실제 칩표면 사진, Roboflow)",
    goal: "결함이 맵의 어디에 있는지 시각화 + 칩표면 사진에서 결함을 박스로 검출",
    exp: "Stage2 모델 Grad-CAM 히트맵 + YOLO 검출(폴리곤→bbox 정제, 11n/m/l 비교, imgsz·batch 튜닝)" },
];

export default function Experiments({ go }) {
  const cmp = m.comparison;
  const [confReal, setConfReal] = useState(false);

  return (
    <div className="grid">
      <div><h1 className="page">Experiments — 단계별 실험 & 성능 향상</h1>
        <div className="sub">각 단계가 어떤 데이터로 무슨 목표를 잡고 어떤 실험을 했는지, 그리고 성능을 어떻게 끌어올렸는지 요약 — 효과 없던 방법(negative)도 그대로 공개</div></div>

      {/* 단계별 데이터·목표·실험 */}
      <div className="grid" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
        {STAGE_META.map((s) => (
          <div key={s.n} className="card" style={{ borderTop: `4px solid ${s.c}` }}>
            <div className="klabel" style={{ color: s.c }}>STAGE {s.n}</div>
            <h3 style={{ margin: "6px 0 12px" }}>{s.t}</h3>
            <div style={{ fontSize: 13.5, lineHeight: 1.6, display: "flex", flexDirection: "column", gap: 9 }}>
              <div><b style={{ color: s.c }}>데이터</b><br />{s.data}</div>
              <div><b style={{ color: s.c }}>목표</b><br />{s.goal}</div>
              <div><b style={{ color: s.c }}>한 실험</b><br />{s.exp}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Stage1 성능 향상 */}
      <Card title="Stage 1 — 성능을 어떻게 올렸나 (PR-AUC)" sub="원시 센서로는 0.31 → 도메인 메커니즘 인코딩 + 하이브리드로 0.81 (leak-free, 반복 CV)">
        <HBars rows={s1.progression.map((p) => ({ label: p.stage, value: p.pr_auc }))}
          max={1} fmt={(v) => v.toFixed(3)}
          colorFn={(r) => (r.value >= 0.78 ? "var(--green)" : r.value >= 0.44 ? "var(--blue)" : "#94a3b8")} />
        <div className="note" style={{ marginTop: 12 }}>
          결함 7건이 전부 <b>압력↓ + 고온/식각</b> 조합이라는 점을 찾아, "결함 = 센서 비정상 조합"을 <b>표적 피처로 직접 인코딩</b>(0.31→0.78). 여기에 비지도 이상점수를 지도모델 입력으로 합친 <b>하이브리드로 0.81</b>. 무차별 전수 조합은 차원의 저주로 오히려 붕괴(0.07~0.22).
        </div>
        <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 14 }}>
          <div>
            <div style={{ fontWeight: 700, marginBottom: 6 }}>고전 ML vs 딥러닝 (원시 피처, 비지도)</div>
            <table>
              <thead><tr><th>모델</th><th>PR-AUC</th><th>ROC</th><th>R@100</th></tr></thead>
              <tbody>
                {[...s1.comparison].sort((a, b) => b.pr_auc - a.pr_auc).map((x, i) => (
                  <tr key={x.model} style={{ background: i === 0 ? "#f0fdf4" : "" }}>
                    <td>{x.model}{x.model.includes("DL") && <span className="badge b-tag" style={{ marginLeft: 6 }}>DL</span>}</td>
                    <td className="mono">{x.pr_auc}</td><td className="mono">{x.roc_auc}</td><td className="mono">{x.recall_at_100}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="sub" style={{ marginTop: 8 }}>소표본·저차원이라 딥러닝(AE)은 더 낮고 불안정 → 고전 ML이 유리.</div>
          </div>
          <div>
            <div style={{ fontWeight: 700, marginBottom: 6 }}>시도했지만 효과 없던 기법</div>
            <table>
              <thead><tr><th>방법</th><th>PR-AUC</th><th>결과</th></tr></thead>
              <tbody>
                {s1.negatives.map((n) => (
                  <tr key={n.method}>
                    <td style={{ fontSize: 13 }}>{n.method}</td>
                    <td className="mono">{n.pr_auc}</td>
                    <td><span className={"badge " + (n.pr_auc >= 0.79 ? "b-warn" : "b-fail")}>{n.verdict}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="sub" style={{ marginTop: 8 }}>1000조합 튜닝·스태킹·ECOD/COPOD·PU·SMOTE 모두 0.81을 못 넘음 = 결함 7건이 통계적 한계.</div>
          </div>
        </div>
      </Card>

      {/* Stage2 성능 향상 */}
      <Card title="Stage 2 — 학습 전략별 성능 추이 (macro-F1 / mAP)" sub="왼→오로 갈수록 더 나은 전략. 가운데 급락 = 합성→실제 전이 실패">
        <LineChart xlabels={cmp.map((c) => c.name)}
          series={[
            { name: "macro-F1", color: "var(--blue)", points: cmp.map((c) => c.macro_f1) },
            { name: "mAP", color: "var(--muted)", dash: "5 4", points: cmp.map((c) => c.mAP) },
          ]} />
        <div style={{ display: "flex", gap: 18, fontSize: 13, color: "var(--muted)" }}>
          <span><span style={{ display: "inline-block", width: 14, borderTop: "2px solid var(--blue)", verticalAlign: "middle" }} /> macro-F1</span>
          <span><span style={{ display: "inline-block", width: 14, borderTop: "2px dashed var(--muted)", verticalAlign: "middle" }} /> mAP</span>
        </div>
        <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 8, gap: 10, fontSize: 13.5 }}>
          {[
            ["합성 학습", "GAN 합성맵으로만 학습 → 합성 테스트 0.985(쉬움)"],
            ["합성→실제 전이", "그 모델을 실제 맵에 적용 → 0.36 붕괴 (핵심 발견)"],
            ["실데이터 직접 학습", "실제 WM-811K로 직접 학습 → 0.86 회복"],
            ["CNN 강화+보정", "증강·채널확대·클래스별 임계보정 → 0.90"],
            ["SE-ResNet 단일", "깊은 백본 + 채널 attention → 0.91"],
            ["SE-ResNet 6-앙상블", "이질 6모델 평균+보정 → 최종 0.935"],
          ].map(([t, d], i) => <div key={i}><b>{t}</b> — {d}</div>)}
        </div>
      </Card>

      <div className="grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <Card title="Stage 2 — 클래스별 F1 (6-앙상블+보정)" sub="Loc(0.86)·Scratch(0.87) 최저 — 사람도 헷갈리는 본질 모호성">
          <HBars rows={m.per_class.map((p) => ({ label: p.cls, value: p.f1 }))} max={1} fmt={(v) => v.toFixed(3)}
            colorFn={(r) => (r.value < 0.8 ? "var(--red)" : r.value < 0.9 ? "var(--amber)" : "var(--blue)")} />
          <div className="note warn" style={{ marginTop: 12 }}>
            <b>0.935가 한계인 이유</b> — 라벨 오류 의심분(1.3%)을 자동 검출·제거해 다시 학습해도 변화 없음(−0.005). 즉 고칠 수 있는 오류가 아니라 <b>Loc↔Edge-Loc처럼 사람이 봐도 헷갈리는 모호성</b>이 한계. 손실함수·Mixup·풀링·더 큰 모델 추가 시도도 0.935를 못 넘음.
          </div>
        </Card>
        <Card title="왜 합성 데이터로 학습하지 않았나" sub="이 프로젝트의 핵심 교훈">
          <div style={{ fontSize: 14, lineHeight: 1.65, display: "flex", flexDirection: "column", gap: 10 }}>
            <div>합성(GAN) 웨이퍼맵으로 학습한 모델은 <b>합성 테스트에선 0.985</b>로 완벽해 보입니다.</div>
            <div>하지만 같은 모델을 <b>실제 맵</b>에 쓰면 <b style={{ color: "var(--red)" }}>macro-F1 0.36</b>으로 무너집니다 — 정상 맵을 결함으로 오탐(0.957), 대부분을 Random으로 오분류.</div>
            <div>합성은 결함 밀도·노이즈 분포가 실제와 달라(<b>도메인 갭</b>), "쉬운 가짜 문제"만 풉니다. 그래서 <b>실데이터(WM-811K)로 직접 학습</b>하는 것이 정답이었고, 합성은 학습용이 아니라 <b>전이 실패를 보여주는 대조군</b>으로만 사용했습니다.</div>
          </div>
        </Card>
      </div>

      <Card title="Stage 2 — 혼동행렬" sub={confReal ? "실데이터 모델: 대각선(정답)에 집중" : "합성 모델로 실제맵 예측: 전이 실패"}>
        <button className={"btn" + (confReal ? " on" : "")} onClick={() => setConfReal(!confReal)} style={{ marginBottom: 12 }}>
          {confReal ? "실데이터 모델 보기 ON" : "합성→실제(전이 실패) 보기"}
        </button>
        <Confusion labels={(confReal ? m.confusion_real : m.confusion_synth_on_real).labels}
          matrix={(confReal ? m.confusion_real : m.confusion_synth_on_real).matrix} />
        <div className="sub" style={{ marginTop: 10 }}>
          {confReal ? "실데이터 모델은 대각선에 집중 — Loc↔Edge-Loc 약간의 혼동만 남음." : "합성 모델은 Random 열로 대량 오분류 = 도메인 갭(전이 실패)."}
        </div>
      </Card>

      {/* Stage3 성능 향상 */}
      <Card title="Stage 3 — 검출 성능 향상 (mAP@0.5)" sub="라벨 정제 + 모델 비교로 향상. 더 큰 모델은 이득 없음(효율 유지)">
        <HBars rows={[
          { label: "bestV2 (기존)", value: 0.739 },
          { label: "YOLO11m (채택)", value: 0.753 },
          { label: "YOLO11l (무이득)", value: 0.755 },
        ]} max={1} fmt={(v) => v.toFixed(3)} colorFn={(r) => (r.label.includes("채택") ? "var(--green)" : "#94a3b8")} />
        <div className="note" style={{ marginTop: 12 }}>
          폴리곤 라벨을 bbox로 정제하고 잘못된 클래스(cls6) 18줄을 제거한 뒤 YOLO11n/m/l을 비교. <b>11m가 0.753</b>으로 bestV2(0.739)를 넘었고, 더 큰 <b>11l(0.755)은 사실상 동률 → 효율적인 11m 채택</b>. ELLIMAC은 칩표면 사진이라 웨이퍼맵과 도메인이 달라 검출 스킬 데모로 활용.
        </div>
      </Card>

      <div className="note" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
        <span>각 단계의 라이브 데모를 직접 보시려면 →</span>
        <span style={{ display: "flex", gap: 8 }}>
          <button className="btn" onClick={() => go && go("stage1")}>Stage 1</button>
          <button className="btn" onClick={() => go && go("stage2")}>Stage 2</button>
          <button className="btn" onClick={() => go && go("stage3")}>Stage 3</button>
          <button className="btn on" onClick={() => go && go("dashboard")}>통합 콘솔 →</button>
        </span>
      </div>
    </div>
  );
}
