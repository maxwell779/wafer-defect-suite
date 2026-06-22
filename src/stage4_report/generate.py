"""Stage 4 — 통합 원인추론 리포트 (3-스테이지 출력 결합).

공정 이상(Stage1) + 웨이퍼 패턴(Stage2) + 위치(Stage3)를 묶어
"왜→무엇→어디→조치" 리포트를 생성. 템플릿 기반(LLM API로 교체 가능 — generate_llm 훅).

실행:  python -m src.stage4_report.generate
"""
from __future__ import annotations
import json
from pathlib import Path
import config

# 웨이퍼맵 패턴 → 공정 연관 도메인 지식(예시 규칙)
PATTERN_CAUSE = {
    "Center": "중앙부 집중 — 척/가스 분포 비대칭, 중앙 압력 편차 의심",
    "Donut": "환형 — 회전 불균일·온도 링 편차",
    "Edge-Ring": "가장자리 링 — 에지 클램프/식각 균일도 문제",
    "Edge-Loc": "가장자리 국소 — 핸들링/엣지 결함",
    "Loc": "국소 — 입자 낙하·국부 오염",
    "Scratch": "선형 — 기계적 긁힘(핸들링)",
    "Random": "산발 — 전반적 공정 불안정/오염",
    "Near-full": "전면 — 심각한 공정 실패(레시피 이탈)",
}


def build_report(wafer_id, proc, pattern, location, anomaly):
    """proc: {feature:val,z,recs}, pattern: 예측 클래스, location: 위치 설명."""
    lines = [f"# 웨이퍼 {wafer_id} 결함 분석 리포트", ""]
    lines.append(f"## 1. 공정(왜) — 이상점수 {anomaly:.2f}")
    if proc.get("recommendations"):
        for r in proc["recommendations"]:
            lines.append(f"- {r['msg']}")
    else:
        lines.append("- 공정 파라미터 정상범위")
    lines.append(f"\n## 2. 웨이퍼 패턴(무엇) — **{pattern}**")
    lines.append(f"- {PATTERN_CAUSE.get(pattern, '패턴 미상')}")
    lines.append(f"\n## 3. 위치(어디)\n- {location}")
    lines.append(f"\n## 4. 종합 추정 원인 & 조치")
    cause = PATTERN_CAUSE.get(pattern, "").split("—")[-1].strip()
    rec = proc["recommendations"][0]["msg"] if proc.get("recommendations") else "공정 모니터링 유지"
    lines.append(f"- 추정: {pattern} 패턴({cause}) + 공정 편차 연계")
    lines.append(f"- 조치: {rec}; 해당 lot 격리·재검사")
    return "\n".join(lines)


def generate_llm(context):
    """(옵션) LLM API로 자연어 리포트. 키 없으면 템플릿 폴백."""
    return None  # TODO: Claude/OpenAI 연결 지점


def main():
    APP = Path(__file__).resolve().parents[2] / "web/src/appdata"
    runs = json.load(open(APP / "process_runs.json", encoding="utf-8"))
    # 가장 이상점수 높은 결함 run 1건으로 데모
    defect = sorted([r for r in runs if r["defect"] == 1], key=lambda r: -r["anomaly_score"])[0]
    proc = {"recommendations": [{"msg": "압력 정상 평균 대비 낮음 → 상향 권장"}]}
    rep = build_report(defect["wafer_id"], proc, "Center", "맵 중앙부(Grad-CAM 빨강)", defect["anomaly_score"])
    out = config.EXPERIMENTS / "stage4_report"; out.mkdir(parents=True, exist_ok=True)
    (out / "sample_report.md").write_text(rep, encoding="utf-8")
    print(rep)
    print(f"\n[saved] {out/'sample_report.md'}")


if __name__ == "__main__":
    main()
