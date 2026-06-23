#!/usr/bin/env bash
# Stage2 하이퍼파라미터 그리드 (DL판 megasearch, 현실 규모)
# lr × width × pool × wd = 3×2×2×2 = 24조합. resnet/asl/augment 고정.
# 각 raw test macro-F1을 RESULTS_stage2grid.md에 기록. 실패 격리.
set +e
cd "$(dirname "$0")/../.." || exit 1
PY="G:/anaconda3/python.exe"
export PYTHONIOENCODING=utf-8
OUT="docs/overnight/RESULTS_stage2grid.md"
LOG="docs/overnight/logs"; mkdir -p "$LOG"
echo "# Stage2 하이퍼파라미터 그리드 ($(date '+%Y-%m-%d %H:%M'))" > "$OUT"
echo "> resnet·asl·augment 고정 · raw test macro-F1(보정 전) · 단일모델 ceiling 탐색" >> "$OUT"
echo "" >> "$OUT"
echo "| lr | width | pool | wd | macro-F1 | mAP | exact |" >> "$OUT"
echo "|---|---|---|---|---|---|---|" >> "$OUT"

best=0; bestcfg=""
for lr in 5e-4 1e-3 2e-3; do
 for w in 48 64; do
  for pool in gap gem; do
   for wd in 1e-4 1e-3; do
    tag="g_${lr}_w${w}_${pool}_wd${wd}"
    lf="$LOG/s2grid_${tag}.log"
    "$PY" -u -m src.stage2_wafermap.train_real --arch resnet --width $w --pool $pool \
      --loss asl --lr $lr --wd $wd --augment --epochs 22 --normal-cap 10000 --tag "_$tag" > "$lf" 2>&1
    line=$(grep -E "^macro-F1" "$lf" | tail -1)
    mf=$(echo "$line" | grep -oE "macro-F1 [0-9.]+" | grep -oE "[0-9.]+")
    map=$(echo "$line" | grep -oE "mAP [0-9.]+" | grep -oE "[0-9.]+")
    ex=$(echo "$line" | grep -oE "exact-match [0-9.]+" | grep -oE "[0-9.]+")
    echo "| $lr | $w | $pool | $wd | ${mf:-FAIL} | ${map:--} | ${ex:--} |" >> "$OUT"
    if [ -n "$mf" ] && awk "BEGIN{exit !($mf>$best)}"; then best=$mf; bestcfg="$tag"; fi
    echo "[$(date '+%H:%M')] $tag → ${mf:-FAIL}"
   done
  done
 done
done
echo "" >> "$OUT"
echo "**best 단일모델: $best ($bestcfg)** · 비교: 6-앙상블+보정 0.935(전체 best 유지)" >> "$OUT"
echo "=== STAGE2 GRID DONE: best $best ($bestcfg) ===" >> "$OUT"
echo "=== STAGE2 GRID DONE ==="
