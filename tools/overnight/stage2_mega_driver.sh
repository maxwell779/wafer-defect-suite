#!/usr/bin/env bash
# Stage2 메가서치 8-shard 병렬 드라이버 (1000조합 → 상위 풀학습)
# 각 샤드 GPU_GUARD_NEED=4000(4GB) → 8×4=32GB (80GB 여유). Stage1 메가서치는 CPU라 무관.
set +e
cd "$(dirname "$0")/../.." || exit 1
PY="G:/anaconda3/python.exe"
export PYTHONIOENCODING=utf-8
LOG="docs/overnight/logs"; mkdir -p "$LOG"
N=${N:-600}; NSH=${NSH:-2}    # 린 재시작: 2병렬·600조합(GPU 직렬화로 8병렬 비효율 확인)
rm -f "$LOG"/s2mega_shard*.json

echo "[$(date '+%H:%M')] Phase1: $NSH-shard 병렬, $N조합"
for s in $(seq 0 $((NSH-1))); do
  GPU_GUARD_NEED=4000 "$PY" -u tools/overnight/stage2_megasearch.py --phase 1 --shard $s --nshards $NSH --n $N \
    > "$LOG/s2mega_p1_shard$s.log" 2>&1 &
done
wait
echo "[$(date '+%H:%M')] Phase1 완료, 샤드 결과:"
ls -la "$LOG"/s2mega_shard*.json 2>/dev/null

echo "[$(date '+%H:%M')] Phase2: 전역 top-12 풀학습"
GPU_GUARD_NEED=8000 "$PY" -u tools/overnight/stage2_megasearch.py --phase 2 --topk 12 \
  > "$LOG/s2mega_p2.log" 2>&1
tail -5 "$LOG/s2mega_p2.log"
echo "=== STAGE2 MEGA DRIVER DONE ($(date '+%H:%M')) ==="
