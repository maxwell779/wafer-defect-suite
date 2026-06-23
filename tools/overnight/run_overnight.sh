#!/usr/bin/env bash
# ── 오버나이트 대대적 자동 실험 오케스트레이터 ────────────────────────────
# CPU 트랙(Stage1) + GPU 트랙(Stage2/3) 병렬. 실패 격리(set +e). 결과 증분 기록.
# 사용: bash tools/overnight/run_overnight.sh  (백그라운드 권장)
set +e
cd "$(dirname "$0")/../.." || exit 1
PY="G:/anaconda3/python.exe"
export PYTHONIOENCODING=utf-8
RES="docs/overnight/RESULTS.md"
LOG="docs/overnight/logs"
mkdir -p "$LOG"
# 현재 진행중 작업 출력파일(이 세션 task)
TASKDIR="C:/Users/pro-vision/AppData/Local/Temp/claude/g--pro-vision/c19feab9-ca97-41e9-89ed-e22b13e16acd/tasks"
S2SWEEP_OUT="$TASKDIR/b2tw90bcj.output"          # Stage2 8-sweep
S1RESEARCH_OUT="$TASKDIR/b6dhbvdlw.output"        # Stage1 research_extra(ECOD/COPOD/PU)

ts(){ date "+%Y-%m-%d %H:%M:%S"; }
note(){ echo "$1" | tee -a "$RES"; }     # RES + stdout
hdr(){ echo -e "\n## $1  _($(ts))_\n" | tee -a "$RES"; }

# 한 실험 실행: $1=라벨 $2=로그파일 $3=결과추출 grep패턴 $4...=명령
runexp(){
  local label="$1" logf="$2" pat="$3"; shift 3
  echo "[$(ts)] START  $label" | tee -a "$RES"
  ( "$@" ) > "$LOG/$logf" 2>&1
  local rc=$?
  echo "[$(ts)] DONE   $label (exit $rc)" | tee -a "$RES"
  echo '```' >> "$RES"; grep -E "$pat" "$LOG/$logf" | tail -20 >> "$RES" 2>/dev/null; echo '```' >> "$RES"
}

wait_marker(){  # $1=파일 $2=마커 $3=최대분
  local f="$1" m="$2" mx="${3:-180}" i=0
  until grep -q "$m" "$f" 2>/dev/null || [ $i -ge $((mx*4)) ]; do sleep 15; i=$((i+1)); done
}
wait_gpu_free(){  # GPU mem < $1 MiB
  local lim="${1:-15000}" i=0
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)" -lt "$lim" ] || [ $i -ge 720 ]; do sleep 15; i=$((i+1)); done
}

echo "# 오버나이트 실험 결과  (시작 $(ts))" > "$RES"
echo "> CPU=Stage1 / GPU=Stage2·3 병렬. 각 실험 실패 격리. best 갱신은 ★ 표기." >> "$RES"

# ════════════════ CPU 트랙 (Stage1) ════════════════
cpu_track(){
  hdr "CPU 트랙 시작 (Stage1)"
  # 현재 진행중 bigsearch + research_extra(ECOD/COPOD/PU) 완료 대기
  wait_marker "/tmp/stage1_bigsearch.log" "ALL DONE" 120
  note "[cpu] 500-bigsearch 완료 감지"
  wait_marker "$S1RESEARCH_OUT" "STAGE1 RESEARCH CHAIN DONE" 120
  note "[cpu] research_extra(ECOD/COPOD/PU) 완료 감지"

  runexp "Stage1 research_extra2 (SMOTE·focal·MI, reps40)" "s1_extra2.log" \
    "±|DONE|pos|===" "$PY" -u -m src.stage1_process.research_extra2

  # 초대형 LGB 랜덤서치 1000조합 (광역 reps8 → 상위30 reps60)
  runexp "Stage1 LGB 1000조합 초대형서치" "s1_bigsearch1000.log" \
    "±|best|top|DONE|===" "$PY" -u tools/overnight/stage1_megasearch.py

  hdr "CPU 트랙 종료"
  note "[cpu] ALL CPU EXPERIMENTS DONE"
}

# ════════════════ GPU 트랙 (Stage2/3) ════════════════
gpu_track(){
  hdr "GPU 트랙 시작 (Stage2/3)"
  # 현재 진행중 Stage2 8-sweep + yolo 완료 대기 (그 후 GPU 여유)
  wait_marker "$S2SWEEP_OUT" "COMPREHENSIVE SWEEP DONE" 300
  note "[gpu] Stage2 8-sweep 완료 감지"
  wait_gpu_free 20000
  note "[gpu] GPU 여유 확보, Stage2 신규 실험 시작"

  # ── Stage2: cleanlab 라벨정제 (최고 ROI) ──
  runexp "Stage2 cleanlab 라벨정제" "s2_cleanlab.log" \
    "오라벨|macro-F1|Δ|baseline|cleaned|saved" \
    "$PY" -u -m src.stage2_wafermap.label_audit --epochs 22

  # ── Stage2: Mixup / CutMix ──
  for MX in mixup cutmix; do
    runexp "Stage2 $MX (resnet w48 asl)" "s2_$MX.log" "TEST|macro_f1|mAP|saved" \
      "$PY" -u -m src.stage2_wafermap.train_real --arch resnet --width 48 --loss asl \
      --augment --mixup $MX --epochs 28 --normal-cap 10000 --tag _$MX
  done

  # ── Stage2: 하이퍼파라미터 그리드 (lr × pool × width) ──
  for LR in 5e-4 2e-3; do for POOL in gap gem; do
    runexp "Stage2 grid lr=$LR pool=$POOL" "s2_grid_${LR}_${POOL}.log" "TEST|macro_f1|mAP|saved" \
      "$PY" -u -m src.stage2_wafermap.train_real --arch resnet --width 48 --loss asl \
      --augment --lr $LR --pool $POOL --epochs 25 --normal-cap 10000 --tag _g${LR}_${POOL}
  done; done

  # ── Stage2: 더 큰 모델(width96) + 더 많은 정상표본 ──
  runexp "Stage2 width96 + normal15k 30ep" "s2_w96_long.log" "TEST|macro_f1|mAP|saved" \
    "$PY" -u -m src.stage2_wafermap.train_real --arch resnet --width 96 --loss asl \
    --augment --epochs 30 --normal-cap 15000 --tag _w96big

  # ── Stage3: yolo11l (더 큰 검출기) ──
  wait_gpu_free 20000
  runexp "Stage3 yolo11l 1280" "s3_yolo11l.log" "mAP|map50|results|saved|Error" \
    "$PY" -u -m src.stage3_detection.train_yolo --model yolo11l.pt --name yolo11l_1280 \
    --imgsz 1280 --epochs 120 --patience 30 --batch 16 --reuse-ds

  hdr "GPU 트랙 종료"
  note "[gpu] ALL GPU EXPERIMENTS DONE"
}

cpu_track &  CPU_PID=$!
gpu_track &  GPU_PID=$!
wait $CPU_PID $GPU_PID
hdr "전체 오버나이트 완료"
note "ALL OVERNIGHT EXPERIMENTS FINISHED at $(ts)"
