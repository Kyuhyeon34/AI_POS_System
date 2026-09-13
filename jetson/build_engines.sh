#!/bin/bash
# ONNX 모델 → TensorRT 엔진(.engine) 을 만든다.
#
# 엔진은 "그 젯슨 전용" 파일이다. 다른 젯슨이나 다른 JetPack 버전에서 만든 엔진은
# 열리지 않으므로 git 에 넣지 않고, 설치할 때 각 젯슨에서 새로 만든다.
#
#   ./jetson/build_engines.sh            설정(settings.json)이 쓰는 모델만, 없는 것만
#   ./jetson/build_engines.sh --all      models/ 의 ONNX 전부
#   ./jetson/build_engines.sh --force    이미 있어도 다시 만든다
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TRTEXEC=/usr/src/tensorrt/bin/trtexec
LOG_DIR="${BREADVISION_RUN_DIR:-$HOME/.breadvision}/logs"
mkdir -p "$LOG_DIR"

ALL=0; FORCE=0
for a in "$@"; do
    case "$a" in
        --all) ALL=1 ;;
        --force) FORCE=1 ;;
        *) echo "알 수 없는 옵션: $a"; exit 2 ;;
    esac
done

[ -x "$TRTEXEC" ] || { echo "trtexec 가 없습니다 ($TRTEXEC). JetPack(TensorRT) 설치를 확인하세요."; exit 1; }
[ -f database/settings.json ] || cp database/settings.example.json database/settings.json

if [ $ALL = 1 ]; then
    ENGINES=$(ls models/*.onnx 2>/dev/null | sed 's/\.onnx$/.engine/')
else
    ENGINES=$(python3 - <<'PY'
import json
from pathlib import Path
s = json.loads(Path("database/settings.json").read_text(encoding="utf-8"))
for e in [s.get("active_engine") or "models/bread_0911.engine"] + (s.get("ensemble_engines") or []):
    p = Path(e)
    try:
        p = p.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        pass
    print(p)
PY
)
fi

n=0
for ENGINE in $ENGINES; do
    ONNX="${ENGINE%.engine}.onnx"
    if [ -f "$ENGINE" ] && [ $FORCE = 0 ]; then
        echo "있음 (건너뜀): $ENGINE"
        continue
    fi
    if [ ! -f "$ONNX" ]; then
        echo "ONNX 가 없어 만들 수 없습니다: $ONNX"
        exit 1
    fi
    n=$((n + 1))
    NAME=$(basename "$ENGINE" .engine)
    echo "[$n] $ONNX → $ENGINE   (5~10분 걸립니다. 화면이 멈춘 것처럼 보여도 기다려 주세요)"
    START=$(date +%s)
    "$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" --fp16 > "$LOG_DIR/trt_$NAME.log" 2>&1 &
    TPID=$!
    # 7분 동안 아무것도 안 나오면 멈춘 줄 안다. 30초마다 진행 중이라고 알려 준다.
    while kill -0 "$TPID" 2>/dev/null; do
        for _ in $(seq 1 30); do kill -0 "$TPID" 2>/dev/null || break; sleep 1; done
        kill -0 "$TPID" 2>/dev/null && echo "    … 변환 중 $(( ($(date +%s) - START) / 60 ))분 $(( ($(date +%s) - START) % 60 ))초 경과 (정상입니다. 기다려 주세요)"
    done
    if wait "$TPID"; then
        echo "    완료 ($(( ($(date +%s) - START) / 60 ))분)  $(du -h "$ENGINE" | cut -f1)"
    else
        rm -f "$ENGINE"
        echo "    실패 — 로그: $LOG_DIR/trt_$NAME.log"
        tail -5 "$LOG_DIR/trt_$NAME.log"
        exit 1
    fi
    if [ ! -f "$ENGINE.names.json" ]; then
        echo "    주의: $ENGINE.names.json (클래스 이름 목록)이 없습니다. 앙상블 짝 검사를 못 합니다."
    fi
done
echo "엔진 준비 끝."
