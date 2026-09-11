#!/bin/bash
# 젯슨에서 BreadVision 을 켠다.
#   · 카메라 영상 서버   (포트 8090)  ← PC 의 POS 화면이 이 영상을 받는다
#   · AI 추론 서버       (포트 8092)  ← PC 가 "이 사진에 무슨 빵?" 하고 물어본다
#
#   사용:  ./jetson/start.sh
#   끄기:  ./jetson/stop.sh        상태: ./jetson/status.sh
#
# 바꿀 수 있는 것 (보통은 건드릴 필요 없음)
#   BREADVISION_STREAM_PORT=8090   BREADVISION_INFER_PORT=8092
#   BREADVISION_STREAM_CAMERA=0    (카메라 번호. 없으면 알아서 다른 번호를 찾는다)
#   BREADVISION_STREAM_SIZE=640x480   BREADVISION_STREAM_QUALITY=80
#   BREADVISION_NO_STREAM=1        (카메라 서버는 켜지 않고 추론 서버만)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STREAM_PORT="${BREADVISION_STREAM_PORT:-8090}"
INFER_PORT="${BREADVISION_INFER_PORT:-8092}"
CAM="${BREADVISION_STREAM_CAMERA:-0}"
SIZE="${BREADVISION_STREAM_SIZE:-640x480}"
QUALITY="${BREADVISION_STREAM_QUALITY:-80}"
RUN_DIR="${BREADVISION_RUN_DIR:-$HOME/.breadvision}"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

"$ROOT/jetson/stop.sh" --quiet || true

# 처음이면 기본 설정을 깐다 (이미 있으면 그대로 둔다 — 가격·모델 선택이 지워지지 않게)
[ -f database/settings.json ] || cp database/settings.example.json database/settings.json
[ -f database/products.json ] || cp database/products.default.json database/products.json

# 어떤 모델로 켤지: database/settings.json 의 active_engine + ensemble_engines
read -r ENGINE ENSEMBLE < <(python3 - <<'PY'
import json
from pathlib import Path
root = Path.cwd()
s = json.loads(Path("database/settings.json").read_text(encoding="utf-8"))
def full(p):
    p = Path(p)
    return str(p if p.is_absolute() else (root / p))
eng = full(s.get("active_engine") or "models/bread_0911.engine")
ens = ",".join(full(e) for e in (s.get("ensemble_engines") or []))
print(eng, ens or "-")
PY
)
[ "$ENSEMBLE" = "-" ] && ENSEMBLE=""

missing=""
for e in "$ENGINE" ${ENSEMBLE//,/ }; do
    [ -f "$e" ] || missing="$missing $e"
done
if [ -n "$missing" ]; then
    echo "TensorRT 엔진 파일이 없습니다:$missing"
    echo "먼저 만들어 주세요 (10~20분):  ./jetson/build_engines.sh"
    exit 1
fi

port_busy() { ss -ltn 2>/dev/null | grep -q ":$1 "; }

W="${SIZE%x*}"; H="${SIZE#*x}"
if [ "${BREADVISION_NO_STREAM:-0}" = "1" ]; then
    echo "[1/2] 카메라 영상 서버: 건너뜀 (BREADVISION_NO_STREAM=1)"
elif port_busy "$STREAM_PORT"; then
    echo "[1/2] 카메라 영상 서버: 포트 $STREAM_PORT 을 이미 다른 프로그램이 쓰고 있어 건너뜁니다."
    echo "      (예전에 설치한 BreadVision 이 켜져 있을 수 있습니다. 그것으로 영상은 나옵니다)"
else
    echo "[1/2] 카메라 영상 서버 시작 (포트 $STREAM_PORT, ${SIZE}, 카메라 $CAM번)"
    setsid nohup python3 -u pc_camera_stream.py \
        --camera "$CAM" --width "$W" --height "$H" --quality "$QUALITY" --port "$STREAM_PORT" \
        > "$LOG_DIR/stream.log" 2>&1 < /dev/null &
    echo $! > "$RUN_DIR/stream.pid"
fi

if port_busy "$INFER_PORT"; then
    echo "[2/2] 포트 $INFER_PORT 을 이미 다른 프로그램이 쓰고 있습니다. 추론 서버를 켤 수 없습니다."
    echo "      누가 쓰는지:  ss -ltnp | grep $INFER_PORT"
    exit 1
fi
echo "[2/2] AI 추론 서버 시작 (포트 $INFER_PORT)"
echo "      대표 모델: $(basename "$ENGINE")${ENSEMBLE:+   앙상블 짝: $(echo "$ENSEMBLE" | tr ',' '\n' | xargs -n1 basename | tr '\n' ' ')}"
# --conf 를 낮게 둔다. 확실한 것만 계산대에 담는 기준(55%)은 PC 쪽이 다시 거른다.
setsid nohup python3 -u inference_server.py --engine "$ENGINE" ${ENSEMBLE:+--ensemble "$ENSEMBLE"} \
    --conf 0.15 --port "$INFER_PORT" > "$LOG_DIR/infer.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/infer.pid"

echo -n "      모델을 올리는 중"
for i in $(seq 1 60); do
    if curl -s --max-time 2 "http://localhost:$INFER_PORT/health" | grep -q '"ready": true'; then
        echo " → 준비 완료"
        break
    fi
    if ! kill -0 "$(cat "$RUN_DIR/infer.pid")" 2>/dev/null; then
        echo; echo "추론 서버가 꺼졌습니다. 로그:"; tail -15 "$LOG_DIR/infer.log"; exit 1
    fi
    echo -n "."; sleep 2
done

"$ROOT/jetson/status.sh"
