#!/bin/bash
set -e
MODEL="$1"; DATA="$2"; EPOCHS="${3:-30}"; IMGSZ="${4:-640}"; RUNNAME="${5:-emergency}"
BASE="$(cd "$(dirname "$0")/.." && pwd)"

# 앞쪽 층 고정(freeze). 6번째 인자 → database/settings.json 의 training_freeze → 0(고정 안 함).
# YOLO11n 은 0~9 번 층이 특징 추출(backbone)이다. freeze=10 이면 그 부분은 이어받은 그대로 두고
# 뒤쪽만 학습한다 — 계산이 줄어 빨라지고 기존 빵을 덜 잊는다. 대신 새 빵에 덜 맞춰질 수 있다.
FREEZE="${6:-$(python3 -c "import json;print(int(json.load(open('$BASE/database/settings.json')).get('training_freeze') or 0))" 2>/dev/null || echo 0)}"
case "$FREEZE" in ''|*[!0-9]*) FREEZE=0;; esac
echo "[학습] freeze=$FREEZE  epochs=$EPOCHS  imgsz=$IMGSZ  model=${MODEL##*/}"

# docker 그룹에 속해 있으면 sudo 없이 실행한다.
# (sudo 를 쓰면 GUI 에서 실행할 때 비밀번호 입력을 기다리다 멈춘다)
DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

case "$MODEL" in "$BASE"/*) ;; *) echo "ERROR: .pt는 $BASE 아래(models/training 권장)에 두세요."; exit 2;; esac
case "$DATA" in "$BASE"/*) ;; *) echo "ERROR: data.yaml 경로 오류"; exit 2;; esac
REL_MODEL="${MODEL#$BASE/}"

# data.yaml 의 path 는 호스트 절대경로("/home/aidl/.../bakery_gui/dataset/master")인데
# 컨테이너 안에서는 프로젝트가 /workspace/bakery_gui 로 마운트되므로 그대로는 못 찾는다.
# 호스트 경로를 컨테이너 경로로 바꾼 사본을 만들어 넘긴다. (원본 data.yaml 은 그대로 둔다)
CONT_DATA="$BASE/dataset/_data_container.yaml"
sed "s|$BASE|/workspace/bakery_gui|g" "$DATA" > "$CONT_DATA"
REL_DATA="dataset/_data_container.yaml"
# 컨테이너 안 경로(/workspace/bakery_gui)를 호스트 경로로 바꿔서 출력한다.
# 그래야 GUI 가 로그에서 읽는 BEST_MODEL= 경로가 젯슨에서 실제로 존재한다.
set -o pipefail
$DOCKER run --rm --runtime nvidia --ipc=host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e YOLO_CONFIG_DIR=/opt/ultralytics -e MPLCONFIGDIR=/tmp/mpl \
  -v "$BASE:/workspace/bakery_gui" -w /workspace/bakery_gui \
  breadvision-train:jp62 \
  python3 -c "from ultralytics import YOLO; from pathlib import Path; m=YOLO('/workspace/bakery_gui/$REL_MODEL'); r=m.train(data='/workspace/bakery_gui/$REL_DATA',epochs=$EPOCHS,imgsz=$IMGSZ,batch=2,workers=2,project='/workspace/bakery_gui/runs',name='$RUNNAME',exist_ok=False,device=0,degrees=0,freeze=$FREEZE); print('BEST_MODEL='+str(Path(r.save_dir)/'weights'/'best.pt'), flush=True)" \
  | sed "s|/workspace/bakery_gui|$BASE|g"
