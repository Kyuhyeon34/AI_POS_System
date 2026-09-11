#!/bin/bash
set -e
PT="$1"; IMGSZ="${2:-640}"
BASE="$(cd "$(dirname "$0")/.." && pwd)"

# docker 그룹에 속해 있으면 sudo 없이 실행한다.
# (sudo 를 쓰면 GUI 에서 실행할 때 비밀번호 입력을 기다리다 멈춘다)
DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

PT="$(realpath "$PT")"
case "$PT" in "$BASE"/*) ;; *) echo "ERROR: best.pt는 프로젝트 내부에 있어야 합니다."; exit 2;; esac
REL="${PT#$BASE/}"
$DOCKER run --rm --runtime nvidia --ipc=host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e YOLO_CONFIG_DIR=/opt/ultralytics -e MPLCONFIGDIR=/tmp/mpl \
  -v "$BASE:/workspace/bakery_gui" -w /workspace/bakery_gui \
  breadvision-train:jp62 \
  yolo export model="/workspace/bakery_gui/$REL" format=onnx imgsz="$IMGSZ" opset=17 dynamic=False simplify=False
ONNX="${PT%.pt}.onnx"
ENGINE="${PT%.pt}.engine"
/usr/src/tensorrt/bin/trtexec --onnx="$ONNX" --saveEngine="$ENGINE" --fp16
echo "ONNX=$ONNX"
echo "ENGINE=$ENGINE"
