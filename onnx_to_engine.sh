#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "사용법: ./onnx_to_engine.sh models/best.onnx"
    exit 1
fi

ONNX="$(realpath "$1")"
ENGINE="${ONNX%.onnx}.engine"

echo "ONNX  : $ONNX"
echo "ENGINE: $ENGINE"

/usr/src/tensorrt/bin/trtexec \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --fp16

echo
echo "완료: $ENGINE"
