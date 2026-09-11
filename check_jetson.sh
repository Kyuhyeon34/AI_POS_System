#!/bin/bash
echo "=== BreadVision v8.2 Jetson Check ==="

echo "[Python]"
python3 --version
echo

echo "[Camera]"
ls -l /dev/video* 2>/dev/null || echo "No /dev/video device"
echo

echo "[OpenCV]"
python3 - <<'PY'
try:
    import cv2
    print("OpenCV:", cv2.__version__)
except Exception as e:
    print("OpenCV ERROR:", e)
PY
echo

echo "[PyQt5]"
python3 - <<'PY'
try:
    import PyQt5
    print("PyQt5 OK")
except Exception as e:
    print("PyQt5 ERROR:", e)
PY
echo

echo "[TensorRT]"
python3 - <<'PY'
try:
    import tensorrt as trt
    print("TensorRT Python:", trt.__version__)
except Exception as e:
    print("TensorRT ERROR:", e)
PY
echo

echo "[PyCUDA]"
python3 - <<'PY'
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    print("PyCUDA OK")
    print("CUDA device:", cuda.Device(0).name())
except Exception as e:
    print("PyCUDA ERROR:", e)
PY
echo

echo "[trtexec]"
if [ -x /usr/src/tensorrt/bin/trtexec ]; then
    echo "/usr/src/tensorrt/bin/trtexec OK"
else
    echo "trtexec NOT FOUND"
fi

echo
echo "NOTE: v8.2 does NOT require torch or ultralytics on Jetson."
