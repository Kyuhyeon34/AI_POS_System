FROM nvcr.io/nvidia/pytorch:25.06-py3-igpu

# 이 이미지의 torch/torchvision/numpy 는 JetPack 에 맞춰 빌드된 것이라 건드리면 안 된다.
#   - ultralytics 를 그냥 설치하면 numpy 2.x 와 다른 torchvision 을 끌어와
#     "Numpy is not available" / torchvision 불일치로 학습이 죽는다.
# 그래서 의존성 해석을 끄고(--no-deps) 필요한 것만 골라 넣는다.
# opencv 는 컨테이너에 화면이 없으므로 headless (X11 라이브러리 불필요).
RUN python3 -m pip install --no-cache-dir --no-deps \
        ultralytics==8.4.128 ultralytics-thop \
 && python3 -m pip install --no-cache-dir --no-deps opencv-python-headless \
 && python3 -m pip install --no-cache-dir \
        "numpy==1.26.4" matplotlib pandas seaborn py-cpuinfo psutil pyyaml requests scipy tqdm pillow \
        filelock polars nvidia-ml-py ultralytics-platform \
        "onnx<1.18" onnxslim

# 컨테이너를 호스트 사용자 권한(--user)으로 돌리면 기본 설정 폴더에 쓸 수 없어서
# ultralytics 가 Arial.ttf 를 받다가 실패한다. 쓰기 가능한 경로를 정하고 폰트를 미리 넣어둔다.
ENV YOLO_CONFIG_DIR=/opt/ultralytics
ENV MPLCONFIGDIR=/tmp/mpl
# 실제 설정 폴더는 $YOLO_CONFIG_DIR/Ultralytics 이다. 폰트도 그 안에 있어야 한다.
RUN mkdir -p /opt/ultralytics/Ultralytics /tmp/mpl \
 && python3 -c "import urllib.request; urllib.request.urlretrieve('https://ultralytics.com/assets/Arial.ttf', '/opt/ultralytics/Ultralytics/Arial.ttf')"

RUN python3 -c "import numpy, torch, torchvision, cv2, ultralytics, polars, onnx; \
print('numpy', numpy.__version__); print('torch', torch.__version__); \
print('torchvision', torchvision.__version__); print('cv2', cv2.__version__); \
print('ultralytics', ultralytics.__version__); \
assert numpy.__version__.startswith('1.26'), 'numpy 가 바뀌었습니다'"

# --user 로 실행해도 쓸 수 있도록 마지막에 권한을 연다
RUN chmod -R 777 /opt/ultralytics /tmp/mpl
