#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# docker 그룹에 속해 있으면 sudo 없이 실행한다.
# (sudo 를 쓰면 GUI 에서 실행할 때 비밀번호 입력을 기다리다 멈춘다)
DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

echo "[1/4] Docker 확인"
if docker info >/dev/null 2>&1; then
  echo "  도커 사용 가능 (sudo 불필요) — 설치·서비스 시작 단계는 건너뜁니다."
else
  if ! command -v docker >/dev/null 2>&1; then
    sudo apt update
    sudo apt install -y docker.io
  fi
  sudo systemctl enable --now docker
  echo "  주의: 이 계정을 docker 그룹에 넣으면 다음부터 sudo 없이 쓸 수 있습니다."
  echo "        sudo usermod -aG docker $USER   (그 뒤 재로그인)"
fi

echo "[2/4] NVIDIA runtime 확인"
if ! $DOCKER info 2>/dev/null | grep -qi nvidia; then
  echo "주의: docker info에 nvidia runtime이 보이지 않습니다. JetPack의 nvidia-container-toolkit 설치 상태를 확인합니다."
  dpkg -l | grep -E 'nvidia-container|nvidia-docker' || true
fi

echo "[3/4] JetPack 6.2용 NVIDIA PyTorch iGPU 이미지 다운로드 (용량 큼)"
$DOCKER pull nvcr.io/nvidia/pytorch:25.06-py3-igpu

echo "[4/4] BreadVision 학습 이미지 생성"
$DOCKER build -t breadvision-train:jp62 -f training/Dockerfile .

echo "[GPU TEST]"
$DOCKER run --rm --runtime nvidia nvcr.io/nvidia/pytorch:25.06-py3-igpu python3 -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")'
echo "완료"
