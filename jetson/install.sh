#!/bin/bash
# BreadVision 젯슨 설치 — 처음 한 번만 실행한다.
#
#   ./jetson/install.sh                 전부 (확인 → 패키지 → 엔진 → 자동 시작 등록 → 켜기)
#   ./jetson/install.sh --no-cron       부팅 자동 시작은 등록하지 않음
#   ./jetson/install.sh --no-start      설치만 하고 켜지는 않음
#   ./jetson/install.sh --skip-engines  엔진 만들기 건너뜀 (이미 만들었을 때)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUN_DIR="${BREADVISION_RUN_DIR:-$HOME/.breadvision}"
mkdir -p "$RUN_DIR/logs"

CRON=1; START=1; ENGINES=1
for a in "$@"; do
    case "$a" in
        --no-cron) CRON=0 ;;
        --no-start) START=0 ;;
        --skip-engines) ENGINES=0 ;;
        *) echo "알 수 없는 옵션: $a"; exit 2 ;;
    esac
done

ok()   { echo "   ✅ $*"; }
warn() { echo "   ⚠️  $*"; }
die()  { echo "   ❌ $*"; exit 1; }

echo "BreadVision 젯슨 설치를 시작합니다.  (폴더: $ROOT)"
echo

echo "[1/6] 젯슨 기본 구성 확인 (JetPack)"
[ -f /etc/nv_tegra_release ] || warn "젯슨이 아닌 것 같습니다. (이 스크립트는 NVIDIA Jetson 용)"
head -1 /etc/nv_tegra_release 2>/dev/null | sed 's/^/   /' || true
python3 -c "import tensorrt" 2>/dev/null && ok "TensorRT $(python3 -c 'import tensorrt; print(tensorrt.__version__)')" \
    || die "TensorRT 파이썬이 없습니다. JetPack 6.x 를 설치한 젯슨인지 확인하세요 (sudo apt install nvidia-jetpack)."
python3 -c "import cv2" 2>/dev/null && ok "OpenCV $(python3 -c 'import cv2; print(cv2.__version__)')" \
    || die "OpenCV(cv2) 가 없습니다. sudo apt install python3-opencv"
[ -x /usr/src/tensorrt/bin/trtexec ] && ok "trtexec" || die "trtexec 가 없습니다 (/usr/src/tensorrt/bin/trtexec)."

echo
echo "[2/6] 필요한 파이썬 패키지"
need_apt=""
python3 -c "import pip" 2>/dev/null || need_apt="$need_apt python3-pip"
python3 -c "import pycuda.driver" 2>/dev/null || need_apt="$need_apt python3-pycuda"
python3 -c "import serial" 2>/dev/null || need_apt="$need_apt python3-serial"
python3 -c "import requests" 2>/dev/null || need_apt="$need_apt python3-requests"
if [ -n "$need_apt" ]; then
    echo "   설치합니다:$need_apt   (비밀번호를 물어보면 젯슨 비밀번호를 입력하세요)"
    sudo apt-get update -qq
    sudo apt-get install -y $need_apt || true
fi
if ! python3 -c "import pycuda.driver" 2>/dev/null; then
    echo "   apt 로 pycuda 를 못 받아 pip 로 설치합니다 (몇 분 걸림)"
    export PATH="/usr/local/cuda/bin:$PATH"
    python3 -m pip install --user pycuda || die "pycuda 설치 실패"
fi
for m in numpy pycuda.driver serial requests; do
    python3 -c "import $m" 2>/dev/null && ok "$m" || die "$m 를 불러올 수 없습니다"
done

echo
echo "[3/6] 장치 권한 (카메라·턴테이블)"
for g in video dialout; do
    if id -nG | tr ' ' '\n' | grep -qx "$g"; then
        ok "$g 그룹"
    else
        sudo usermod -aG "$g" "$USER" && warn "$g 그룹에 추가했습니다. 한 번 재부팅(또는 로그아웃)해야 적용됩니다."
    fi
done

echo
echo "[4/6] 기본 설정 파일"
[ -f database/settings.json ] && ok "database/settings.json (이미 있어 그대로 둠)" \
    || { cp database/settings.example.json database/settings.json; ok "database/settings.json 만듦"; }
[ -f database/products.json ] && ok "database/products.json (이미 있어 그대로 둠)" \
    || { cp database/products.default.json database/products.json; ok "database/products.json 만듦 (12종)"; }
chmod +x jetson/*.sh training/*.sh *.sh 2>/dev/null || true

echo
echo "[5/6] TensorRT 엔진 만들기"
if [ $ENGINES = 1 ]; then
    ./jetson/build_engines.sh
else
    echo "   건너뜀 (--skip-engines)"
fi

echo
echo "[6/6] 부팅 자동 시작"
MARK="# breadvision:$ROOT"
if [ $CRON = 1 ]; then
    ( crontab -l 2>/dev/null | grep -vF "$MARK" ; \
      echo "@reboot sleep 20 && $ROOT/jetson/start.sh >> $RUN_DIR/logs/boot.log 2>&1 $MARK" ) | crontab -
    ok "등록했습니다 — 젯슨을 켜면 20초 뒤 자동으로 켜집니다."
else
    echo "   건너뜀 (--no-cron)"
fi

if [ $START = 1 ]; then
    echo
    echo "지금 켭니다…"
    ./jetson/start.sh
else
    echo
    echo "설치 끝. 켜려면:  ./jetson/start.sh"
fi
