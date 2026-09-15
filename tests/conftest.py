"""테스트 공통 설정.

카메라·GPU·젯슨·인터넷 없이 도는 테스트만 둔다. PC 에서 `pytest` 한 줄로 끝난다.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 화면 없는 환경(GitHub Actions 등)에서도 Qt 위젯을 만들 수 있게 한다.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# pip opencv 가 심는 Qt 플러그인 경로를 지운다 (main.py 와 같은 이유).
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
# 테스트 중에 젯슨을 찾거나 원격 추론으로 붙지 않게 한다.
for key in ("BREADVISION_INFER_URL", "BREADVISION_CAMERA", "BREADVISION_JETSON_STREAM"):
    os.environ.pop(key, None)


def det(cid, conf, xyxy):
    """테스트용 검출 한 개."""
    return {"class_id": cid, "confidence": conf, "xyxy": list(xyxy)}
