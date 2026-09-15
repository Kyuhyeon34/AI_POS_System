"""어느 카메라를 열지 정한다 (젯슨 영상 주소 / PC 카메라 번호 / 직접 넣은 주소).

settings.json 의 camera_source 에는 사람이 고른 "키" 를 저장한다.
    "jetson"  젯슨이 내보내는 영상(8090)
    "0", "1"  PC 카메라 번호
    그 밖     직접 넣은 주소 (http://…/stream, rtsp://…)
"""
import os
from pathlib import Path
from urllib.parse import urlparse

from core.jsonfile import read_json

DEFAULT_STREAM_PORT = 8090


def is_jetson_device():
    return Path("/etc/nv_tegra_release").exists()


def _stream_port(base_dir):
    cfg = read_json(Path(base_dir) / "jetson.json")
    try:
        return int(cfg.get("stream_port", DEFAULT_STREAM_PORT)) if isinstance(cfg, dict) else DEFAULT_STREAM_PORT
    except (TypeError, ValueError):
        return DEFAULT_STREAM_PORT


def jetson_stream_url(base_dir, environ=None, on_jetson=None):
    """젯슨이 내보내는 카메라 영상 주소. 모르면 None."""
    env = os.environ if environ is None else environ
    url = env.get("BREADVISION_JETSON_STREAM", "").strip()
    if url:
        return url
    cam = env.get("BREADVISION_CAMERA", "").strip()
    if cam.startswith("http"):
        return cam
    port = _stream_port(base_dir)
    infer = env.get("BREADVISION_INFER_URL", "").strip()
    if infer:
        host = urlparse(infer).hostname
        if host:
            return f"http://{host}:{port}/stream"
    if is_jetson_device() if on_jetson is None else on_jetson:     # 젯슨 자신에서 켠 경우
        return f"http://127.0.0.1:{port}/stream"
    return None


def resolve_camera_source(key, stream_url, environ=None):
    """저장된 선택(key) → CameraManager 에 줄 값 (카메라 번호 또는 영상 주소)."""
    env = os.environ if environ is None else environ
    key = str(key or "").strip()
    if key == "jetson":
        return stream_url or 0
    if key.isdigit():
        return int(key)
    if key:
        return key                                    # 직접 넣은 주소
    cam = env.get("BREADVISION_CAMERA", "").strip()   # 고른 적이 없으면 예전 방식
    if cam:
        return int(cam) if cam.isdigit() else cam
    return 0


def camera_key_of(src, stream_url):
    if isinstance(src, int):
        return str(src)
    return "jetson" if src == stream_url else str(src)


def camera_source_label(src, stream_url):
    if isinstance(src, int):
        return f"PC 카메라 {src}"
    if src == stream_url:
        return "젯슨 카메라"
    return str(src)
