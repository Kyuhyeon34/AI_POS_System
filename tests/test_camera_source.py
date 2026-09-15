"""core/camera_source.py — 어느 카메라를 열지."""
import json

from core.camera_source import (camera_key_of, camera_source_label, jetson_stream_url,
                                resolve_camera_source)


def test_stream_url_priority(tmp_path):
    assert jetson_stream_url(tmp_path, {"BREADVISION_JETSON_STREAM": "http://a/stream"}) == "http://a/stream"
    assert jetson_stream_url(tmp_path, {"BREADVISION_CAMERA": "http://b/stream"}) == "http://b/stream"
    env = {"BREADVISION_INFER_URL": "http://192.168.0.23:8092"}
    assert jetson_stream_url(tmp_path, env, on_jetson=False) == "http://192.168.0.23:8090/stream"
    assert jetson_stream_url(tmp_path, {}, on_jetson=True) == "http://127.0.0.1:8090/stream"
    assert jetson_stream_url(tmp_path, {}, on_jetson=False) is None


def test_stream_port_from_jetson_json(tmp_path):
    (tmp_path / "jetson.json").write_text(json.dumps({"stream_port": 9000}), encoding="utf-8")
    env = {"BREADVISION_INFER_URL": "http://10.0.0.5:8092"}
    assert jetson_stream_url(tmp_path, env) == "http://10.0.0.5:9000/stream"


def test_resolve_camera_source():
    url = "http://j/stream"
    assert resolve_camera_source("jetson", url, {}) == url
    assert resolve_camera_source("jetson", None, {}) == 0
    assert resolve_camera_source("1", url, {}) == 1
    assert resolve_camera_source("rtsp://cam", url, {}) == "rtsp://cam"
    assert resolve_camera_source("", url, {"BREADVISION_CAMERA": "2"}) == 2
    assert resolve_camera_source(None, url, {}) == 0


def test_key_and_label():
    url = "http://j/stream"
    assert camera_key_of(0, url) == "0"
    assert camera_key_of(url, url) == "jetson"
    assert camera_key_of("rtsp://x", url) == "rtsp://x"
    assert camera_source_label(1, url) == "PC 카메라 1"
    assert camera_source_label(url, url) == "젯슨 카메라"
