"""core/jsonfile.py — 설정 파일이 깨져도 현장 설정이 조용히 사라지지 않는가."""
import json

import jetson_ops
from core.jsonfile import read_json, write_json


def test_missing_file_returns_default(tmp_path):
    assert read_json(tmp_path / "none.json") == {}
    assert read_json(tmp_path / "none.json", default={"a": 1}) == {"a": 1}


def test_broken_file_is_backed_up(tmp_path):
    f = tmp_path / "settings.json"
    f.write_text('{"active_engine": "models/m13_A.engine", ', encoding="utf-8")   # 쓰다 끊긴 파일
    assert read_json(f) == {}
    backup = tmp_path / "settings.json.broken.bak"
    assert backup.read_text(encoding="utf-8").startswith('{"active_engine"')


def test_write_json_roundtrip_and_no_temp_left(tmp_path):
    f = tmp_path / "sub" / "products.json"
    write_json(f, {"0": {"class_name": "salt", "display_name": "소금빵", "price": 3000}})
    assert json.loads(f.read_text(encoding="utf-8"))["0"]["display_name"] == "소금빵"
    assert [p.name for p in f.parent.iterdir()] == ["products.json"]


def test_jetson_remember_ensemble_keeps_other_settings(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"active_engine": "models/m13_A.engine", "camera_source": "jetson"}),
                        encoding="utf-8")
    monkeypatch.setattr(jetson_ops, "SETTINGS", settings)
    jetson_ops.remember_ensemble(["models/m13_B.engine"])
    data = json.loads(settings.read_text(encoding="utf-8"))
    assert data == {"active_engine": "models/m13_A.engine", "camera_source": "jetson",
                    "ensemble_engines": ["models/m13_B.engine"]}
