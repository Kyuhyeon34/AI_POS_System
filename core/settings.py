"""POS 설정 (database/settings.json)."""
import copy
from pathlib import Path

from core.jsonfile import read_json, write_json

DEFAULTS = {
    "training_base_model": "models/m12_A.pt",
    "active_engine": "",
    "active_model_classes": {},
}


def load_settings(path):
    """설정을 읽고 옛 키를 옮긴 뒤, 빠진 기본값을 채워 바로 저장한다.

    settings.json 이 없으면 settings.example.json 을 시작점으로 쓴다.
    파일이 깨졌으면 core/jsonfile 이 .broken.bak 으로 백업해 두므로 조용히 사라지지 않는다.
    """
    path = Path(path)
    source = path if path.exists() else path.with_name("settings.example.json")
    raw = read_json(source)
    if not isinstance(raw, dict):
        raw = {}

    # v7 migration: 예전에는 active_model 하나에 .pt/.engine 을 같이 썼다
    old_active = raw.get("active_model", "")
    if old_active and old_active.lower().endswith(".pt"):
        raw.setdefault("training_base_model", old_active)
    elif old_active and old_active.lower().endswith(".engine"):
        raw.setdefault("active_engine", old_active)

    for k, v in DEFAULTS.items():
        raw.setdefault(k, copy.deepcopy(v))

    write_json(path, raw)
    return raw


def save_settings(path, settings):
    write_json(path, settings)
