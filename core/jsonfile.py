"""설정 JSON 을 안전하게 읽고 쓴다.

예전 방식의 문제
    try: json.loads(...) except Exception: 설정 = {}
    → 파일이 깨졌을 때(쓰는 도중 전원이 꺼지는 등) 아무 말 없이 빈 설정으로 시작하고,
      곧바로 그 빈 설정을 저장해서 현장 설정(카메라·모델·앙상블)이 흔적도 없이 사라졌다.

지금 방식
    · 파일이 없으면: 조용히 기본값 (처음 설치한 경우라 정상)
    · 깨졌으면: <이름>.broken.bak 으로 복사해 두고 기록을 남긴 뒤 기본값
    · 쓸 때는 임시 파일에 다 쓴 다음 한 번에 바꿔치기한다 (쓰다 끊겨도 원본이 남는다)
"""
import json
import os
import shutil
from pathlib import Path

from core.log import get_logger

log = get_logger("jsonfile")


def read_json(path, default=None):
    """JSON 을 읽는다. 없거나 깨졌으면 default (깨진 파일은 백업한다)."""
    path = Path(path)
    fallback = {} if default is None else default
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        backup = path.with_name(path.name + ".broken.bak")
        try:
            shutil.copyfile(path, backup)
        except OSError:
            backup = None
        log.error("%s 을(를) 읽지 못해 기본값으로 시작합니다 (백업: %s): %s", path.name, backup, e)
        return fallback


def write_json(path, data):
    """JSON 을 저장한다. 다 쓴 뒤에 바꿔치기하므로 도중에 끊겨도 원본이 깨지지 않는다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
