"""로그를 한 곳에서 설정한다.

왜 print 가 아니라 logging 인가
    예전에는 실패를 `except Exception: pass` 로 넘기거나 print 로만 찍었다.
    PC 의 POS 는 검은 창을 닫으면 그 내용이 사라지고, 무엇이 언제 실패했는지
    나중에 알 방법이 없었다. 로그 파일에 시각·위치·오류 내용(traceback)을 남긴다.

어디에 남나
    ~/.breadvision/logs/<앱이름>.log  (2MB 씩 3개까지 돌려 쓴다)
    BREADVISION_LOG_DIR 로 바꿀 수 있다. 콘솔(검은 창)에도 같이 나온다.
    BREADVISION_DEBUG=1 이면 자세한 로그(DEBUG)까지 남긴다.

setup_logging() 을 부르지 않은 모듈(테스트, 단독 실행)에서도 get_logger() 는 쓸 수 있다.
그때는 파이썬 기본 동작대로 WARNING 이상만 콘솔에 나온다.
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT_NAME = "breadvision"
_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"


def log_dir():
    return Path(os.environ.get("BREADVISION_LOG_DIR") or Path.home() / ".breadvision" / "logs")


def get_logger(name):
    """모듈별 로거. 예) get_logger("camera") → breadvision.camera"""
    return logging.getLogger(f"{ROOT_NAME}.{name}")


def setup_logging(app, to_file=True):
    """프로그램 시작 때 한 번 부른다. 로그 파일 경로를 돌려준다 (파일을 안 쓰면 None).

    to_file=False: 젯슨 서버처럼 start.sh 가 이미 표준출력을 로그 파일로 돌려 둔 경우.
    """
    root = logging.getLogger(ROOT_NAME)
    level = logging.DEBUG if os.environ.get("BREADVISION_DEBUG") == "1" else logging.INFO
    root.setLevel(level)
    if getattr(root, "_bv_configured", None):
        return root._bv_configured or None

    formatter = logging.Formatter(_FORMAT)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    path = None
    if to_file:
        path = log_dir() / f"{app}.log"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(path, maxBytes=2 * 1024 * 1024, backupCount=3,
                                          encoding="utf-8")
            handler.setFormatter(formatter)
            root.addHandler(handler)
        except OSError as e:
            root.warning("로그 파일을 만들지 못해 콘솔에만 남깁니다: %s (%s)", path, e)
            path = None

    root.propagate = False
    root._bv_configured = path or ""
    return path
