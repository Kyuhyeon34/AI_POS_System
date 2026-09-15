"""예상하지 못한 오류가 나도 POS 가 통째로 꺼지지 않게 한다.

PyQt5 는 버튼 처리 함수(slot) 안에서 처리되지 않은 예외가 나면
기본 설정에서 프로그램 전체를 종료한다(qFatal). 계산 중에 POS 가 꺼지면 곤란하다.

sys.excepthook 을 바꿔 두면 PyQt5 는 종료하지 않고 이 함수를 부른다.
여기서 오류를 로그 파일에 남기고, 사용자에게는 무슨 일이 있었는지 창으로 알린다.
"""
import sys

from core.log import get_logger, log_dir

log = get_logger("app")


def install_excepthook(app_name="pos"):
    showing = {"now": False}          # 오류 창을 띄우다 또 오류가 나면 창이 끝없이 뜬다

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("처리되지 않은 오류", exc_info=(exc_type, exc, tb))
        if showing["now"]:
            return
        try:
            from PyQt5.QtWidgets import QApplication, QMessageBox
            if QApplication.instance() is None:
                return
            showing["now"] = True
            QMessageBox.critical(
                None, "예상하지 못한 오류",
                f"{exc_type.__name__}: {exc}\n\n"
                "프로그램은 계속 쓸 수 있습니다. 같은 오류가 반복되면 아래 로그를 확인하세요.\n"
                f"{log_dir() / (app_name + '.log')}")
        except Exception:                 # noqa: BLE001 - 오류 창조차 못 띄우면 로그만 남긴다
            log.exception("오류 창을 띄우지 못했습니다")
        finally:
            showing["now"] = False

    sys.excepthook = hook
    return hook
