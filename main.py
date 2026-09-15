
import os
import sys

from PyQt5.QtWidgets import QApplication

# gui.main_window -> camera/detector -> cv2 순서로 import 된다.
# pip opencv-python 은 import 될 때 자기 번들 Qt(5.15.16) 플러그인 경로를
# QT_QPA_PLATFORM_PLUGIN_PATH 에 강제로 써 넣는다.
# 그대로 두면 PyQt5(Qt 5.15.3)가 버전이 다른 xcb 플러그인을 물어 죽는다.
# 지워서 PyQt5 가 자기 플러그인을 정상 경로에서 찾도록 한다.
from gui.main_window import MainWindow
from gui.app_errors import install_excepthook
from core.log import get_logger, setup_logging

# PC 에서 켰다면 젯슨을 찾아 환경변수를 대신 채워 준다.
# 젯슨 자신에서 켰을 때는 아무 일도 하지 않는다. (pc_link.py)
import pc_link

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

if __name__ == "__main__":
    log_path = setup_logging("pos")
    get_logger("app").info("BreadVision POS 시작 (로그: %s)", log_path)
    pc_link.autoconfigure()

    app = QApplication(sys.argv)
    # 버튼 처리 중 예상 못 한 오류가 나도 POS 가 꺼지지 않고 오류 창 + 로그로 남긴다
    install_excepthook("pos")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
