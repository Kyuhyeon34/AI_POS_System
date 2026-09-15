"""카메라 미리보기 라벨. POS 탭과 수집 탭이 같은 것을 쓴다."""
import os

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QLabel

from core.log import get_logger

log = get_logger("view")


def decide_preview_max():
    """미리보기 그림의 최대 변 길이(px). 0 이면 제한 없이 라벨을 꽉 채운다.

    SSH X11 로 보고 있다면 그림 크기가 곧 전송량이다.
    전송량 = 가로 x 세로 x 4 x fps 이고, 젯슨 랜선은 100Mbps = 12.5 MB/s 다.
    실측(892x503 라벨을 꽉 채웠을 때):

        제한 없음  670x503  40 MB/s  → 약 9 fps
        640        640x480  32 MB/s  → 약 12 fps
        480        480x360  19 MB/s  → 30 fps

    느리면 BREADVISION_PREVIEW_MAX=640 처럼 줄이면 된다.
    """
    env = os.environ.get("BREADVISION_PREVIEW_MAX", "").strip()
    if env:
        try:
            return max(0, int(env))
        except ValueError:
            log.warning("BREADVISION_PREVIEW_MAX 가 숫자가 아닙니다: %s", env)

    disp = os.environ.get("DISPLAY", "")
    host, _, _num = disp.rpartition(":")
    if disp and host not in ("", "unix"):
        log.info("원격 X 화면(SSH X11)입니다. 미리보기가 클수록 느려집니다. "
                 "느리면: BREADVISION_PREVIEW_MAX=640 (약 12fps) 또는 480 (30fps)")
    return 0


class CameraView(QLabel):
    def __init__(self, placeholder, min_width, min_height, preview_max=0, style=None, parent=None):
        super().__init__(placeholder, parent)
        self.preview_max = preview_max
        self._area = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(min_width, min_height)
        if style:
            self.setStyleSheet(style)

    def area(self):
        """라벨이 레이아웃에서 원래 차지하는 크기 (가로, 세로).

        원격 화면에서는 라벨을 그림 크기로 고정해 버리므로, 그 뒤에 width() 를 읽으면
        줄어든 값이 나와 화면이 점점 작아진다. 처음(레이아웃이 준) 크기를 기억해 둔다.
        창 크기가 바뀌면 MainWindow.resizeEvent 가 forget_area() 로 지운다.

        이 라벨 자신의 resizeEvent 에서 지우면 안 된다 — 그림 크기로 고정하는 순간에도
        resizeEvent 가 오므로, 줄어든 크기를 다시 기억해 버린다.
        """
        if self._area is None:
            area = (self.width(), self.height())
            if area[0] > 0 and area[1] > 0:
                self._area = area
            return area
        return self._area

    def forget_area(self):
        self._area = None
        self.reset_size()

    def reset_size(self):
        """카메라를 끄면 라벨 고정을 풀어 안내 문구가 제자리에 보이게 한다."""
        if not self.preview_max:
            return
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)

    def show_frame(self, frame, guide=False):
        # Qt 의 SmoothTransformation 축소보다 cv2.resize 가 두 배 이상 빠르다.
        h0, w0 = frame.shape[:2]
        aw, ah = self.area()

        if self.preview_max:
            # 원격 화면(SSH X11 등): 라벨을 그림 크기에 딱 맞춰 고정하는 게 핵심이다.
            # 라벨이 그림보다 크면 Qt 는 매 프레임 라벨 둘레까지 다시 그리고 그 픽셀이 전부 회선을 탄다.
            #   실측(892x503 라벨 안에 640x480 그림): 213 MB/s → 고정하면 640px 32 MB/s
            aw = min(aw, self.preview_max)
            ah = min(ah, self.preview_max)

        if aw > 0 and ah > 0:
            # 라벨을 꽉 채운다 (가로세로 비율 유지). 키울 때 LINEAR, 줄일 때 AREA.
            r = min(aw / float(w0), ah / float(h0))
            if abs(r - 1.0) > 0.01:
                nw, nh = max(1, int(w0 * r)), max(1, int(h0 * r))
                interp = cv2.INTER_LINEAR if r > 1.0 else cv2.INTER_AREA
                frame = cv2.resize(frame, (nw, nh), interpolation=interp)

        pix = _to_pixmap(frame)

        if guide:
            painter = QPainter(pix)
            painter.setPen(QPen(QColor(230, 45, 45), 4))
            pw, ph = pix.width(), pix.height()
            bw, bh = int(pw * 0.45), int(ph * 0.45)
            x1, y1 = (pw - bw) // 2, (ph - bh) // 2
            painter.drawRect(x1, y1, bw, bh)
            painter.setFont(QFont("Arial", 13, QFont.Bold))
            painter.drawText(x1, max(25, y1 - 10), "PLACE ONE BREAD INSIDE THIS BOX")
            painter.end()

        # 원격 화면일 때만 라벨을 그림 크기에 딱 맞춘다. 로컬 화면에서 고정하면
        # 라벨이 쪼그라들어 레이아웃이 남는 자리를 다른 위젯에 줘 버리고 다시는 커지지 않는다.
        if self.preview_max and (self.width() != pix.width() or self.height() != pix.height()):
            self.setFixedSize(pix.width(), pix.height())
        self.setPixmap(pix)

    def show_scaled(self, frame):
        """한 장을 라벨 크기에 맞춰 보여준다 (「상품 인식 실행」 결과)."""
        if frame is None:
            return
        self.setPixmap(_to_pixmap(frame).scaled(self.size(), Qt.KeepAspectRatio,
                                                Qt.SmoothTransformation))


def _to_pixmap(frame):
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())
