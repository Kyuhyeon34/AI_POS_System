"""QR 인식을 GUI 스레드 밖에서 돌린다.

실측(640x480): cv2.QRCodeDetector 는 한 장에 평균 27ms, 최대 72ms 걸린다.
이걸 화면 갱신 타이머 안에서 부르면 그 시간 동안 화면도 버튼도 멈춘다.
6프레임마다 한 번만 불러도 초당 5번, 매번 27ms 씩 끊긴다.

추론과 같은 방식으로 전담 스레드에 넘긴다. 대기열은 한 칸이라 밀린 프레임은 버린다.
QR 은 손에 든 휴대폰을 읽는 것이라 몇 프레임 늦어도 아무 차이가 없다.
"""
import threading

import cv2
from PyQt5.QtCore import QThread, pyqtSignal


class QRWorker(QThread):
    result = pyqtSignal(str, object)      # 읽은 값(없으면 ""), 좌표

    def __init__(self, fast_width=800, parent=None):
        super().__init__(parent)
        self.fast_width = int(fast_width)
        self._detector = None             # 스레드 안에서 만든다
        self._cv = threading.Condition()
        self._pending = None
        self._stop = False
        self.busy = False                 # GUI 쪽에서 중복 제출을 막을 때 본다

    def submit(self, frame):
        with self._cv:
            self._pending = frame
            self.busy = True
            self._cv.notify()

    def shutdown(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self.wait(2000)

    def run(self):
        self._detector = cv2.QRCodeDetector()
        while True:
            with self._cv:
                while not self._stop and self._pending is None:
                    self._cv.wait()
                if self._stop:
                    return
                frame, self._pending = self._pending, None

            data, points = "", None
            try:
                img = frame
                scale = 1.0
                h, w = img.shape[:2]
                if w > self.fast_width:
                    scale = self.fast_width / float(w)
                    img = cv2.resize(img, (self.fast_width, int(h * scale)),
                                     interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                d, pts, _ = self._detector.detectAndDecode(gray)
                if d:
                    data = d
                    if pts is not None and scale != 1.0:
                        pts = pts / scale      # 좌표를 원본 크기로 되돌린다
                    points = pts
            except Exception:
                # QR 은 없어도 되는 기능이다. 실패하면 그냥 없는 것으로 둔다.
                data, points = "", None

            self.busy = False
            self.result.emit(data, points)
