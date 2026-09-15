"""카메라 열기·닫기·바꾸기와 화면 갱신 박자.

카메라는 POS 탭과 수집 탭이 함께 쓴다. 누가 켜든 끄든 여기 하나를 거친다.
    frame_ready(frame, now, draw)  새 프레임이 왔다. draw=True 면 지금 그려도 된다
    started(label) / stopped()     카메라가 켜졌다 / 꺼졌다
"""
import os
import time

import cv2
from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMessageBox

from camera.camera import CameraManager, _backends
from core.camera_source import (camera_key_of, camera_source_label, jetson_stream_url,
                                resolve_camera_source)
from core.log import get_logger

log = get_logger("camera")


class CameraController(QObject):
    frame_ready = pyqtSignal(object, float, bool)
    started = pyqtSignal(str)
    stopped = pyqtSignal()

    # 화면 그리기가 느린 환경(SSH X11 포워딩은 프레임당 100ms 가까이 든다)에서
    # 매 프레임 그리면 UI 스레드가 꽉 차서 버튼이 안 눌린다.
    # 타이머가 제때 도는지(=루프가 밀렸는지)를 보고 화면 갱신 간격을 스스로 조절한다.
    INTERVAL = 0.033            # 카메라 타이머 주기 (초)
    DISPLAY_MIN_GAP = 0.03      # 최소 간격 — 약 30fps
    DISPLAY_MAX_GAP = 2.0       # 최대 간격 — SSH X11 처럼 한 장에 0.5초씩 걸리는 환경

    def __init__(self, ctx, live=None, camera_factory=CameraManager, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.live = live
        self._factory = camera_factory
        self.camera = camera_factory(self.resolve(ctx.settings.get("camera_source", "")), use_env=False)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)

        self.frame = None           # 가장 최근 원본 프레임 (인식·수집에 쓴다)
        self.last_frame_id = -1
        self.last_display_ts = 0.0
        self.last_tick_ts = 0.0
        self.display_gap = self.DISPLAY_MIN_GAP

        # BREADVISION_DEBUG=1 로 실행하면 2초마다 실제 성능을 로그에 남긴다
        self.debug_perf = os.environ.get("BREADVISION_DEBUG", "") == "1"
        self._dbg = {"t0": time.perf_counter(), "frames": 0, "drawn": 0, "lag": 0.0}

    # ── 카메라 고르기 ────────────────────────────────────────────────────
    def stream_url(self):
        return jetson_stream_url(self.ctx.paths.base)

    def resolve(self, key):
        return resolve_camera_source(key, self.stream_url())

    def current_key(self):
        return camera_key_of(self.camera.index, self.stream_url())

    def source_label(self):
        return camera_source_label(self.camera.index, self.stream_url())

    def is_running(self):
        return self.timer.isActive()

    # ── 켜기·끄기 ────────────────────────────────────────────────────────
    def start(self):
        """카메라를 연다. 실패하면 예외를 그대로 올린다 (start_with_message 참고)."""
        if self.camera.cap is None:
            self.camera.start()
        if not self.timer.isActive():
            self.timer.start(int(self.INTERVAL * 1000))
        self.started.emit(self.source_label())

    def start_with_message(self, parent):
        try:
            self.start()
            return True
        except (RuntimeError, OSError, cv2.error) as e:
            log.warning("카메라를 열지 못했습니다: %s (%s)", self.source_label(), e)
            QMessageBox.critical(
                parent, "카메라 연결 실패",
                f"{self.source_label()} 을(를) 열지 못했습니다.\n\n{e}\n\n"
                "POS 화면 아래 카메라 목록에서 다른 카메라를 골라 보세요.\n"
                "(젯슨 카메라가 안 되면 젯슨이 켜져 있는지, 같은 네트워크인지 확인)"
            )
            return False

    def stop(self):
        self.timer.stop()
        self.camera.stop()
        self.stopped.emit()

    def switch(self, key):
        """다른 카메라로 바꾸고 선택을 기억한다. (켜기는 부르는 쪽이 한다)"""
        src = self.resolve(key)
        self.stop()
        self.camera = self._factory(src, use_env=False)
        self.ctx.settings["camera_source"] = key
        self.ctx.save_settings()

    def scan_local(self, max_index=5):
        """이 컴퓨터에 연결된 카메라 번호를 찾는다. {번호: "1280x720"}

        같은 카메라를 두 번 열 수 없으므로, 지금 PC 카메라를 쓰는 중이면 잠시 끈다.
        돌아올 때 다시 켜야 하는지(was_local)를 같이 돌려준다.
        """
        was_local = isinstance(self.camera.index, int) and self.camera.cap is not None
        if was_local:
            self.stop()
        # 번호마다 백엔드를 다 시도하면 윈도우에서 한참 걸린다. 그 OS 의 첫 방식만 쓴다.
        _name, api = next((n, a) for n, a in _backends() if not isinstance(a, str))
        found = {}
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for i in range(max_index):
                cap = cv2.VideoCapture(i, api)
                ok = cap.isOpened() and cap.read()[0]
                if ok:
                    found[i] = (f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                                f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                cap.release()
        finally:
            QApplication.restoreOverrideCursor()
        return found, was_local

    # ── 박자 ────────────────────────────────────────────────────────────
    def tick(self):
        # 이벤트 루프가 밀렸는지 먼저 잰다. 반드시 맨 앞에서 재야 한다.
        # 아래 early return 뒤에서 재면 "새 프레임이 오는 간격" 을 재게 되어,
        # 15fps 카메라를 쓰면 멀쩡한데도 밀린 걸로 오판한다.
        now = time.perf_counter()
        lag = 0.0
        if self.last_tick_ts:
            lag = now - self.last_tick_ts
            if lag > self.INTERVAL * 2.5:
                self.display_gap = min(self.display_gap * 1.3, self.DISPLAY_MAX_GAP)
            else:
                self.display_gap = max(self.display_gap * 0.85, self.DISPLAY_MIN_GAP)
        self.last_tick_ts = now
        if self.debug_perf:
            self._debug_tick(now, lag)

        ok, frame, frame_id = self.camera.read_latest()
        if not ok or frame_id == self.last_frame_id:
            return                      # 아직 새 프레임이 없다. 그릴 필요도 없다.
        self.last_frame_id = frame_id
        self._dbg["frames"] += 1
        self.frame = frame.copy()       # 손대지 않은 원본은 인식·수집용으로 둔다

        draw = now - self.last_display_ts >= self.display_gap
        if draw:
            self.last_display_ts = now
            self._dbg["drawn"] += 1
        self.frame_ready.emit(frame, now, draw)

    def _debug_tick(self, now, lag):
        d = self._dbg
        d["lag"] = max(d["lag"], lag * 1000)
        span = now - d["t0"]
        if span < 2.0:
            return
        count, ms = self.live.take_perf() if self.live else (0, 0.0)
        avg = ms / count if count else 0.0
        log.info("[성능] 카메라 %5.1f fps | 화면 %5.1f fps | 추론 %4.1f fps (%4.0f ms) "
                 "| 타이머 최대지연 %5.0f ms | 갱신간격 %4.0f ms",
                 d["frames"] / span, d["drawn"] / span, count / span, avg,
                 d["lag"], self.display_gap * 1000)
        self._dbg = {"t0": now, "frames": 0, "drawn": 0, "lag": 0.0}
