"""실시간 인식 — 대기 중에도 화면의 빵을 찾아 둔다. 주문 반영은 하지 않는다.

POS 탭(초록 박스 표시)과 수집 탭(자동 라벨 후보)이 같은 결과를 함께 쓴다.

프레임을 추론 스레드에 던지기만 하고 즉시 돌아온다. 결과는 on_result 로 나중에 온다.
간격은 직전 추론에 걸린 시간의 2배로 둔다. 느린 환경에서는 알아서 뜸해지고,
빠른 환경이라도 GPU 를 절반 이상 쓰지 않는다.
"""
import time

from PyQt5.QtCore import QObject


class LiveDetection(QObject):
    # 실시간 추론 최소 간격 (초) — 최대 10회/초.
    # 카메라 30fps 를 전부 추론하지 않는다. GPU 를 다 쓰면 화면·QR·수집이 같이 느려진다.
    MIN_GAP = 0.10
    MAX_BACKOFF = 5.0

    def __init__(self, detector, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.dets = []              # 가장 최근 결과
        self.busy = False           # 추론 스레드가 아직 처리 중인가
        self.gap = 0.25             # 추론 간격 (초). 실제 걸린 시간에 맞춰 조절된다
        self.next_ts = 0.0          # 다음 추론 허용 시각
        self.perf_count = 0         # BREADVISION_DEBUG 성능 출력용
        self.perf_ms = 0.0
        detector.live_ready.connect(self.on_result)
        detector.failed.connect(self.on_failed)

    def update(self, frame, frame_id, now):
        if not self.detector.ready() or now < self.next_ts:
            return
        if self.busy:
            return                  # 아직 앞엣것을 처리 중이다. 밀어 넣지 않는다.
        self.busy = True
        self.next_ts = now + self.gap
        self.detector.submit_live(frame.copy(), frame_id)

    def clear(self):
        self.dets = []

    def on_result(self, dets, frame_id, took):
        """추론 스레드가 결과를 보내왔다. (GUI 스레드에서 실행된다)"""
        self.busy = False
        self.dets = dets
        self.gap = max(self.MIN_GAP, took * 2)
        self.perf_count += 1
        self.perf_ms += took * 1000.0

    def on_failed(self, message):
        """원격 서버가 끊기는 등. 화면만 비우고 잠시 뜸하게 재시도한다."""
        self.busy = False
        self.dets = []
        self.gap = min(self.gap * 2, self.MAX_BACKOFF)
        self.next_ts = time.perf_counter() + self.gap

    def take_perf(self):
        count, ms = self.perf_count, self.perf_ms
        self.perf_count, self.perf_ms = 0, 0.0
        return count, ms
