"""POS 메인 창 — 탭을 조립하고 카메라 프레임을 보이는 탭에 나눠 준다.

예전에는 4,143줄짜리 MainWindow 클래스 하나에 모든 기능이 있었다.
지금은 탭마다 독립 위젯이고, 여기서는 조립과 연결만 한다.

    gui/context.py            탭들이 함께 쓰는 상태 (상품·설정·추론기·젯슨 창구) + 변경 신호
    gui/camera_controller.py  카메라 켜기·끄기·바꾸기, 화면 갱신 박자
    gui/live_detection.py     실시간 인식 (POS 와 수집 탭이 결과를 함께 쓴다)
    gui/tabs/pos_tab.py       POS: 인식·주문·결제 (+ tabs/member_qr.py 회원 QR)
    gui/tabs/product_tab.py   상품 관리
    gui/tabs/collect_tab.py   데이터 수집 (규칙은 core/collect_labeler.py)
    gui/tabs/train_tab.py     모델 배포·학습 (+ gui/remote_jobs.py 젯슨 작업)
    gui/tabs/member_tab.py    회원 관리 안내
    core/                     Qt 없이 도는 계산 (tests/ 에서 검증)
"""
import os

from PyQt5.QtWidgets import QMainWindow, QTabWidget

from ai.factory import make_detector
from ai.ops_client import OpsClient
from ai.worker import DetectorWorker
from camera.camera import CameraManager
from core.log import get_logger
from gui.camera_controller import CameraController
from gui.camera_view import decide_preview_max
from gui.context import AppContext
from gui.live_detection import LiveDetection
from gui.qr_worker import QRWorker
from gui.style import STYLESHEET, load_fonts
from gui.tabs.collect_tab import CollectTab
from gui.tabs.member_tab import MemberTab
from gui.tabs.pos_tab import PosTab
from gui.tabs.product_tab import ProductTab
from gui.tabs.train_tab import TrainTab
from membership.supabase_api import SupabaseMembershipAPI

log = get_logger("app")

_AUTO = object()
QR_FAST_WIDTH = 800         # QR 검사는 이 폭으로 줄여서 본다 (약 29ms)


def connect_jetson_ops():
    """젯슨에 작업(분할·학습·변환)을 시킬 수 있는가. PC 에서 젯슨에 연결됐을 때만 창구를 만든다.

    업체는 PC 만 만진다 — 버튼을 누르면 젯슨이 대신 실행한다.
    젯슨에서 GUI 를 직접 켰을 때는 None 이고, 로컬에서 실행한다.
    """
    url = os.environ.get("BREADVISION_INFER_URL", "").strip()
    if not url:
        return None
    client = OpsClient(url)
    client.refresh()
    if client.available:
        log.info("젯슨 작업 원격 실행 사용: %s", url)
        return client
    log.warning("%s 서버가 작업 실행을 지원하지 않습니다 (젯슨의 inference_server.py 를 최신본으로 다시 띄우세요)", url)
    return None


class MainWindow(QMainWindow):
    def __init__(self, detector=None, qr_worker=None, camera_factory=CameraManager,
                 ops=_AUTO, membership_api=None, paths=None):
        """인자를 비워 두면 실제 장비를 쓴다. 테스트에서는 가짜 부품을 넣는다."""
        super().__init__()
        load_fonts()                              # 위젯에 스타일을 붙이기 전에 등록해야 Jua 가 적용된다
        self.setWindowTitle("BreadVision AI Bakery POS")
        self.resize(1560, 940)

        if detector is None:
            # 추론은 전담 스레드에서 돈다. GUI 는 프레임만 던지고 바로 돌아온다 (ai/worker.py)
            detector = DetectorWorker(make_detector)
            detector.boot()
        if qr_worker is None:
            qr_worker = QRWorker(QR_FAST_WIDTH)   # QR 도 같은 이유로 스레드에서 (gui/qr_worker.py)
            qr_worker.start()
        self.detector = detector
        self.qr_worker = qr_worker

        self.ctx = AppContext(detector,
                              ops=connect_jetson_ops() if ops is _AUTO else ops,
                              membership_api=membership_api or SupabaseMembershipAPI(),
                              paths=paths, parent=self)
        self.live = LiveDetection(detector, self)
        self.camera = CameraController(self.ctx, self.live, camera_factory, self)
        preview_max = decide_preview_max()

        self.pos_tab = PosTab(self.ctx, self.camera, self.live, qr_worker, preview_max)
        self.product_tab = ProductTab(self.ctx)
        self.collect_tab = CollectTab(self.ctx, self.camera, self.live, preview_max)
        self.train_tab = TrainTab(self.ctx)
        self.member_tab = MemberTab()

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self.pos_tab, "POS")
        self.tabs.addTab(self.product_tab, "상품 관리")
        self.tabs.addTab(self.collect_tab, "데이터 수집")
        self.tabs.addTab(self.train_tab, "모델 배포 및 모델 학습")
        self.tabs.addTab(self.member_tab, "회원 관리")

        self.ctx.status.connect(self._show_status)
        self.camera.frame_ready.connect(self._on_frame)
        self.product_tab.product_registered.connect(self.collect_tab.select_product)

        self.setStyleSheet(STYLESHEET)
        self.train_tab.try_load_saved_engine()

    def _show_status(self, text, msec):
        if text:
            self.statusBar().showMessage(text, msec)
        else:
            self.statusBar().clearMessage()

    def _on_frame(self, frame, now, draw):
        """새 프레임. QR 스캔은 매 프레임, 그리기는 보이는 탭만 (보이지 않는 탭까지 그릴 이유가 없다)."""
        current = self.tabs.currentWidget()
        self.pos_tab.on_new_frame(frame, now, current is self.pos_tab)
        if not draw:
            return
        if current is self.pos_tab:
            self.pos_tab.render(frame, now)
        elif current is self.collect_tab:
            self.collect_tab.render(frame, now)

    def resizeEvent(self, event):
        # 창이 바뀌면 미리보기 라벨이 차지할 크기도 다시 재야 한다.
        for view in (self.pos_tab.view, self.collect_tab.view):
            view.forget_area()
        super().resizeEvent(event)

    def closeEvent(self, event):
        self.camera.stop()
        self.collect_tab.shutdown()
        self.train_tab.shutdown()
        self.detector.shutdown()
        self.qr_worker.shutdown()
        event.accept()
