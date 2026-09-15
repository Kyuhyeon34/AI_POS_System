"""POS 창 전체를 가짜 부품(카메라·추론기·QR·Supabase)으로 띄워 주요 흐름을 눌러 본다.

실제 카메라·젯슨·인터넷은 없다. 대신 "화면이 뜨는가, 버튼을 누르면 계산이 맞는가,
탭끼리 신호가 제대로 이어지는가" 를 확인한다. 실제 장비 확인은 사람이 해야 한다.
"""
import json

import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import QObject, pyqtSignal  # noqa: E402
from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402

from conftest import det  # noqa: E402

PRODUCTS = {
    "0": {"class_name": "salt", "display_name": "소금빵", "price": 3000},
    "1": {"class_name": "pie", "display_name": "파이", "price": 4500},
}


class FakeDetector(QObject):
    live_ready = pyqtSignal(object, int, float)
    failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.model_path = None
        self.dets = []
        self.loaded = []
        self.shut = False

    def ready(self):
        return True

    def load(self, path):
        self.loaded.append(path)
        self.model_path = path

    def refresh(self):
        return False

    def infer(self, frame, draw=True, tta=False):
        return frame, list(self.dets), {}

    def submit_live(self, frame, frame_id):
        self.live_ready.emit(list(self.dets), frame_id, 0.01)

    def shutdown(self):
        self.shut = True


class FakeQR(QObject):
    result = pyqtSignal(str, object)

    def __init__(self):
        super().__init__()
        self.busy = False
        self.next = ""
        self.shut = False

    def submit(self, frame):
        self.result.emit(self.next, None)

    def shutdown(self):
        self.shut = True


class FakeCamera:
    def __init__(self, index=0, use_env=True):
        self.index = index
        self.cap = None
        self.n = 0

    def start(self):
        self.cap = object()

    def stop(self):
        self.cap = None

    def read_latest(self):
        if self.cap is None:
            return False, None, self.n
        self.n += 1
        return True, np.full((480, 640, 3), 30, np.uint8), self.n


class FakeMembership:
    def __init__(self):
        self.payments = []
        self.fail = False

    def find_member_by_qr_payload(self, data):
        return {"id": "u1", "name": "홍길동", "point": 100} if data == "BV1:ok" else None

    def complete_payment(self, member_id, total):
        if self.fail:
            raise RuntimeError("network down")
        self.payments.append((member_id, total))
        return {"earned_point": int(total * 0.03), "current_point": 100 + int(total * 0.03)}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialogs(monkeypatch):
    """메시지 창이 뜨면 테스트가 멈추므로, 무엇이 떴는지 기록만 한다."""
    shown = []

    def record(kind):
        def fn(parent, title, text="", *a, **k):
            shown.append((kind, title, text))
            return QMessageBox.Yes
        return fn

    for kind in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(record(kind)))
    return shown


@pytest.fixture
def win(qapp, tmp_path, dialogs):
    from gui.context import Paths
    from gui.main_window import MainWindow

    (tmp_path / "database").mkdir()
    (tmp_path / "database" / "products.default.json").write_text(
        json.dumps(PRODUCTS, ensure_ascii=False), encoding="utf-8")
    w = MainWindow(detector=FakeDetector(), qr_worker=FakeQR(), camera_factory=FakeCamera,
                   ops=None, membership_api=FakeMembership(), paths=Paths(base=tmp_path))
    yield w
    w.close()


def tick(w, n=1):
    for _ in range(n):
        w.camera.last_display_ts = 0.0          # 매번 그리게
        w.camera.tick()


def test_window_has_five_tabs_in_order(win):
    assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == \
        ["POS", "상품 관리", "데이터 수집", "모델 배포 및 모델 학습", "회원 관리"]


def test_manual_buttons_and_cart_total(win):
    pos = win.pos_tab
    assert pos.product_grid.count() == 2
    pos.add_to_cart(0)
    pos.add_to_cart(0)
    pos.add_to_cart(1)
    assert pos.total_value.text() == "10,500원"
    assert pos.item_count_label.text() == "총 3개"

    pos.cart_table.selectRow(0)
    pos.change_selected_qty(-1)
    assert pos.total_value.text() == "7,500원"
    pos.delete_selected()
    assert pos.total_value.text() == "4,500원"


def test_recognize_adds_confident_detections(win, dialogs):
    win.camera.start()
    tick(win)
    win.detector.dets = [det(0, 0.9, (10, 10, 100, 100)), det(1, 0.3, (200, 10, 300, 100)),
                         det(7, 0.9, (400, 10, 500, 100))]
    win.pos_tab.recognize()
    assert win.pos_tab.cart.items == {0: 1}                 # 0.3 은 빠지고, 7번은 모르는 상품
    assert dialogs[-1][1] == "등록되지 않은 클래스"


def test_camera_frames_render_on_pos_and_collect(win):
    win.camera.start()
    assert win.pos_tab.camera_state.text().startswith("CAMERA  ON")
    win.detector.dets = [det(0, 0.9, (10, 10, 100, 100))]
    tick(win, 2)
    assert win.pos_tab.view.pixmap() is not None
    assert win.live.dets                                     # 실시간 인식 결과가 들어왔다

    win.tabs.setCurrentWidget(win.collect_tab)
    tick(win)
    win.collect_tab.mode_combo.setCurrentIndex(1)            # 여러 개 방식
    tick(win)
    assert win.collect_tab.view.pixmap() is not None

    win.camera.stop()
    assert win.pos_tab.camera_state.text() == "CAMERA  OFF"


def test_register_product_reaches_other_tabs(win, tmp_path):
    tab = win.product_tab
    tab.display_edit.setText("크림빵")
    tab.class_edit.setText("Cream Bread")
    tab.price_spin.setValue(2500)
    tab.register_product()

    assert win.ctx.products["2"]["class_name"] == "cream_bread"
    assert win.pos_tab.product_grid.count() == 3
    assert win.collect_tab.selected_class_id() == 2          # 수집 탭이 새 상품을 골라 둔다
    assert "2: cream_bread" in (tmp_path / "dataset" / "data.yaml").read_text(encoding="utf-8")
    saved = json.loads((tmp_path / "database" / "products.json").read_text(encoding="utf-8"))
    assert saved["2"]["price"] == 2500


def test_member_qr_scan_and_payment(win, dialogs):
    win.camera.start()
    tick(win)
    panel = win.pos_tab.member_panel
    panel.toggle_scan()
    assert panel.scanning

    win.qr_worker.next = "BV1:ok"
    tick(win, 3)
    assert panel.member["name"] == "홍길동"
    assert panel.status_label.text() == "회원: 홍길동"

    win.pos_tab.add_to_cart(1)
    win.pos_tab.complete_payment()
    assert win.ctx.membership_api.payments == [("u1", 4500)]
    assert not win.pos_tab.cart
    assert panel.member is None
    assert "적립: +135P" in dialogs[-1][2]


def test_payment_failure_keeps_order(win, dialogs):
    panel = win.pos_tab.member_panel
    panel.member = {"id": "u1", "name": "홍길동", "point": 0}
    win.ctx.membership_api.fail = True
    win.pos_tab.add_to_cart(0)
    win.pos_tab.complete_payment()
    assert dialogs[-1][1] == "포인트 적립 실패"
    assert win.pos_tab.cart.items == {0: 1}                  # 다시 누를 수 있게 주문이 남아 있다


def test_collect_with_fixed_guide_box_saves_files(win, tmp_path, dialogs):
    win.camera.start()
    tick(win)
    tab = win.collect_tab
    tab.select_product(0)
    tab.turn_check.setChecked(False)
    tab.auto_label_check.setChecked(False)
    tab.target_spin.setValue(10)
    tab.start_collection()
    for _ in range(10):
        tab.capture()
    assert not tab.collecting
    labels = sorted((tmp_path / "dataset" / "collected" / "labels").glob("salt_*.txt"))
    assert len(labels) == 10
    assert labels[0].read_text(encoding="utf-8").startswith("0 0.500000 0.500000 0.450000 0.450000")
    assert dialogs[-1][1] == "데이터 수집 완료"
    assert "10장" in win.product_tab.table.item(0, 4).text()


def test_train_tab_without_jetson(win, dialogs):
    assert "연결되지 않았습니다" in win.train_tab.ops_status.text()
    win.train_tab.start_training()
    assert dialogs[-1][1].startswith("긴급 전이학습")


def test_close_shuts_workers(qapp, win):
    win.close()
    assert win.detector.shut and win.qr_worker.shut
