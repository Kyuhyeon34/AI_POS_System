
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QProcess
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QLineEdit, QSpinBox, QDoubleSpinBox, QProgressBar,
    QMessageBox,
    QCheckBox, QFormLayout, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QFileDialog, QPlainTextEdit, QFrame,
    QGridLayout, QAbstractItemView, QInputDialog, QApplication
)

from camera.camera import CameraManager
from ai.factory import make_detector
from ai.worker import DetectorWorker
from ai.ops_client import OpsClient
from gui.qr_worker import QRWorker
from ai.render import draw_detections
from ai.segment import object_box, object_boxes
from gui.turntable_worker import TurntableWorker
from membership.supabase_api import SupabaseMembershipAPI


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "products.json"
SETTINGS_PATH = BASE_DIR / "database" / "settings.json"

COLLECT_IMG = BASE_DIR / "dataset" / "collected" / "images"
COLLECT_LBL = BASE_DIR / "dataset" / "collected" / "labels"

TRAIN_IMG = BASE_DIR / "dataset" / "master" / "images" / "train"
VAL_IMG = BASE_DIR / "dataset" / "master" / "images" / "val"
TRAIN_LBL = BASE_DIR / "dataset" / "master" / "labels" / "train"
VAL_LBL = BASE_DIR / "dataset" / "master" / "labels" / "val"

DATA_YAML = BASE_DIR / "dataset" / "data.yaml"
RUNS_DIR = BASE_DIR / "runs"


class MainWindow(QMainWindow):
    # 탭 순서 (build 순서와 같아야 한다)
    TAB_POS, TAB_PRODUCT, TAB_COLLECT, TAB_TRAIN, TAB_MEMBER = range(5)

    QR_EVERY = 6            # 6프레임(약 0.2초)마다 한 번만 QR 검사
    QR_FAST_WIDTH = 800     # 프리뷰 QR 검사는 이 폭으로 축소해서 (약 29ms)
    QR_HOLD_FRAMES = 45     # 검출 결과 유지 시간 (약 1.5초)

    # 화면 그리기가 느린 환경(SSH X11 포워딩은 프레임당 100ms 가까이 든다)에서
    # 매 프레임 그리면 UI 스레드가 꽉 차서 버튼이 안 눌린다.
    # 실제 그리기는 Qt 이벤트 루프에서 일어나므로 show_frame() 호출 시간으로는 알 수 없다.
    # 대신 타이머가 제때 도는지(=루프가 밀렸는지)를 보고 화면 갱신 간격을 스스로 조절한다.
    CAMERA_INTERVAL = 0.033     # 카메라 타이머 주기 (초)
    DISPLAY_MIN_GAP = 0.03      # 최소 간격 (초) — 약 30fps
    DISPLAY_MAX_GAP = 2.0       # 최대 간격 (초) — SSH X11 처럼 한 장에 0.5초씩
                                # 걸리는 환경에서는 여기까지 물러난다.
    LIVE_MIN_GAP = 0.10         # 실시간 추론 최소 간격 (초) — 최대 10회/초.
                                # 카메라 30fps 를 전부 추론하지 않는다. GPU 를
                                # 다 쓰면 화면·QR·수집이 같이 느려진다.
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BreadVision AI Bakery POS")
        self.resize(1560, 940)

        self.products = self.load_products()
        self.settings = self.load_settings()

        # 카메라: POS 탭의 카메라 목록에서 고른 것 → 없으면 환경변수(젯슨 영상) → 0번
        self.camera = CameraManager(self.resolve_camera_source(), use_env=False)
        # 추론은 전담 스레드에서 돈다. GUI 는 프레임만 던지고 바로 돌아온다.
        # (자세한 이유는 ai/worker.py 주석 참고)
        self.detector = DetectorWorker(make_detector)
        self.detector.live_ready.connect(self.on_live_result)
        self.detector.failed.connect(self.on_live_failed)
        self.detector.boot()

        # QR 도 같은 이유로 스레드에서 돌린다 (gui/qr_worker.py 주석 참고)
        self.qr_worker = QRWorker(self.QR_FAST_WIDTH)
        self.qr_worker.result.connect(self.on_qr_result)
        self.qr_worker.start()

        # 젯슨에 작업을 시킬 수 있는가.
        # 업체는 PC 만 만진다 — 데이터셋 분할·전이학습·ENGINE 변환을 여기서 눌러
        # 젯슨이 대신 실행하게 한다. (젯슨에서 직접 돌 때는 self.ops 가 None 이고
        #  예전처럼 로컬에서 실행한다)
        self.ops = None
        url = os.environ.get("BREADVISION_INFER_URL", "").strip()
        if url:
            client = OpsClient(url)
            client.refresh()
            if client.available:
                self.ops = client
                print(f"[젯슨 작업] 원격 실행 사용: {url}")
            else:
                print(f"[젯슨 작업] {url} 서버가 작업 실행을 지원하지 않습니다 "
                      "(젯슨의 inference_server.py 를 최신본으로 다시 띄우세요)")
        self.remote_job = None
        self.remote_timer = QTimer(self)
        self.remote_timer.setInterval(1000)
        self.remote_timer.timeout.connect(self.poll_remote_job)
        self.last_inference_frame = None
        self.frame = None

        self.collecting = False
        self.collect_count = 0
        self.collect_skipped = 0
        self.live_dets = []              # 실시간 인식 결과 (초록 박스로 표시)
        self.live_next_ts = 0.0          # 다음 추론 허용 시각
        self.live_gap = 0.25             # 추론 간격 (초). 실제 걸린 시간에 맞춰 조절된다
        self.live_busy = False           # 추론 스레드가 아직 처리 중인가
        self.collect_need_model = False  # 배경 분리가 실패할 때만 모델을 부른다
        self.collect_focus = 0           # 「다음 물체」로 고르고 있는 새 물체의 번호
        self.collect_manual = []         # 사용자가 품목을 지정한 박스들 [(box, class_id)]
        self.collect_ignore = []         # 빵이 아니라고 표시한 박스들 (책상·손 등)
        self.collect_lock = None         # 촬영 대상으로 고정한 것 {xyxy, class_id}
        self.turn_worker = None          # 턴테이블을 돌리는 스레드
        self.lock_tracker = None         # 고정한 빵을 그림으로 따라가는 추적기
        self.collect_boxes = []          # 이번 프레임의 라벨 후보 (collect_label_boxes 결과)
        self.last_segment_candidate = None
        self.target_count = 0

        self.cart = {}  # class_id -> qty
        self.membership_api = SupabaseMembershipAPI()
        self.current_member = None
        self.qr_detector = cv2.QRCodeDetector()
        self.last_qr_data = ""
        self.last_qr_points = None
        self.qr_seen_frames = 0
        self.qr_frame_tick = 0
        self.qr_state = None
        self.last_frame_id = -1
        self.last_display_ts = 0.0
        self.last_tick_ts = 0.0
        self.display_gap = self.DISPLAY_MIN_GAP
        # BREADVISION_DEBUG=1 로 실행하면 2초마다 실제 성능을 터미널에 찍는다
        self.debug_perf = os.environ.get("BREADVISION_DEBUG", "") == "1"
        self.preview_max = self._decide_preview_max()
        self._preview_area_cache = {}   # 라벨별 원래 크기
        self._dbg_t0 = time.perf_counter()
        self._dbg_ticks = 0
        self._dbg_frames = 0
        self._dbg_drawn = 0
        self._dbg_lag = 0.0
        self._dbg_infer = 0          # 추론 스레드가 실제로 돈 횟수
        self._dbg_infer_ms = 0.0

        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_frame)

        self.collect_timer = QTimer(self)
        self.collect_timer.timeout.connect(self.capture_dataset)

        self.train_process = None
        self.last_best_model = ""
        self.pending_engine = ""

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(self.build_pos_tab(), "POS")
        self.tabs.addTab(self.build_product_tab(), "상품 관리")
        self.tabs.addTab(self.build_collect_tab(), "데이터 수집")
        self.tabs.addTab(self.build_train_tab(), "모델 배포 및 모델 학습")
        self.tabs.addTab(self.build_membership_tab(), "회원 관리")

        self.apply_style()
        self.refresh_all()
        self.try_load_saved_engine()

    # -------------------- STYLE --------------------
    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0A0F16;
                color: #E8EEF5;
                font-size: 14px;
                font-family: "Malgun Gothic";
            }

            QTabWidget::pane {
                border: 1px solid #1E2A38;
                background: #0E141C;
                top: -1px;
            }

            QTabBar::tab {
                background: #111923;
                color: #8FA0B3;
                border: 1px solid #202D3B;
                border-bottom: none;
                padding: 13px 24px;
                min-width: 150px;
                font-weight: 700;
            }

            QTabBar::tab:hover {
                background: #152131;
                color: #DCE8F6;
            }

            QTabBar::tab:selected {
                background: #0F1722;
                color: #5DA9FF;
                border-top: 3px solid #2D8CFF;
            }

            QLabel#PageTitle {
                font-size: 25px;
                font-weight: 900;
                color: #F4F8FC;
            }

            QLabel#PageDesc {
                color: #8E9FB1;
                font-size: 13px;
            }

            QLabel#TotalCaption {
                font-size: 15px;
                font-weight: 700;
                color: #9FB0C2;
            }

            QLabel#TotalValue {
                color: #FFFFFF;
                font-size: 44px;
                font-weight: 900;
            }

            QLabel#ReceiptInfo {
                color: #91A3B5;
                font-size: 13px;
                font-weight: 700;
            }

            QGroupBox {
                background: #101822;
                border: 1px solid #243343;
                border-radius: 8px;
                margin-top: 14px;
                padding: 13px;
                font-weight: 800;
                color: #DDE6EF;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #AFC4D9;
            }

            QPushButton {
                background: #141E29;
                color: #DCE6F0;
                border: 1px solid #2A3948;
                border-radius: 6px;
                padding: 10px 13px;
                font-weight: 700;
            }

            QPushButton:hover {
                background: #192838;
                border-color: #3C5A78;
            }

            QPushButton:pressed {
                background: #0D1520;
            }

            QPushButton#Primary {
                background: #1468C3;
                color: white;
                border: 1px solid #2B7ED5;
            }

            QPushButton#Primary:hover {
                background: #1B77D6;
            }

            QPushButton#Success {
                background: #0E5EA8;
                color: white;
                border: 1px solid #2478C3;
                font-size: 17px;
                padding: 15px;
            }

            QPushButton#Success:hover {
                background: #1170C6;
            }

            QPushButton#Danger {
                background: #1A222C;
                color: #FF7B7B;
                border: 1px solid #5C2A2A;
            }

            QPushButton#Danger:hover {
                background: #2A171A;
                border-color: #8A3C3C;
            }

            QPushButton#Warning {
                background: #0B3F73;
                color: white;
                border: 1px solid #1E6CAF;
            }

            QPushButton#Warning:hover {
                background: #0E518F;
            }

            QPushButton#Dark {
                background: #0C1219;
                color: #DCE5EE;
                border: 1px solid #243140;
            }

            QPushButton#ProductButton {
                background: #0F1720;
                color: #E8EEF5;
                border: 1px solid #263646;
                min-height: 64px;
                font-size: 14px;
                text-align: left;
                padding-left: 13px;
            }

            QPushButton#ProductButton:hover {
                background: #12243A;
                border: 1px solid #2D8CFF;
                color: #FFFFFF;
            }

            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: #0D141D;
                color: #E6EDF5;
                border: 1px solid #2A3948;
                border-radius: 6px;
                padding: 8px;
                min-height: 24px;
                selection-background-color: #245B91;
            }

            QComboBox QAbstractItemView {
                background: #101822;
                color: #E6EDF5;
                selection-background-color: #1E5F99;
                border: 1px solid #2B3A49;
            }

            QTableWidget {
                background: #0C1219;
                alternate-background-color: #101821;
                gridline-color: #1D2A37;
                selection-background-color: #173A5D;
                selection-color: #FFFFFF;
                border: 1px solid #24313F;
                font-size: 14px;
            }

            QHeaderView::section {
                background: #141D27;
                color: #AFC0D1;
                padding: 9px;
                border: none;
                border-right: 1px solid #24313F;
                font-weight: 800;
            }

            QPlainTextEdit {
                background: #080D12;
                color: #C9D5E1;
                border: 1px solid #24313F;
                font-family: Consolas;
            }

            QProgressBar {
                background: #0C1219;
                border: 1px solid #263442;
                border-radius: 5px;
                text-align: center;
                min-height: 22px;
                color: #DCE6F0;
            }

            QProgressBar::chunk {
                background: #2D8CFF;
                border-radius: 4px;
            }

            QFrame#Card {
                background: #0F1720;
                border: 1px solid #223141;
                border-radius: 8px;
            }

            QFrame#POSHeader {
                background: #0B1118;
                border: 1px solid #1D2A37;
                border-radius: 8px;
            }

            QFrame#TotalBox {
                background: #0A1B2D;
                border: 1px solid #245B91;
                border-radius: 8px;
            }

            QScrollBar:vertical {
                background: #0B1118;
                width: 11px;
                margin: 0;
            }

            QScrollBar::handle:vertical {
                background: #2A3948;
                min-height: 25px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical:hover {
                background: #3A5066;
            }
        """)

    # -------------------- DATA --------------------
    def load_products(self):
        if not DB_PATH.exists():
            # 처음 설치한 PC/젯슨: 기본 12종 목록으로 시작한다 (가격은 「상품 관리」에서)
            default = DB_PATH.with_name("products.default.json")
            if not default.exists():
                return {}
            DB_PATH.write_text(default.read_text(encoding="utf-8"), encoding="utf-8")

        data = json.loads(DB_PATH.read_text(encoding="utf-8"))

        # legacy format -> master format
        if data and not all(str(k).isdigit() for k in data.keys()):
            converted = {}
            for key, value in data.items():
                cid = int(value.get("class_id", len(converted)))
                converted[str(cid)] = {
                    "class_name": key,
                    "display_name": value.get("name", key),
                    "price": int(value.get("price", 0)),
                }
            data = converted
            DB_PATH.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        return data

    def save_products(self):
        DB_PATH.write_text(
            json.dumps(self.products, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def load_settings(self):
        default = {
            "training_base_model": "models/m12_A.pt",
            "active_engine": "",
            "active_model_classes": {}
        }
        try:
            if SETTINGS_PATH.exists():
                raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            elif SETTINGS_PATH.with_name("settings.example.json").exists():
                raw = json.loads(SETTINGS_PATH.with_name("settings.example.json").read_text(encoding="utf-8"))
            else:
                raw = {}
        except Exception:
            raw = {}

        # v7 migration
        old_active = raw.get("active_model", "")
        if old_active and old_active.lower().endswith(".pt"):
            raw.setdefault("training_base_model", old_active)
        elif old_active and old_active.lower().endswith(".engine"):
            raw.setdefault("active_engine", old_active)

        for k, v in default.items():
            raw.setdefault(k, v)

        self.settings = raw
        self.save_settings()
        return raw

    def save_settings(self):
        SETTINGS_PATH.write_text(
            json.dumps(self.settings, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def next_class_id(self):
        if not self.products:
            return 0
        return max(int(k) for k in self.products) + 1

    def count_collected_for_class(self, class_name):
        return len(list(COLLECT_IMG.glob(f"{class_name}_*.jpg")))

    def refresh_all(self):
        self.refresh_product_table()
        self.refresh_collect_combo()
        self.refresh_product_buttons()
        self.refresh_cart_table()
        self.refresh_train_summary()

    # -------------------- POS --------------------
    def build_pos_tab(self):
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # POS 상단 상태바
        header = QFrame()
        header.setObjectName("POSHeader")
        header_lay = QHBoxLayout(header)

        title_box = QVBoxLayout()
        title = QLabel("BreadVision POS | JETSON")
        title.setStyleSheet("color:#F6F9FC; font-size:24px; font-weight:900;")
        subtitle = QLabel("AI Recognition  ·  Order  ·  Price Calculation  ·  Payment")
        subtitle.setStyleSheet("color:#7F93A8; font-size:12px;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        self.camera_state = QLabel("CAMERA  OFF")
        self.camera_state.setStyleSheet(
            "background:#131D27; color:#91A3B5; padding:8px 12px; "
            "border:1px solid #2B3947; border-radius:6px; font-weight:800;"
        )

        self.active_model_badge = QLabel("MODEL  NOT SET")
        self.active_model_badge.setStyleSheet(
            "background:#131D27; color:#6EAFFF; padding:8px 12px; "
            "border:1px solid #244A70; border-radius:6px; font-weight:800;"
        )

        header_lay.addLayout(title_box)
        header_lay.addStretch()
        header_lay.addWidget(self.camera_state)
        header_lay.addWidget(self.active_model_badge)
        root.addWidget(header)

        # 메인 POS 영역
        main = QHBoxLayout()
        main.setSpacing(10)

        # 좌측: 카메라 + 인식 버튼
        left_card = QFrame()
        left_card.setObjectName("Card")
        left = QVBoxLayout(left_card)

        camera_head = QHBoxLayout()
        camera_title = QLabel("LIVE VIEW / PRODUCT RECOGNITION")
        camera_title.setStyleSheet("font-size:18px; font-weight:900;")
        guide = QLabel("트레이를 카메라 영역에 맞춘 뒤 인식을 실행하세요.")
        guide.setObjectName("PageDesc")
        camera_head.addWidget(camera_title)
        camera_head.addStretch()
        camera_head.addWidget(guide)
        left.addLayout(camera_head)

        self.pos_camera = QLabel("NO CAMERA SIGNAL\n카메라 시작 버튼을 눌러주세요.")
        self.pos_camera.setAlignment(Qt.AlignCenter)
        self.pos_camera.setMinimumSize(760, 500)
        self.pos_camera.setStyleSheet(
            "background:#20252C; color:white; border-radius:7px; "
            "font-size:17px; font-weight:700;"
        )
        left.addWidget(self.pos_camera, 1)

        cam_btns = QHBoxLayout()

        btn_cam_on = QPushButton("카메라 시작")
        btn_cam_on.setObjectName("Dark")
        btn_cam_on.setToolTip("USB 카메라를 연결하고 실시간 화면을 시작합니다.")
        btn_cam_on.clicked.connect(self.start_camera)

        self.live_detect_check = QCheckBox("실시간 인식 표시")
        self.live_detect_check.setChecked(True)
        self.live_detect_check.setToolTip(
            "대기 중에도 화면에서 빵을 찾아 초록 박스로 표시합니다.\n"
            "주문에 담는 것은 「상품 인식 실행」 버튼을 눌렀을 때만 이뤄집니다."
        )

        btn_ai = QPushButton("상품 인식 실행")
        btn_ai.setObjectName("Warning")
        btn_ai.setToolTip("현재 카메라 화면에서 빵을 인식하고 주문 목록에 반영합니다.")
        btn_ai.clicked.connect(self.ai_detect_placeholder)

        btn_cam_off = QPushButton("카메라 종료")
        btn_cam_off.setToolTip("카메라 연결을 종료합니다.")
        btn_cam_off.clicked.connect(self.stop_camera)

        # 카메라 고르기 — PC 에서 젯슨 영상 대신 내장 카메라가 열리는 일이 있어서
        # 사람이 직접 바꿀 수 있게 한다. 고른 것은 settings.json 에 기억한다.
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(200)
        self.camera_combo.setToolTip(
            "어느 카메라를 쓸지 고릅니다. 고르면 바로 바뀌고 다음 실행에도 기억합니다.\n"
            "· 젯슨 카메라: 젯슨에 꽂힌 USB 카메라 영상(8090)\n"
            "· PC 카메라: 이 컴퓨터의 카메라 (보통 0번이 내장)"
        )
        self.refresh_camera_combo()
        self.camera_combo.activated.connect(self.on_camera_combo)

        btn_cam_find = QPushButton("카메라 찾기")
        btn_cam_find.setToolTip("이 컴퓨터에 연결된 카메라 번호를 찾아 목록에 표시합니다.")
        btn_cam_find.clicked.connect(self.scan_cameras)

        cam_btns.addWidget(self.camera_combo)
        cam_btns.addWidget(btn_cam_find)
        cam_btns.addWidget(btn_cam_on)
        cam_btns.addWidget(btn_ai, 2)
        cam_btns.addWidget(btn_cam_off)
        cam_btns.addWidget(self.live_detect_check)
        left.addLayout(cam_btns)

        # 수동 상품 선택 영역
        quick = QGroupBox("MANUAL PRODUCT SELECT")
        quick_lay = QGridLayout(quick)
        self.product_button_layout = quick_lay
        left.addWidget(quick)

        # 우측: 실제 POS 계산/영수증 영역
        right_card = QFrame()
        right_card.setObjectName("Card")
        right = QVBoxLayout(right_card)

        receipt_head = QHBoxLayout()
        receipt_title = QLabel("ORDER")
        receipt_title.setStyleSheet("font-size:20px; font-weight:900;")
        self.item_count_label = QLabel("총 0개")
        self.item_count_label.setObjectName("ReceiptInfo")
        receipt_head.addWidget(receipt_title)
        receipt_head.addStretch()
        receipt_head.addWidget(self.item_count_label)
        right.addLayout(receipt_head)

        member_box = QGroupBox("MEMBERSHIP / QR")
        member_lay = QVBoxLayout(member_box)
        member_top = QHBoxLayout()
        self.member_status_label = QLabel("회원 미인증")
        self.member_status_label.setStyleSheet("font-size:16px; font-weight:900;")
        self.member_point_label = QLabel("")
        self.member_point_label.setObjectName("ReceiptInfo")
        member_top.addWidget(self.member_status_label)
        member_top.addStretch()
        member_top.addWidget(self.member_point_label)
        member_lay.addLayout(member_top)

        self.qr_live_label = QLabel("QR 대기 중")
        self.qr_live_label.setStyleSheet(
            "color:#91A3B5; font-size:12px; font-weight:700;"
        )
        member_lay.addWidget(self.qr_live_label)

        member_btns = QHBoxLayout()
        btn_member_scan = QPushButton("회원 QR 스캔")
        btn_member_scan.setObjectName("Primary")
        btn_member_scan.setToolTip("현재 카메라 화면에서 휴대폰 멤버십 QR을 읽습니다.")
        btn_member_scan.clicked.connect(self.scan_member_qr)
        btn_member_clear = QPushButton("회원 해제")
        btn_member_clear.setObjectName("Dark")
        btn_member_clear.clicked.connect(self.clear_member)
        member_btns.addWidget(btn_member_scan, 2)
        member_btns.addWidget(btn_member_clear)
        member_lay.addLayout(member_btns)

        right.addWidget(member_box)

        self.cart_table = QTableWidget(0, 5)
        self.cart_table.setHorizontalHeaderLabels(
            ["상품", "단가", "수량", "금액", ""]
        )
        self.cart_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cart_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cart_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cart_table.setAlternatingRowColors(True)
        self.cart_table.verticalHeader().setVisible(False)
        self.cart_table.setMinimumWidth(520)
        self.cart_table.setMinimumHeight(380)
        right.addWidget(self.cart_table, 1)

        qty_box = QGridLayout()

        btn_minus = QPushButton("수량 감소")
        btn_minus.setToolTip("선택한 상품 수량을 1개 줄입니다.")
        btn_minus.clicked.connect(lambda: self.change_selected_qty(-1))

        btn_plus = QPushButton("수량 증가")
        btn_plus.setObjectName("Primary")
        btn_plus.setToolTip("선택한 상품 수량을 1개 늘립니다.")
        btn_plus.clicked.connect(lambda: self.change_selected_qty(1))

        btn_delete = QPushButton("선택 항목 삭제")
        btn_delete.setObjectName("Danger")
        btn_delete.setToolTip("선택한 상품을 주문에서 제거합니다.")
        btn_delete.clicked.connect(self.delete_selected_cart_item)

        btn_clear = QPushButton("주문 초기화")
        btn_clear.setObjectName("Dark")
        btn_clear.setToolTip("현재 주문 내역을 전부 초기화합니다.")
        btn_clear.clicked.connect(self.clear_cart)

        qty_box.addWidget(btn_minus, 0, 0)
        qty_box.addWidget(btn_plus, 0, 1)
        qty_box.addWidget(btn_delete, 1, 0)
        qty_box.addWidget(btn_clear, 1, 1)
        right.addLayout(qty_box)

        total_box = QFrame()
        total_box.setObjectName("TotalBox")
        total_lay = QVBoxLayout(total_box)

        total_cap_row = QHBoxLayout()
        cap = QLabel("TOTAL")
        cap.setObjectName("TotalCaption")
        self.total_qty_detail = QLabel("상품 0종 / 수량 0개")
        self.total_qty_detail.setObjectName("ReceiptInfo")
        total_cap_row.addWidget(cap)
        total_cap_row.addStretch()
        total_cap_row.addWidget(self.total_qty_detail)

        self.total_value = QLabel("0원")
        self.total_value.setObjectName("TotalValue")
        self.total_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        total_lay.addLayout(total_cap_row)
        total_lay.addWidget(self.total_value)
        right.addWidget(total_box)

        pay = QPushButton("결제 처리")
        pay.setObjectName("Success")
        pay.setToolTip("현재 주문 금액을 결제 완료 처리하고 새 주문 상태로 초기화합니다.")
        pay.clicked.connect(self.complete_payment)
        right.addWidget(pay)

        pos_help = QLabel(
            "Workflow  |  Camera → Recognition → Order Check → Total → Payment"
        )
        pos_help.setStyleSheet(
            "color:#6F8498; padding:6px 2px; font-size:12px;"
        )
        pos_help.setWordWrap(True)
        right.addWidget(pos_help)

        main.addWidget(left_card, 3)
        main.addWidget(right_card, 2)
        root.addLayout(main)

        return w

    def refresh_product_buttons(self):
        if not hasattr(self, "product_button_layout"):
            return

        while self.product_button_layout.count():
            item = self.product_button_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        ordered = sorted(self.products.items(), key=lambda x: int(x[0]))
        for i, (cid, p) in enumerate(ordered):
            btn = QPushButton(
                f"{p['display_name']}   |   {int(p['price']):,}원"
            )
            btn.setObjectName("ProductButton")
            btn.setToolTip(
                f"{p['display_name']} 1개를 주문 목록에 추가합니다."
            )
            btn.clicked.connect(
                lambda checked=False, class_id=int(cid): self.add_to_cart(class_id)
            )
            self.product_button_layout.addWidget(btn, i // 2, i % 2)

    def add_to_cart(self, class_id):
        self.cart[class_id] = self.cart.get(class_id, 0) + 1
        self.refresh_cart_table()

    def refresh_cart_table(self):
        if not hasattr(self, "cart_table"):
            return

        rows = sorted(self.cart.items(), key=lambda x: x[0])
        self.cart_table.setRowCount(len(rows))

        total = 0
        for r, (cid, qty) in enumerate(rows):
            product = self.products.get(str(cid))
            if not product:
                continue

            price = int(product["price"])
            subtotal = price * qty
            total += subtotal

            self.cart_table.setItem(r, 0, QTableWidgetItem(product["display_name"]))
            self.cart_table.setItem(r, 1, QTableWidgetItem(f"{price:,}원"))
            self.cart_table.setItem(r, 2, QTableWidgetItem(str(qty)))
            self.cart_table.setItem(r, 3, QTableWidgetItem(f"{subtotal:,}원"))

            delete_btn = QPushButton("삭제")
            delete_btn.setObjectName("Danger")
            delete_btn.setToolTip(f"{product['display_name']}을 주문에서 제거합니다.")
            delete_btn.clicked.connect(
                lambda checked=False, class_id=cid: self.remove_cart_item(class_id)
            )
            self.cart_table.setCellWidget(r, 4, delete_btn)

        self.total_value.setText(f"{total:,}원")
        total_qty = sum(self.cart.values())
        product_kinds = len([q for q in self.cart.values() if q > 0])
        if hasattr(self, "item_count_label"):
            self.item_count_label.setText(f"총 {total_qty}개")
        if hasattr(self, "total_qty_detail"):
            self.total_qty_detail.setText(f"상품 {product_kinds}종 / 수량 {total_qty}개")

    def remove_cart_item(self, cid):
        self.cart.pop(cid, None)
        self.refresh_cart_table()

    def delete_selected_cart_item(self):
        row = self.cart_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "상품 선택", "삭제할 상품을 주문 목록에서 먼저 선택해주세요.")
            return
        display_name = self.cart_table.item(row, 0).text()
        cid = next(
            (int(k) for k, p in self.products.items()
             if p["display_name"] == display_name),
            None
        )
        if cid is not None:
            self.cart.pop(cid, None)
            self.refresh_cart_table()

    def change_selected_qty(self, delta):
        row = self.cart_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "상품 선택", "주문 목록에서 상품을 먼저 선택해주세요.")
            return

        display_name = self.cart_table.item(row, 0).text()
        cid = next(
            (int(k) for k, p in self.products.items()
             if p["display_name"] == display_name),
            None
        )
        if cid is None:
            return

        self.cart[cid] = self.cart.get(cid, 0) + delta
        if self.cart[cid] <= 0:
            self.cart.pop(cid, None)
        self.refresh_cart_table()

    def clear_cart(self):
        if not self.cart:
            return
        self.cart.clear()
        self.refresh_cart_table()

    def scan_member_qr(self):
        frame = getattr(self, "frame", None)
        if frame is None:
            QMessageBox.information(
                self, "카메라 필요",
                "카메라를 먼저 시작한 뒤 휴대폰의 멤버십 QR을 카메라에 보여주세요."
            )
            return

        # Prefer the QR detected continuously in update_frame().
        data = getattr(self, "last_qr_data", "") or ""
        if not data:
            data, points = self._decode_qr_robust(frame)
            if data:
                self.last_qr_data = data
                self.last_qr_points = points

        if not data:
            QMessageBox.information(
                self, "QR 미검출",
                "QR을 아직 읽지 못했습니다.\n"
                "휴대폰 밝기를 높이고 QR 전체가 카메라 화면에 들어오게 한 뒤, "
                "초록색 테두리가 나타날 때 다시 눌러주세요."
            )
            return

        if not data.startswith("BV1:"):
            QMessageBox.warning(
                self, "멤버십 QR 아님",
                "QR은 인식했지만 BreadVision 회원 QR(BV1:) 형식이 아닙니다."
            )
            return

        try:
            member = self.membership_api.find_member_by_qr_payload(data)
        except Exception as e:
            QMessageBox.critical(self, "Supabase 회원 조회 오류", str(e))
            return

        if not member:
            QMessageBox.warning(
                self, "회원 없음",
                "QR은 정상 인식했지만 Supabase에서 해당 회원을 찾지 못했습니다."
            )
            return

        self.current_member = member
        self.refresh_member_status()
        QMessageBox.information(
            self, "회원 인증 완료",
            f"{member['name']}님\n보유 포인트: {int(member['point']):,}P"
        )


    def refresh_member_status(self):
        if not hasattr(self, "member_status_label"):
            return
        if not self.current_member:
            self.member_status_label.setText("회원 미인증")
            self.member_point_label.setText("")
            return
        latest = self.current_member
        if latest:
            self.current_member = latest
        self.member_status_label.setText(f"회원: {self.current_member['name']}")
        self.member_point_label.setText(f"{int(self.current_member['point']):,}P")

    def clear_member(self):
        self.current_member = None
        self.refresh_member_status()

    def complete_payment(self):
        if not self.cart:
            QMessageBox.information(self, "주문 없음", "결제할 상품이 없습니다.")
            return

        total = 0
        for cid, qty in self.cart.items():
            p = self.products.get(str(cid))
            if p:
                total += int(p["price"]) * qty

        extra = ""
        if self.current_member:
            result = self.membership_api.complete_payment(self.current_member["id"], total)
            extra = (
                f"\n\n회원: {self.current_member['name']}"
                f"\n적립: +{result['earned_point']:,}P"
                f"\n보유 포인트: {result['current_point']:,}P"
            )

        QMessageBox.information(
            self,
            "결제 완료",
            f"결제가 완료되었습니다.\n총 금액: {total:,}원{extra}"
        )
        self.cart.clear()
        self.clear_member()
        self.refresh_cart_table()
        if hasattr(self, "member_table"):
            self.refresh_member_table()

    def build_membership_tab(self):
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)

        title = QLabel("회원 관리")
        title.setObjectName("PageTitle")
        desc = QLabel("중앙 DB: Supabase   |   회원 앱: Android APK")
        desc.setObjectName("PageDesc")
        root.addWidget(title)
        root.addWidget(desc)

        flow = QGroupBox("시연 흐름")
        flow_lay = QVBoxLayout(flow)
        flow_lay.addWidget(QLabel(
            "Android APK 회원가입/로그인 → 개인 QR 표시 → POS 회원 QR 스캔 "
            "→ 결제 → 결제금액의 3% 포인트 적립"
        ))
        root.addWidget(flow)

        dbbox = QGroupBox("중앙 DB 상태")
        db_lay = QVBoxLayout(dbbox)
        db_lay.addWidget(QLabel(
            "회원정보와 포인트는 Jetson 로컬 DB가 아니라 Supabase 중앙 DB에 저장됩니다."
        ))
        root.addWidget(dbbox)

        root.addStretch()
        return w

    def ai_detect_placeholder(self):
        if not self.detector.ready():
            engine = self.settings.get("active_engine", "")
            if engine:
                try:
                    self.detector.load(engine)
                except Exception as e:
                    QMessageBox.critical(self, "ENGINE 로드 실패", str(e))
                    return
            else:
                QMessageBox.information(
                    self,
                    "POS 추론 모델 미설정",
                    "모델 학습 탭에서 Jetson용 TensorRT .engine 파일을 먼저 선택해주세요."
                )
                return

        frame = getattr(self, "frame", None)
        if frame is None:
            QMessageBox.information(
                self,
                "카메라 프레임 없음",
                "카메라를 시작한 뒤 트레이에 빵을 올리고 다시 인식해주세요."
            )
            return

        try:
            # 「빵 인식」 은 한 번만 보므로 좌우반전 TTA 로 한 번 더 본다 (19 → 39ms).
            # 실측: 원본에서 rotus 0.22 로 잘못 잡히던 휘낭시에가 fruit 0.76 으로 통과.
            # 실시간 미리보기(33ms 주기)는 계속 단일로 돈다 (ai/worker.py submit_live).
            _plotted, detections, _counts = self.detector.infer(frame.copy(), draw=False, tta=True)
        except Exception as e:
            QMessageBox.critical(self, "YOLO 추론 실패", str(e))
            return

        # 확실한 것만 주문에 담는다. 서버는 낮은 것까지 보내온다.
        detections = self.pos_dets(detections)
        class_counts = {}
        for d in detections:
            cid = int(d["class_id"])
            class_counts[cid] = class_counts.get(cid, 0) + 1

        plotted, _ = draw_detections(frame.copy(), detections,
                                     labels=self.class_name_map())
        # 카메라 화면에 bbox 결과 즉시 표시
        self.show_frame_on_label(plotted, self.pos_camera)

        # 인식할 때마다 검출된 수량을 주문에 "더한다".
        # 덮어쓰면 같은 빵을 다시 인식했을 때 수량이 그대로여서 두 번째 누름이 무시된다.
        # products.json 의 키는 문자열("0"), 검출 결과의 class_id 는 정수(0) 이므로
        # 조회할 때 str() 로 맞춰준다. (장바구니 키는 add_to_cart 와 같게 정수로 둔다)
        added = {}
        for cid, count in class_counts.items():
            if str(cid) in self.products:
                key = int(cid)
                self.cart[key] = self.cart.get(key, 0) + count
                added[key] = count

        self.refresh_cart_table()

        if added:
            names = ", ".join(
                f"{self.products[str(k)]['display_name']} +{v}"
                for k, v in sorted(added.items())
            )
            self.statusBar().showMessage(f"주문 추가: {names}", 4000)
        else:
            self.statusBar().showMessage("인식된 상품이 없습니다.", 4000)

        unknown_ids = [cid for cid in class_counts if str(cid) not in self.products]
        if unknown_ids:
            QMessageBox.warning(
                self,
                "등록되지 않은 클래스",
                "YOLO에서 검출했지만 products.json에 없는 Class ID: "
                + ", ".join(map(str, unknown_ids))
            )

    def build_product_tab(self):
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(22, 18, 22, 18)

        title = QLabel("상품 및 가격 관리")
        title.setObjectName("PageTitle")
        desc = QLabel(
            "판매할 빵을 등록하고 가격을 관리합니다. 신규 등록 시 YOLO Class ID가 자동 배정됩니다."
        )
        desc.setObjectName("PageDesc")
        root.addWidget(title)
        root.addWidget(desc)

        self.product_table = QTableWidget(0, 5)
        self.product_table.setHorizontalHeaderLabels(
            ["Class ID", "YOLO Class Name", "화면 표시 상품명", "판매 가격", "수집 이미지"]
        )
        self.product_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.product_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.product_table.setAlternatingRowColors(True)
        root.addWidget(self.product_table)

        group = QGroupBox("신규 상품 추가")
        form = QFormLayout(group)

        self.new_display_name = QLineEdit()
        self.new_display_name.setPlaceholderText("예: 크림빵")

        self.new_class_name = QLineEdit()
        self.new_class_name.setPlaceholderText("예: cream_bread")

        self.new_price = QSpinBox()
        self.new_price.setRange(0, 1000000)
        self.new_price.setSingleStep(100)
        self.new_price.setSuffix(" 원")

        self.auto_class_id_label = QLabel("-")
        self.auto_class_id_label.setStyleSheet("font-weight:800; color:#0B63CE;")

        form.addRow("상품명", self.new_display_name)
        form.addRow("YOLO Class Name", self.new_class_name)
        form.addRow("판매 가격", self.new_price)
        form.addRow("자동 배정 Class ID", self.auto_class_id_label)
        root.addWidget(group)

        buttons = QHBoxLayout()

        add = QPushButton("＋ 신규 상품 등록")
        add.setObjectName("Primary")
        add.setToolTip(
            "상품명·Class Name·가격을 저장하고 Class ID를 자동 배정합니다. "
            "등록 즉시 POS와 데이터 수집 탭에도 반영됩니다."
        )
        add.clicked.connect(self.register_product)

        update = QPushButton("₩ 선택 상품 가격 변경")
        update.setToolTip(
            "위 표에서 선택한 상품의 가격을 현재 입력된 판매 가격으로 변경합니다."
        )
        update.clicked.connect(self.update_selected_price)

        buttons.addWidget(add)
        buttons.addWidget(update)
        root.addLayout(buttons)

        how = QLabel(
            "버튼 설명  |  신규 상품 등록: 상품 마스터에 새 품목 추가  |  "
            "선택 상품 가격 변경: 표에서 고른 품목의 판매가 수정"
        )
        how.setObjectName("PageDesc")
        root.addWidget(how)

        return w

    def refresh_product_table(self):
        if not hasattr(self, "product_table"):
            return

        rows = sorted(self.products.items(), key=lambda x: int(x[0]))
        self.product_table.setRowCount(len(rows))

        for r, (cid, p) in enumerate(rows):
            values = [
                cid,
                p["class_name"],
                p["display_name"],
                f"{int(p['price']):,}원",
                f"{self.count_collected_for_class(p['class_name'])}장",
            ]
            for c, value in enumerate(values):
                self.product_table.setItem(r, c, QTableWidgetItem(value))

        self.auto_class_id_label.setText(str(self.next_class_id()))

    def register_product(self):
        display_name = self.new_display_name.text().strip()
        class_name = self.new_class_name.text().strip()
        price = self.new_price.value()

        if not display_name or not class_name:
            QMessageBox.warning(self, "입력 오류", "상품명과 YOLO Class Name을 입력해주세요.")
            return

        class_name = class_name.lower().replace(" ", "_")

        if any(p["class_name"] == class_name for p in self.products.values()):
            QMessageBox.warning(self, "중복", "이미 존재하는 YOLO Class Name입니다.")
            return

        cid = self.next_class_id()
        self.products[str(cid)] = {
            "class_name": class_name,
            "display_name": display_name,
            "price": price,
        }
        self.save_products()

        COLLECT_IMG.mkdir(parents=True, exist_ok=True)
        COLLECT_LBL.mkdir(parents=True, exist_ok=True)

        self.generate_data_yaml()

        self.new_display_name.clear()
        self.new_class_name.clear()
        self.new_price.setValue(0)

        self.refresh_all()

        idx = self.collect_product_combo.findData(cid)
        if idx >= 0:
            self.collect_product_combo.setCurrentIndex(idx)

        QMessageBox.information(
            self,
            "상품 등록 완료",
            f"{display_name}이 등록되었습니다.\n"
            f"Class ID: {cid}\n"
            f"판매 가격: {price:,}원\n\n"
            "POS / 데이터 수집 / 모델 학습 목록에 함께 반영되었습니다."
        )

    def update_selected_price(self):
        row = self.product_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "상품 선택", "가격을 바꿀 상품을 표에서 먼저 선택해주세요.")
            return

        cid = self.product_table.item(row, 0).text()
        old = int(self.products[cid]["price"])
        new = self.new_price.value()
        self.products[cid]["price"] = new
        self.save_products()
        self.refresh_all()

        QMessageBox.information(
            self,
            "가격 변경 완료",
            f"{self.products[cid]['display_name']}\n"
            f"{old:,}원 → {new:,}원"
        )

    # -------------------- COLLECTION --------------------
    def build_collect_tab(self):
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(22, 18, 22, 18)

        title = QLabel("신규 상품 데이터 수집")
        title.setObjectName("PageTitle")
        desc = QLabel(
            "빵 1개를 빨간 중앙 박스 안에 놓으면 이미지와 YOLO Bounding Box 라벨을 자동 저장합니다."
        )
        desc.setObjectName("PageDesc")
        root.addWidget(title)
        root.addWidget(desc)

        content = QHBoxLayout()

        left_card = QFrame()
        left_card.setObjectName("Card")
        left = QVBoxLayout(left_card)

        self.collect_camera = QLabel("먼저 「수집용 카메라 켜기」를 눌러주세요.")
        self.collect_camera.setAlignment(Qt.AlignCenter)
        self.collect_camera.setMinimumSize(900, 650)
        self.collect_camera.setStyleSheet(
            "background:#20252C; color:white; border-radius:8px; font-size:16px;"
        )
        left.addWidget(self.collect_camera)

        right_card = QFrame()
        right_card.setObjectName("Card")
        right = QVBoxLayout(right_card)

        settings_group = QGroupBox("수집 설정")
        form = QFormLayout(settings_group)

        self.collect_product_combo = QComboBox()

        self.target_spin = QSpinBox()
        self.target_spin.setRange(10, 5000)
        self.target_spin.setValue(200)
        self.target_spin.setSuffix(" 장")

        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.1, 10.0)
        self.interval_spin.setSingleStep(0.1)
        self.interval_spin.setValue(0.5)
        self.interval_spin.setSuffix(" 초")

        self.auto_label_check = QCheckBox("AI 자동 라벨 (빵을 움직이며 촬영)")
        self.auto_label_check.setChecked(True)
        self.auto_label_check.setToolTip(
            "현재 POS 모델로 빵 위치를 찾아 그 박스를 라벨로 저장합니다.\n"
            "끄면 화면 정중앙 고정 박스를 라벨로 씁니다(빵을 가이드 박스에 맞춰야 함)."
        )

        # 수집 방식 — 한 개(예전 방식) / 여러 개(고르기·고정)
        self.collect_mode_combo = QComboBox()
        self.collect_mode_combo.addItem("한 개 — 빵 1개 자동 인식 (예전 방식)", "single")
        self.collect_mode_combo.addItem("여러 개 — 박스 고르기·고정", "multi")
        self.collect_mode_combo.setToolTip(
            "한 개: 흰 바탕에 빵 1개만 올리면 박스를 자동으로 잡아 고른 품목으로 저장합니다.\n"
            "여러 개: 화면의 빵을 모두 라벨링합니다. 모르는 빵은 골라서 품목을 지정해야 저장됩니다.")
        mode_idx = self.collect_mode_combo.findData(self.settings.get("collect_mode", "single"))
        self.collect_mode_combo.setCurrentIndex(max(0, mode_idx))
        self.collect_mode_combo.currentIndexChanged.connect(self.on_collect_mode_changed)

        form.addRow("수집할 상품", self.collect_product_combo)
        form.addRow("수집 방식", self.collect_mode_combo)
        form.addRow("목표 촬영 수", self.target_spin)
        form.addRow("자동 촬영 간격", self.interval_spin)
        self.turn_check = QCheckBox("촬영 중 턴테이블 자동 회전")
        self.turn_check.setChecked(True)
        self.turn_check.setToolTip(
            "목표 장수에 맞춰 조금씩 돌리며 찍습니다.\n"
            "카메라 뒤 케이블 때문에 한 방향으로 계속 돌지 않고 왕복만 합니다.")

        self.span_spin = QSpinBox()
        self.span_spin.setRange(10, 180)
        self.span_spin.setValue(180)
        self.span_spin.setSuffix(" 도")
        self.span_spin.setToolTip("왼쪽으로 이만큼 갔다가 오른쪽으로 이만큼 돌아옵니다.")

        # 2번(좌우) 모터. 촬영 내내 가운데 → 왼쪽 → 오른쪽 → 가운데 로 한 번 훑는다.
        self.span2_spin = QSpinBox()
        self.span2_spin.setRange(0, 720)      # 한 바퀴가 넘는 왕복도 쓴다 (실측: ±400도)
        self.span2_spin.setValue(400)
        self.span2_spin.setSuffix(" 도")
        self.span2_spin.setToolTip(
            "촬영하는 동안 좌우 모터가 가운데 → 왼쪽 → 오른쪽 → 가운데 로 한 번 움직입니다.\n"
            "이 장비에서는 400도가 끝에서 끝까지입니다.\n"
            "회전판과 동시에 움직여서 같은 빵을 더 여러 각도로 담습니다.\n"
            "0 으로 두면 좌우 모터는 움직이지 않습니다.")

        form.addRow("라벨 방식", self.auto_label_check)
        form.addRow("턴테이블", self.turn_check)
        form.addRow("왕복 각도 (±)", self.span_spin)
        self.roi_spin = QSpinBox()
        self.roi_spin.setRange(0, 100)
        self.roi_spin.setValue(60)
        self.roi_spin.setSuffix(" %")
        self.roi_spin.setToolTip(
            "화면 가운데 이 범위 안에 있는 물체만 잡습니다 (한 개 방식).\n"
            "가장자리의 케이블·손·트레이 모서리를 무시해서 박스가 덜 튑니다.\n"
            "0 으로 두면 화면 전체를 봅니다.")
        form.addRow("가운데 영역", self.roi_spin)
        form.addRow("좌우 모터 (±)", self.span2_spin)
        self.smooth_check = QCheckBox("끊김 없이 계속 움직이기")
        self.smooth_check.setChecked(True)
        self.smooth_check.setToolTip(
            "촬영 시간 내내 두 모터가 아주 느리게 계속 움직입니다 (권장).\n"
            "끄면 한 장 찍을 때마다 조금 움직였다 멈추는 예전 방식으로 돕니다.")
        form.addRow("움직임", self.smooth_check)
        right.addWidget(settings_group)

        # 화면에 빵이 여러 개일 때 — 모델이 아는 것은 자동으로 라벨이 붙고(초록),
        # 모르는 것만 여기서 골라 품목을 준다(주황 → 파랑).
        pick_group = QGroupBox("박스 고르기 — 이름 바로잡기")
        pick = QVBoxLayout(pick_group)

        pick_hint = QLabel(
            "데이터에 없는 새 빵은 모델이 비슷한 기존 빵으로 잘못 부릅니다.\n"
            "그 박스를 골라 올바른 품목으로 바꿔 두고 촬영하세요.\n\n"
            "🔴 빨강 = 지금 고른 것\n"
            "🟢 초록 = 모델이 부른 그대로 저장\n"
            "🟡 노랑 = 신뢰도 55% 미만 — 이름이 틀렸을 수 있음\n"
            "🔵 파랑 = 내가 바꾼 것\n"
            "🟠 주황 = 이름 없음 — 지정하거나 무시해야 저장됩니다\n\n"
            "고정하면 주황은 모두 숨겨지고, 인식이 끊겨도\n"
            "빨간 테두리가 그 자리에 머뭅니다."
        )
        pick_hint.setObjectName("PageDesc")
        pick_hint.setWordWrap(True)
        pick.addWidget(pick_hint)

        next_obj = QPushButton("다음 ▶")
        next_obj.setToolTip("고를 박스를 바꿉니다. 고른 것은 빨간 테두리가 됩니다.\n"
                            "초록(모델이 부른 것)도 고를 수 있습니다.")
        next_obj.clicked.connect(self.cycle_collect_focus)
        pick.addWidget(next_obj)

        self.lock_btn = QPushButton("이것으로 고정 (촬영 대상)")
        self.lock_btn.setObjectName("Success")
        self.lock_btn.setToolTip(
            "고른 박스를 촬영 대상으로 고정합니다.\n"
            "고정하면 인식이 잠깐 끊겨도 빨간 테두리가 그 자리에 머물고,\n"
            "이름 없는 주황 박스는 화면에서 사라집니다.")
        self.lock_btn.clicked.connect(self.toggle_collect_lock)
        pick.addWidget(self.lock_btn)

        assign_obj = QPushButton("이것을 위 품목으로 지정")
        assign_obj.setObjectName("Primary")
        assign_obj.setToolTip("고른 박스의 라벨을 「수집할 상품」 으로 덮어씁니다.\n"
                              "모델이 뭐라고 불렀든 이 품목으로 저장됩니다.")
        assign_obj.clicked.connect(self.assign_collect_focus)
        pick.addWidget(assign_obj)

        ignore_obj = QPushButton("이건 빵이 아님 (무시)")
        ignore_obj.setToolTip("책상·손·그림자, 또는 모델 오검출을 후보에서 뺍니다.\n"
                              "그대로 두면 저장이 막히거나 틀린 라벨이 학습됩니다.")
        ignore_obj.clicked.connect(self.ignore_collect_focus)
        pick.addWidget(ignore_obj)

        clear_assign = QPushButton("지정·무시 초기화")
        clear_assign.setToolTip("바꾼 품목과 무시를 모두 되돌립니다 (모델이 부른 대로).")
        clear_assign.clicked.connect(self.clear_collect_assign)
        pick.addWidget(clear_assign)

        self.pick_group = pick_group             # 「여러 개」 방식에서만 보인다
        right.addWidget(pick_group)

        self.collect_progress = QProgressBar()
        self.collect_progress.setRange(0, 100)
        right.addWidget(self.collect_progress)

        self.collect_status = QLabel("현재 상태: 대기 중")
        self.collect_status.setStyleSheet(
            "background:#EEF3F8; border:1px solid #D5DDE5; padding:10px; border-radius:7px;"
        )
        self.collect_status.setWordWrap(True)
        right.addWidget(self.collect_status)

        cam_on = QPushButton("수집 카메라 시작")
        cam_on.setObjectName("Primary")
        cam_on.setToolTip("카메라를 켜서 신규 빵 배치 상태를 확인합니다.")
        cam_on.clicked.connect(self.start_camera)

        collect_start = QPushButton("자동 수집 시작")
        collect_start.setObjectName("Success")
        collect_start.setToolTip(
            "설정한 간격으로 자동 촬영하며 JPG 이미지와 YOLO TXT 라벨을 함께 저장합니다."
        )
        collect_start.clicked.connect(self.start_collection)

        collect_stop = QPushButton("수집 중지")
        collect_stop.setObjectName("Danger")
        collect_stop.setToolTip("진행 중인 자동 촬영을 즉시 중지합니다.")
        collect_stop.clicked.connect(self.stop_collection)

        export_btn = QPushButton("학습용 ZIP으로 내보내기")
        export_btn.setToolTip(
            "수집한 사진과 라벨을 dataset/import/ 에 ZIP 으로 저장합니다.\n"
            "그 뒤 「모델 배포 및 모델 학습」 탭에서 Train/Val 분할 → 학습을 진행하세요."
        )
        export_btn.clicked.connect(self.export_collected_zip)

        clear_btn = QPushButton("수집 데이터 비우기")
        clear_btn.setObjectName("Danger")
        clear_btn.setToolTip(
            "지금까지 찍은 사진과 라벨을 모두 지웁니다.\n"
            "지우기 전에 몇 장인지, 어떤 품목인지 보여주고 한 번 더 물어봅니다."
        )
        clear_btn.clicked.connect(self.clear_collected_data)

        right.addWidget(cam_on)
        right.addWidget(collect_start)
        right.addWidget(collect_stop)
        right.addWidget(export_btn)
        right.addWidget(clear_btn)

        guide = QLabel()
        guide.setWordWrap(True)
        guide.setStyleSheet(
            "background:#0C1722; border:1px solid #24405B; padding:12px; border-radius:8px; color:#AFC4D9;"
        )
        self.collect_guide = guide
        right.addWidget(guide)
        right.addStretch()

        content.addWidget(left_card, 3)
        content.addWidget(right_card, 1)
        root.addLayout(content)
        self.apply_collect_mode()
        return w

    def refresh_collect_combo(self):
        if not hasattr(self, "collect_product_combo"):
            return

        current = self.collect_product_combo.currentData()
        self.collect_product_combo.clear()

        for cid, p in sorted(self.products.items(), key=lambda x: int(x[0])):
            count = self.count_collected_for_class(p["class_name"])
            self.collect_product_combo.addItem(
                f"[Class {cid}] {p['display_name']} — 현재 {count}장",
                int(cid)
            )

        if current is not None:
            idx = self.collect_product_combo.findData(current)
            if idx >= 0:
                self.collect_product_combo.setCurrentIndex(idx)

    # ---- 카메라 고르기 ---------------------------------------------------
    def jetson_stream_url(self):
        """젯슨이 내보내는 카메라 영상(8090) 주소. 모르면 None."""
        url = os.environ.get("BREADVISION_JETSON_STREAM", "").strip()
        if url:
            return url
        env = os.environ.get("BREADVISION_CAMERA", "").strip()
        if env.startswith("http"):
            return env
        try:
            port = int(json.loads((BASE_DIR / "jetson.json").read_text(encoding="utf-8"))
                       .get("stream_port", 8090))
        except Exception:
            port = 8090
        infer = os.environ.get("BREADVISION_INFER_URL", "").strip()
        if infer:
            from urllib.parse import urlparse
            host = urlparse(infer).hostname
            if host:
                return f"http://{host}:{port}/stream"
        if Path("/etc/nv_tegra_release").exists():       # 젯슨 자신에서 켠 경우
            return f"http://127.0.0.1:{port}/stream"
        return None

    def resolve_camera_source(self, key=None):
        """저장된 선택(key) → CameraManager 에 줄 값 (카메라 번호 또는 영상 주소)."""
        if key is None:
            key = self.settings.get("camera_source", "")
        key = str(key).strip()
        if key == "jetson":
            return self.jetson_stream_url() or 0
        if key.isdigit():
            return int(key)
        if key:
            return key                                   # 직접 넣은 주소
        env = os.environ.get("BREADVISION_CAMERA", "").strip()   # 고른 적이 없으면 예전 방식
        if env:
            return int(env) if env.isdigit() else env
        return 0

    def camera_key_of(self, src):
        if isinstance(src, int):
            return str(src)
        return "jetson" if src == self.jetson_stream_url() else str(src)

    def camera_source_label(self, src):
        if isinstance(src, int):
            return f"PC 카메라 {src}"
        if src == self.jetson_stream_url():
            return "젯슨 카메라"
        return str(src)

    def refresh_camera_combo(self, found=None):
        """found: {번호: "1280x720"} — 「카메라 찾기」 결과. None 이면 0~2번을 그냥 보여준다."""
        combo = self.camera_combo
        combo.blockSignals(True)
        combo.clear()
        if self.jetson_stream_url():
            combo.addItem("📷 젯슨 카메라 (8090)", "jetson")
        indices = found if found is not None else {i: "" for i in range(3)}
        cur = self.camera_key_of(self.camera.index) if hasattr(self, "camera") else ""
        if cur.isdigit() and int(cur) not in indices:
            indices = {**indices, int(cur): ""}
        for i in sorted(indices):
            label = f"💻 PC 카메라 {i}" + (" (보통 내장)" if i == 0 else "")
            if indices[i]:
                label += f"  {indices[i]}"
            combo.addItem(label, str(i))
        if cur and cur != "jetson" and not cur.isdigit():
            combo.addItem(f"🔗 {cur}", cur)
        combo.addItem("✏️ 번호/주소 직접 입력…", "__custom__")
        idx = combo.findData(cur)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def on_camera_combo(self, i):
        key = self.camera_combo.itemData(i)
        if key == "__custom__":
            text, ok = QInputDialog.getText(
                self, "카메라 직접 입력",
                "카메라 번호 또는 영상 주소를 넣으세요.\n\n"
                "예) 1\n예) http://192.168.0.23:8090/stream",
                text=str(self.camera.index),
            )
            if not ok or not text.strip():
                self.refresh_camera_combo()
                return
            key = text.strip()
        self.switch_camera(key)

    def switch_camera(self, key):
        src = self.resolve_camera_source(key)
        self.stop_camera()
        self.camera = CameraManager(src, use_env=False)
        self.settings["camera_source"] = key
        self.save_settings()
        self.refresh_camera_combo()
        self.start_camera()                              # 고르면 바로 켜서 보여준다

    def scan_cameras(self):
        """이 컴퓨터에 연결된 카메라 번호를 찾는다."""
        from camera.camera import _backends
        using_local = isinstance(self.camera.index, int) and self.camera.cap is not None
        if using_local:
            self.stop_camera()                           # 같은 카메라를 두 번 열 수 없다
        # 번호마다 백엔드를 다 시도하면 윈도우에서 한참 걸린다. 그 OS 의 첫 방식만 쓴다.
        name, api = next((n, a) for n, a in _backends() if not isinstance(a, str))
        found = {}
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for i in range(5):
                cap = cv2.VideoCapture(i, api)
                ok = cap.isOpened() and cap.read()[0]
                if ok:
                    found[i] = (f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                                f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                cap.release()
        finally:
            QApplication.restoreOverrideCursor()
        self.refresh_camera_combo(found)
        if using_local:
            self.start_camera()
        if found:
            lines = "\n".join(f"  · {i}번  {s}" for i, s in sorted(found.items()))
            msg = (f"이 컴퓨터에서 찾은 카메라:\n{lines}\n\n"
                   "보통 0번이 노트북 내장 카메라, 1번이 USB 카메라입니다.\n"
                   "카메라 목록에서 골라 주세요.")
        else:
            msg = "이 컴퓨터에서 쓸 수 있는 카메라를 찾지 못했습니다.\n(젯슨 카메라는 목록의 「젯슨 카메라」를 고르세요)"
        if self.jetson_stream_url():
            msg += f"\n\n젯슨 카메라 주소: {self.jetson_stream_url()}"
        QMessageBox.information(self, "카메라 찾기", msg)

    def start_camera(self):
        try:
            if self.camera.cap is None:
                self.camera.start()
            if not self.camera_timer.isActive():
                self.camera_timer.start(int(self.CAMERA_INTERVAL * 1000))

            self.camera_state.setText(f"CAMERA  ON · {self.camera_source_label(self.camera.index)[:24]}")
            self.camera_state.setStyleSheet(
                "background:#0E2740; color:#7CC0FF; padding:8px 12px; "
                "border:1px solid #245B91; border-radius:6px; font-weight:800;"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "카메라 연결 실패",
                f"{self.camera_source_label(self.camera.index)} 을(를) 열지 못했습니다.\n\n{e}\n\n"
                "POS 화면 아래 카메라 목록에서 다른 카메라를 골라 보세요.\n"
                "(젯슨 카메라가 안 되면 젯슨이 켜져 있는지, 같은 네트워크인지 확인)"
            )

    def stop_camera(self):
        self.last_qr_data = ""
        self.last_qr_points = None
        self.qr_seen_frames = 0
        self.qr_state = None
        if hasattr(self, "qr_live_label"):
            self.qr_live_label.setText("QR 대기 중")
        self.camera_timer.stop()
        self.collect_timer.stop()
        self.collecting = False
        self.camera.stop()
        for name in ("pos_camera", "collect_camera"):
            lab = getattr(self, name, None)
            if lab is not None:
                self._reset_preview_size(lab)

        if hasattr(self, "camera_state"):
            self.camera_state.setText("CAMERA  OFF")
            self.camera_state.setStyleSheet(
                "background:#131D27; color:#91A3B5; padding:8px 12px; "
                "border:1px solid #2B3947; border-radius:6px; font-weight:800;"
            )

    def _decode_qr_robust(self, frame, fast=False):
        """Try several OpenCV QR decoding paths and return (data, points).

        fast=True 는 라이브 프리뷰용. 축소한 흑백 한 장만 본다 (1280x720 기준 약 16ms).
        fast=False 는 「회원 QR 스캔」 버튼용. 원본/흑백/CLAHE/확대를 모두 시도한다 (약 240ms).
        """
        if frame is None:
            return "", None

        # (이미지, 좌표를 원본 크기로 되돌릴 배율)
        candidates = []

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if fast:
                # 프리뷰에서는 축소본 한 장만. QR은 큼직해서 축소해도 잘 잡힌다.
                h, w = gray.shape[:2]
                scale = self.QR_FAST_WIDTH / float(w) if w > self.QR_FAST_WIDTH else 1.0
                if scale < 1.0:
                    small = cv2.resize(gray, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
                    candidates.append((small, 1.0 / scale))
                else:
                    candidates.append((gray, 1.0))
            else:
                candidates.append((frame, 1.0))
                candidates.append((gray, 1.0))

                # Mobile screens can be reflective/low contrast. CLAHE often helps.
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                candidates.append((clahe.apply(gray), 1.0))

                # Slight upscale helps small QR codes.
                h, w = gray.shape[:2]
                if min(h, w) < 900:
                    candidates.append((
                        cv2.resize(gray, None, fx=1.5, fy=1.5,
                                   interpolation=cv2.INTER_CUBIC),
                        1.0 / 1.5,
                    ))
        except Exception:
            candidates = [(frame, 1.0)]

        for img, back in candidates:
            try:
                data, points, _ = self.qr_detector.detectAndDecode(img)
                if data:
                    # 축소/확대본에서 찾았으면 좌표를 원본 크기로 되돌린다.
                    if points is not None and back != 1.0:
                        try:
                            points = points * back
                        except Exception:
                            pass
                    return data.strip(), points
            except Exception:
                pass

        return "", None

    def _draw_qr_overlay(self, frame, data, points):
        if points is None:
            return frame            # 그릴 게 없으면 복사도 하지 않는다
        display = frame.copy()

        try:
            pts = points.reshape(-1, 2).astype(int)
            if len(pts) >= 4:
                for i in range(len(pts)):
                    p1 = tuple(pts[i])
                    p2 = tuple(pts[(i + 1) % len(pts)])
                    cv2.line(display, p1, p2, (0, 255, 0), 4)

                x = int(min(p[0] for p in pts))
                y = int(min(p[1] for p in pts))
                label = "MEMBER QR" if data.startswith("BV1:") else "QR DETECTED"
                cv2.putText(
                    display, label, (max(10, x), max(35, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA
                )
        except Exception:
            pass
        return display

    QR_LABEL_STYLE = {
        "member": ("● 회원 QR 인식됨 — 스캔 버튼을 누르세요",
                   "color:#35D07F; font-size:12px; font-weight:900;"),
        "other": ("● QR 인식됨 — BreadVision 회원 QR은 아닙니다",
                  "color:#F3B64A; font-size:12px; font-weight:900;"),
        "none": ("QR 대기 중 — 휴대폰 QR을 카메라에 보여주세요",
                 "color:#91A3B5; font-size:12px; font-weight:700;"),
    }

    def _set_qr_state(self, state):
        """QR 안내 라벨 갱신. 상태가 바뀔 때만 건드린다.

        setStyleSheet() 는 스타일 재계산 + 위젯 다시 그리기를 유발한다.
        매 프레임 호출하면 그것만으로 이벤트 루프가 계속 바쁘다.
        """
        if state == self.qr_state:
            return
        self.qr_state = state
        if not hasattr(self, "qr_live_label"):
            return
        text, style = self.QR_LABEL_STYLE[state]
        self.qr_live_label.setText(text)
        self.qr_live_label.setStyleSheet(style)

    def class_name_map(self):
        """{class_id: 영문 class_name} — cv2 로 그릴 라벨용 (한글은 못 그린다)"""
        return {int(cid): p["class_name"] for cid, p in self.products.items()}

    @staticmethod
    def _iou(a, b):
        """두 박스가 얼마나 겹치는가 (0~1)."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    MATCH_IOU = 0.35        # 이 이상 겹치면 "같은 물체" 로 본다

    # 같은 빵도 움직이면 0.45 까지 떨어진다. 그래서 문턱은 낮게 두고,
    # 명백히 다른 것(초코머핀 vs 바게트 = 0.00)만 잘라낸다.
    # 애매한 구간은 점수에 곱해서 가까운 쪽이 이기게 한다.
    LOCK_SIG_MIN = 0.35      # 겉모습이 이만큼도 안 닮았으면 다른 빵이다
    # 추적기와 이만큼 겹치는 검출만 믿는다.
    # 낮게 잡으면(0.4) 바로 옆에 붙은 다른 빵이 통과해 버린다. 추적기는 한 프레임
    # 뒤처지므로 진짜 같은 빵이라도 겹침이 0.75 언저리까지 떨어지는데,
    # 옆 빵은 0.5 근처였다. 그 사이인 0.6 으로 가른다.
    LOCK_SNAP_IOU = 0.60

    @staticmethod
    def _make_tracker(frame, box):
        """고정한 빵을 그림으로 따라가는 추적기.

        검출만으로는 안 된다. 신뢰도가 55% 근처에서 흔들리면 초록↔노랑을
        오가다 아예 안 잡히는 프레임이 생기고, 그때 옆 빵이 그 자리를 차지하면
        빨강이 옮겨 붙는다. 특히 새 빵이 기존 빵과 애매하게 겹쳐 있으면
        잘라낸 그림까지 비슷해져서 겉모습으로도 못 가른다.

        추적기는 모델을 거치지 않고 화면의 그림만 보고 따라가므로
        인식이 끊긴 구간에서도 그 빵을 놓치지 않는다.

        KCF 를 쓴다. CSRT 가 더 정확하지만 젯슨에서 37ms/프레임이라 미리보기가
        느려진다 (KCF 24ms). 어차피 검출과 맞아떨어질 때마다 자리를 다시
        잡아 주므로 조금씩 밀리는 것은 따라잡힌다.
        """
        x1, y1, x2, y2 = [int(v) for v in box]
        w, h = x2 - x1, y2 - y1
        if w < 8 or h < 8:
            return None
        try:
            t = cv2.TrackerKCF_create()
            t.init(frame, (x1, y1, w, h))
            return t
        except Exception:
            return None

    @staticmethod
    def _track(tracker, frame):
        """추적기가 알려주는 이번 프레임의 자리. 놓쳤으면 None."""
        if tracker is None:
            return None
        try:
            ok, box = tracker.update(frame)
        except Exception:
            return None
        if not ok:
            return None
        x, y, w, h = box
        if w < 4 or h < 4:
            return None
        return (float(x), float(y), float(x + w), float(y + h))

    LOCK_MAX_MISS = 12       # 이만큼 연달아 놓치면 따라가기를 멈춘다

    @staticmethod
    def _patch_sig(frame, box):
        """박스 안 그림을 8x8 로 줄인 값. 같은 빵인지 겉모습으로 확인하는 용도.

        위치와 크기만 보면 "내 빵이 옆으로 갔다" 와 "옆 빵이 내 자리로 왔다" 를
        구분할 수 없다. 턴테이블이 돌면 두 상황이 실제로 번갈아 일어난다.
        색과 무늬까지 보면 갈린다 — 초코머핀과 바게트는 절대 안 닮았다.
        """
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, int(box[0])))
        y1 = max(0, min(h - 1, int(box[1])))
        x2 = max(x1 + 1, min(w, int(box[2])))
        y2 = max(y1 + 1, min(h, int(box[3])))
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        small = cv2.resize(patch, (8, 8), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32).ravel()

    @staticmethod
    def _sig_match(a, b):
        """두 겉모습이 얼마나 닮았나 (0~1)."""
        if a is None or b is None:
            return 1.0                      # 비교할 수 없으면 막지 않는다
        diff = float(np.abs(a - b).mean())
        return max(0.0, 1.0 - diff / 40.0)  # 채널당 평균 40 차이면 완전히 다른 것

    def _match_locked(self, lock_box, boxes, frame=None, sig=None):
        """고정한 물체가 이번 프레임의 어느 박스인지 찾는다. 없으면 None.

        세 가지를 함께 본다.
          · 중심이 얼마나 멀어졌나 — 자기 크기의 60% 밖이면 다른 물체
          · 크기가 비슷한가 — 넓이가 2.5배 이상 차이나면 다른 물체
          · 겉모습이 닮았나 — 색·무늬가 다르면 옆 빵이다 (이게 결정적이다)
        """
        lx1, ly1, lx2, ly2 = lock_box
        lcx, lcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
        lw, lh = lx2 - lx1, ly2 - ly1
        larea = max(1.0, float(lw * lh))
        limit = 0.6 * max(lw, lh)
        if limit <= 0:
            return None

        best, best_score = None, 0.0
        for bx in boxes:
            x1, y1, x2, y2 = bx["xyxy"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
            if dist > limit:
                continue
            area = max(1.0, float((x2 - x1) * (y2 - y1)))
            ratio = min(area, larea) / max(area, larea)
            if ratio < 0.4:
                continue
            look = 1.0
            if sig is not None and frame is not None:
                look = self._sig_match(sig, self._patch_sig(frame, bx["xyxy"]))
                if look < self.LOCK_SIG_MIN:
                    continue                # 겉모습이 다르면 아무리 가까워도 아니다
            score = (1.0 - dist / limit) * ratio * look
            if score > best_score:
                best, best_score = bx, score
        return best

    def collect_label_boxes(self, frame, detections=None):
        """이 프레임에 저장될 라벨들. [{xyxy, class_id, model_id, fixed, source}, ...]

        화면에 있는 것을 **전부** 잡는다. 두 가지 경로를 합친다.

          source="model"  모델이 빵이라고 잡은 것. 클래스가 이미 있다.
                          데이터에 없는 새 빵은 여기서 "비슷한 기존 빵" 으로
                          잘못 불린다. 그래서 이것도 골라 바꿀 수 있어야 한다.
          source="new"    모델이 못 본 덩어리 (배경 분리). class_id 가 없으므로
                          품목을 주거나 무시해야 저장된다.

        한 장에 빵이 여러 개면 전부 라벨을 달아야 한다. 하나라도 빠지면
        YOLO 가 그것을 배경으로 학습해 앞으로 못 잡게 된다.
        """
        h, w = frame.shape[:2]
        area_all = float(w * h)
        if detections is None:
            detections = self.live_dets

        boxes = []
        for d in (detections or []):
            if d.get("confidence", 0) < self.AUTOLABEL_CONF:
                continue
            x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
            ratio = ((x2 - x1) * (y2 - y1)) / area_all
            if ratio < self.AUTOLABEL_MIN_AREA or ratio > self.AUTOLABEL_MAX_AREA:
                continue
            cid = int(d["class_id"])
            boxes.append({"xyxy": (x1, y1, x2, y2), "class_id": cid, "model_id": cid,
                          "fixed": False, "locked": False, "source": "model",
                          "conf": float(d.get("confidence", 0))})

        # 모델이 못 본 덩어리 = 아직 이름이 없는 물체
        for box in object_boxes(frame,
                                min_area=self.AUTOLABEL_MIN_AREA,
                                max_area=self.AUTOLABEL_MAX_AREA):
            if any(self._iou(box, b["xyxy"]) >= self.MATCH_IOU for b in boxes):
                continue
            boxes.append({"xyxy": box, "class_id": None, "model_id": None,
                          "fixed": False, "locked": False, "source": "new", "conf": 0.0})

        # 빵이 아니라고 표시한 것은 뺀다 (책상·손·모델 오검출).
        # 그대로 두면 저장이 막히거나 틀린 라벨이 학습된다.
        if self.collect_ignore:
            alive, kept_ignore = [], []
            for b in boxes:
                match = next((b["xyxy"] for ig in self.collect_ignore
                              if self._iou(ig, b["xyxy"]) >= self.MATCH_IOU), None)
                if match is None:
                    alive.append(b)
                else:
                    kept_ignore.append(match)        # 움직였으면 자리도 갱신
            boxes, self.collect_ignore = alive, kept_ignore

        # 사용자가 바꿔 준 품목을 따라붙인다. 빵을 움직이며 찍으므로
        # 겹침으로 같은 물체를 찾아 지정을 이어주고, 그 자리도 갱신한다.
        kept = []
        for old_box, cid in self.collect_manual:
            best, best_iou = None, self.MATCH_IOU
            for b in boxes:
                v = self._iou(old_box, b["xyxy"])
                if v > best_iou:
                    best, best_iou = b, v
            if best is not None:
                best["class_id"] = cid
                best["fixed"] = True
                kept.append((best["xyxy"], cid))
        self.collect_manual = kept

        # ── 촬영 대상 고정
        #
        # 검출만 보고 따라가면 빨강이 옆 빵으로 튄다. 새 빵은 신뢰도가 55% 근처에서
        # 흔들려 초록↔노랑을 오가다 아예 안 잡히는 프레임이 생기는데, 그때
        # 겹쳐 있던 기존 빵이 그 자리를 차지해 버리기 때문이다.
        #
        #   1) 그림을 직접 따라가는 추적기가 기준이다 (검출과 무관하게 돈다)
        #   2) 검출은 자리를 다듬는 데만 쓴다 — 추적기와 충분히 겹치고
        #      겉모습도 닮은 것 하나만 받아들인다
        #   3) 둘 다 놓치면 직전 속도만큼만 흘려보내고 자리를 지킨다
        #   4) 이름 없는 것(주황)은 화면에서도 라벨에서도 뺀다
        if self.collect_lock:
            lock = self.collect_lock
            vx, vy = lock.get("vel", (0.0, 0.0))
            x1, y1, x2, y2 = lock["xyxy"]

            # 1) 그림으로 따라간다. 검출이 끊겨도, 옆 빵과 겹쳐도 흔들리지 않는다.
            tracked = self._track(self.lock_tracker, frame)

            # 2) 검출로 자리를 다듬는다.
            if tracked is not None:
                # 추적기는 한 프레임 뒤처지므로 직전 속도만큼 앞으로 밀어 두고 잰다.
                # 이걸 안 하면 옆 빵이 마침 한두 프레임 전 내 자리에 와 있을 때
                # 겹침이 똑같이 나와 어느 쪽인지 가릴 수 없다.
                # 겉모습까지 곱해 같은 빵 쪽이 이기게 한다.
                ax1, ay1, ax2, ay2 = tracked
                aim = (ax1 + vx, ay1 + vy, ax2 + vx, ay2 + vy)
                sig = lock.get("sig")
                best, best_score = None, 0.0
                for cand in boxes:
                    v = self._iou(aim, cand["xyxy"])
                    if v < self.LOCK_SNAP_IOU:
                        continue
                    look = self._sig_match(sig, self._patch_sig(frame, cand["xyxy"]))
                    if look < self.LOCK_SIG_MIN:
                        continue
                    score = v * look
                    if score > best_score:
                        best, best_score = cand, score
            else:
                predicted = (x1 + vx, y1 + vy, x2 + vx, y2 + vy)
                best = self._match_locked(predicted, boxes, frame, lock.get("sig"))

            if best is not None:
                ox, oy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                nx1, ny1, nx2, ny2 = best["xyxy"]
                dx = (nx1 + nx2) / 2.0 - ox
                dy = (ny1 + ny2) / 2.0 - oy
                lock["vel"] = (0.5 * vx + 0.5 * dx, 0.5 * vy + 0.5 * dy)
                lock["xyxy"] = best["xyxy"]
                lock["miss"] = 0
                new_sig = self._patch_sig(frame, best["xyxy"])
                if new_sig is not None:
                    old_sig = lock.get("sig")
                    lock["sig"] = (new_sig if old_sig is None
                                   else 0.8 * old_sig + 0.2 * new_sig)
                # 추적기가 밀렸으면 검출 자리로 다시 잡아 준다
                if tracked is None or self._iou(tracked, best["xyxy"]) < 0.75:
                    self.lock_tracker = self._make_tracker(frame, best["xyxy"])
                best["class_id"] = lock["class_id"]
                best["fixed"] = True
                best["locked"] = True

            elif tracked is not None:
                # 검출은 없지만 추적기는 보고 있다. 그 자리를 그대로 쓴다.
                # 이 경우가 핵심이다 — 신뢰도가 오르내려 인식이 끊긴 구간에서도
                # 빨강이 그 빵에 붙어 있게 된다.
                lock["vel"] = ((tracked[0] - x1) * 0.5 + vx * 0.5,
                               (tracked[1] - y1) * 0.5 + vy * 0.5)
                lock["xyxy"] = tracked
                lock["miss"] = 0
                boxes.append({"xyxy": tracked, "class_id": lock["class_id"],
                              "model_id": None, "fixed": True, "locked": True,
                              "source": "lock", "conf": 0.0})

            else:
                # 추적기까지 놓쳤다. 직전 속도만큼만 흘려보내고 자리를 지킨다.
                miss = lock.get("miss", 0) + 1
                lock["miss"] = miss
                if miss <= self.LOCK_MAX_MISS and (abs(vx) > 0.3 or abs(vy) > 0.3):
                    h_, w_ = frame.shape[:2]
                    nx1 = max(-20.0, min(w_ + 20.0, x1 + vx))
                    ny1 = max(-20.0, min(h_ + 20.0, y1 + vy))
                    lock["xyxy"] = (nx1, ny1, nx1 + (x2 - x1), ny1 + (y2 - y1))
                lock["vel"] = (vx * 0.7, vy * 0.7)
                # 추적기를 이 자리에서 다시 세워 본다
                self.lock_tracker = self._make_tracker(frame, lock["xyxy"])
                boxes.append({"xyxy": lock["xyxy"], "class_id": lock["class_id"],
                              "model_id": None, "fixed": True, "locked": True,
                              "source": "lock", "conf": 0.0})

            boxes = [b for b in boxes if b["class_id"] is not None]

        boxes.sort(key=lambda b: (b["xyxy"][0] + b["xyxy"][2]) / 2)   # 왼쪽부터
        return boxes

    def new_boxes(self):
        """고를 수 있는 박스 = 화면에서 잡힌 것 전부 (모델이 부른 것 포함)."""
        return list(self.collect_boxes)

    def cycle_collect_focus(self):
        """「다음 ▶」 — 고를 박스를 옮긴다. 고정 중에는 움직이지 않는다."""
        if self.collect_lock:
            self.collect_status.setText(
                "현재 상태: 촬영 대상이 고정되어 있습니다.\n"
                "다른 것을 고르려면 「고정 해제」 를 먼저 누르세요.")
            return
        n = len(self.new_boxes())
        if n == 0:
            self.collect_status.setText("현재 상태: 인식된 빵이 없습니다.\n"
                                        "빵을 화면에 올려주세요.")
            return
        self.collect_focus = (self.collect_focus + 1) % n
        self.update_collect_hint()

    def assign_collect_focus(self):
        """포커스한 물체에 콤보박스의 품목을 지정한다."""
        news = self.new_boxes()
        if not news:
            self.collect_status.setText("현재 상태: 바꿀 빵이 없습니다.")
            return
        if self.collect_product_combo.count() == 0:
            return
        idx = min(self.collect_focus, len(news) - 1)
        cid = int(self.collect_product_combo.currentData())
        box = news[idx]["xyxy"]
        # 같은 자리에 이미 지정한 것이 있으면 갈아끼운다
        self.collect_manual = [(b, c) for b, c in self.collect_manual
                               if self._iou(b, box) < self.MATCH_IOU]
        self.collect_manual.append((box, cid))
        news[idx]["class_id"] = cid
        news[idx]["fixed"] = True        # 다음 프레임을 기다리지 않고 바로 반영
        self.update_collect_hint()

    def ignore_collect_focus(self):
        """포커스한 것이 빵이 아니라고 표시한다 (책상·손·그림자)."""
        news = self.new_boxes()
        if not news:
            self.collect_status.setText("현재 상태: 무시할 빵이 없습니다.")
            return
        box = news[min(self.collect_focus, len(news) - 1)]["xyxy"]
        self.collect_manual = [(b, c) for b, c in self.collect_manual
                               if self._iou(b, box) < self.MATCH_IOU]
        self.collect_ignore.append(box)
        self.collect_focus = 0
        self.update_collect_hint()

    def toggle_collect_lock(self):
        """고른 박스를 촬영 대상으로 고정한다 (누르면 해제)."""
        if self.collect_lock:
            self.collect_lock = None
            self.lock_tracker = None
            self.lock_btn.setText("이것으로 고정 (촬영 대상)")
            self.update_collect_hint()
            return

        boxes = self.collect_boxes
        if not boxes:
            self.collect_status.setText("현재 상태: 고정할 것이 없습니다.")
            return
        if self.collect_product_combo.count() == 0:
            return
        bx = boxes[min(self.collect_focus, len(boxes) - 1)]
        # 고정 = "지금 이걸 찍는다" 이므로 라벨은 「수집할 상품」 을 따른다.
        # 데이터에 없는 새 빵은 모델이 비슷한 기존 빵으로 부르는데,
        # 그 이름을 그대로 쓰면 틀린 라벨이 쌓인다.
        cid = int(self.collect_product_combo.currentData())
        sig = None
        if self.frame is not None:
            sig = self._patch_sig(self.frame, bx["xyxy"])
        self.collect_lock = {"xyxy": bx["xyxy"], "class_id": cid,
                             "vel": (0.0, 0.0), "sig": sig, "miss": 0}
        self.lock_tracker = (self._make_tracker(self.frame, bx["xyxy"])
                             if self.frame is not None else None)
        if self.lock_tracker is None:
            print("[고정] 추적기를 만들지 못했습니다. 검출만으로 따라갑니다.")
        self.collect_manual = [(b, c) for b, c in self.collect_manual
                               if self._iou(b, bx["xyxy"]) < self.MATCH_IOU]
        self.collect_manual.append((bx["xyxy"], int(cid)))
        bx["class_id"] = int(cid)
        bx["fixed"] = True
        bx["locked"] = True
        self.lock_btn.setText("고정 해제")
        self.update_collect_hint()

    def clear_collect_assign(self):
        """지정·무시·고정을 전부 되돌린다."""
        self.collect_manual = []
        self.collect_ignore = []
        self.collect_lock = None
        self.lock_tracker = None
        self.collect_focus = 0
        if hasattr(self, "lock_btn"):
            self.lock_btn.setText("이것으로 고정 (촬영 대상)")
        self.update_collect_hint()

    def update_collect_hint(self):
        """무엇이 어떤 이름으로 저장될지 보여준다."""
        if not hasattr(self, "collect_status"):
            return
        if not self.collect_boxes:
            self.collect_status.setText("현재 상태: 잡힌 물체가 없습니다.\n"
                                        "빵을 화면에 올려주세요.")
            return
        names = self.products
        lines, todo = [], 0
        if self.collect_lock:
            cid = str(self.collect_lock["class_id"])
            nm = names.get(cid, {}).get("display_name", "?")
            lines.append(f"🔒 고정: {nm} — 이것만 찍습니다")
        for i, b in enumerate(self.collect_boxes):
            mark = "🔒" if b.get("locked") else ("▶" if i == self.collect_focus else "  ")
            if b["class_id"] is None:
                todo += 1
                lines.append(f"{mark} [{i + 1}] 이름 없음 — 지정하거나 무시")
                continue
            now = names.get(str(b["class_id"]), {}).get("display_name", "?")
            if b["fixed"] and b["model_id"] is not None and b["model_id"] != b["class_id"]:
                was = names.get(str(b["model_id"]), {}).get("display_name", "?")
                lines.append(f"{mark} [{i + 1}] {now}   ← 모델은 {was} 라고 함")
            elif b["fixed"]:
                lines.append(f"{mark} [{i + 1}] {now} (내가 지정)")
            elif b["conf"] < self.LOW_CONF:
                lines.append(f"{mark} [{i + 1}] {now} ⚠ {b['conf'] * 100:.0f}% — 확인 필요")
            else:
                lines.append(f"{mark} [{i + 1}] {now} (모델 그대로)")
        if todo:
            lines.append(f"※ 이름 없는 것 {todo}개 — 저장되지 않습니다")
        self.collect_status.setText("\n".join(lines))

    # 박스 색 (BGR)
    BOX_FOCUS = (0, 0, 235)       # 빨강 — 지금 고른 것
    BOX_MODEL = (0, 220, 0)       # 초록 — 모델이 부른 그대로 저장
    BOX_FIXED = (255, 170, 0)     # 파랑 — 내가 품목을 바꾼 것
    BOX_TODO = (0, 165, 255)      # 주황 — 이름이 없다. 지정하거나 무시해야 저장된다
    BOX_LOW = (0, 255, 255)       # 노랑 — 모델이 확신하지 못한다 (50% 미만)

    # ---- 수집 방식 (한 개 / 여러 개) -------------------------------------
    GUIDE_SINGLE = (
        "사용 순서 — 한 개 방식\n"
        "① 수집할 상품 선택 → ② 카메라 켜기 → ③ 흰 바탕에 빵 1개만 올리기 → "
        "④ 자동 수집 시작\n\n"
        "초록 박스 = 저장될 라벨입니다. 「NO BOX」 가 뜨면 그 장면은 저장하지 않습니다.\n"
        "화면에서 가장 뚜렷한 물체 하나만 잡으니, 다른 빵·손·물건이 같이 찍히지 않게 하세요.\n\n"
        "다 찍으면 「학습용 ZIP으로 내보내기」 → 「모델 배포 및 모델 학습」 탭에서\n"
        "분할(합성 켜기) → 전이학습 순서로 진행합니다."
    )
    GUIDE_MULTI = (
        "사용 순서 — 여러 개 방식\n"
        "① 품목 선택 → ② 카메라 켜기 → ③ 새 빵을 「다음 ▶」 으로 골라 「이것을 위 품목으로 지정」 → "
        "④ 자동 데이터 수집 시작\n\n"
        "주황(이름 없음)이 남아 있으면 저장되지 않습니다. 지정하거나 「무시」 하세요."
    )

    def collect_mode(self):
        combo = getattr(self, "collect_mode_combo", None)
        return combo.currentData() if combo is not None else "single"

    def apply_collect_mode(self):
        single = self.collect_mode() == "single"
        if hasattr(self, "pick_group"):
            self.pick_group.setVisible(not single)
        if hasattr(self, "collect_guide"):
            self.collect_guide.setText(self.GUIDE_SINGLE if single else self.GUIDE_MULTI)

    def on_collect_mode_changed(self, _index=None):
        self.settings["collect_mode"] = self.collect_mode()
        self.save_settings()
        # 여러 개 방식에서 고른·고정한 상태가 남으면 한 개 방식과 섞인다. 비운다.
        if hasattr(self, "collect_status"):
            self.clear_collect_assign()
        self.apply_collect_mode()

    def collect_preview_single(self, frame):
        """한 개 방식 미리보기 (예전 방식): 지금 찍으면 저장될 박스 하나만 초록으로 그린다.

        품목은 콤보박스에서 고른 것으로 저장한다. 모델이 뭐라고 부르든 상관없다.
        """
        view = frame.copy()
        # 가운데 영역 안내선 — 이 안에 빵을 놓으면 된다
        roi = self.roi_spin.value() / 100.0 if hasattr(self, "roi_spin") else 0.0
        if roi > 0:
            h, w = view.shape[:2]
            rx, ry = int(w * (1 - roi) / 2), int(h * (1 - roi) / 2)
            cv2.rectangle(view, (rx, ry), (w - rx, h - ry), (150, 150, 150), 1)
            cv2.putText(view, "put ONE bread here", (rx + 6, ry + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
        box, why = self.label_box_pixels(frame, self.live_dets)
        if box is None:
            # 무엇을 잡았는데 왜 버렸는지 보여준다 (주황 박스)
            cand = getattr(self, "last_segment_candidate", None)
            if cand:
                cv2.rectangle(view, (int(cand[0]), int(cand[1])),
                              (int(cand[2]), int(cand[3])), (0, 165, 255), 2)
            cv2.putText(view, f"NO BOX - {why}", (18, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 235), 2, cv2.LINE_AA)
            cv2.putText(view, "put ONE bread on a plain background", (18, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 235), 2, cv2.LINE_AA)
            return view

        x1, y1, x2, y2 = [int(v) for v in box]
        cid = self.collect_product_combo.currentData()
        name = self.class_name_map().get(int(cid), "?") if cid is not None else "?"
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 0), 3)
        cv2.putText(view, f"{name}  ({why})", (x1, max(24, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)
        return view

    def collect_preview(self, frame):
        """수집 탭 미리보기: 화면에서 잡힌 박스를 전부 그린다.

        고른 것은 빨강이다. 초록·파랑·주황 중 무엇이든 「다음 ▶」 로 고를 수 있다.
        """
        if self.collect_mode() == "single":
            return self.collect_preview_single(frame)
        view = frame.copy()
        self.collect_boxes = self.collect_label_boxes(frame)
        names = self.class_name_map()
        if self.collect_boxes:
            # 고정했으면 빨강은 그것만. 인식이 끊겨도 다른 데로 튀지 않는다.
            locked_at = next((i for i, b in enumerate(self.collect_boxes)
                              if b.get("locked")), None)
            if locked_at is not None:
                self.collect_focus = locked_at
            else:
                self.collect_focus %= len(self.collect_boxes)

        for i, bx in enumerate(self.collect_boxes):
            x1, y1, x2, y2 = [int(v) for v in bx["xyxy"]]
            focused = (i == self.collect_focus)

            if bx["class_id"] is None:
                base, tag = self.BOX_TODO, "SELECT ME"
            elif bx["fixed"]:
                base = self.BOX_FIXED
                tag = names.get(bx["class_id"], "?")
                if bx["model_id"] is not None and bx["model_id"] != bx["class_id"]:
                    tag += f" (was {names.get(bx['model_id'], '?')})"
            elif bx["conf"] < self.LOW_CONF:
                # 모델이 확신하지 못한 것. 이름은 그대로 저장되지만 틀렸을 수 있으니
                # 눈에 띄게 해 둔다. 아니면 「지정」 으로 고치거나 「무시」 하면 된다.
                base = self.BOX_LOW
                tag = f"{names.get(bx['class_id'], '?')} {bx['conf']:.2f} ?"
            else:
                base = self.BOX_MODEL
                tag = f"{names.get(bx['class_id'], '?')} {bx['conf']:.2f}"

            color = self.BOX_FOCUS if focused else base
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 4 if focused else 2)
            cv2.putText(view, f"[{i + 1}] {tag}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # 안내는 아래쪽에 둔다. 위에 두면 박스 라벨과 겹쳐 둘 다 안 읽힌다.
        msg = color = None
        if not self.collect_boxes:
            msg, color = "NO OBJECT - put bread in view", (40, 40, 235)
        elif self.collect_lock:
            pass                       # 고정 중에는 주황이 없다. 안내도 필요 없다.
        elif any(b["class_id"] is None for b in self.collect_boxes):
            msg, color = "name or ignore the orange box before capturing", self.BOX_TODO
        if msg:
            y = view.shape[0] - 16
            cv2.putText(view, msg, (18, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, msg, (18, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2, cv2.LINE_AA)
        return view

    def update_live_detections(self, frame, now):
        """대기 중에도 화면의 빵을 찾아 둔다. 주문 반영은 하지 않는다.

        여기서는 프레임을 추론 스레드에 던지기만 하고 즉시 돌아온다.
        결과는 on_live_result 로 나중에 온다. 화면은 그동안 계속 부드럽게 돈다.

        간격은 직전 추론에 걸린 시간의 2배로 둔다. 느린 환경에서는 알아서 뜸해지고,
        빠른 환경이라도 GPU 를 절반 이상 쓰지 않는다.
        """
        if not self.detector.ready() or now < self.live_next_ts:
            return
        if self.live_busy:
            return                       # 아직 앞엣것을 처리 중이다. 밀어 넣지 않는다.

        self.live_busy = True
        self.live_next_ts = now + self.live_gap
        self.detector.submit_live(frame.copy(), self.last_frame_id)

    def on_live_result(self, dets, frame_id, took):
        """추론 스레드가 결과를 보내왔다. (GUI 스레드에서 실행된다)"""
        self.live_busy = False
        self.live_dets = dets
        self.live_gap = max(self.LIVE_MIN_GAP, took * 2)
        self._dbg_infer += 1
        self._dbg_infer_ms += took * 1000.0

    def on_live_failed(self, message):
        """원격 서버가 끊기는 등. 화면만 비우고 잠시 뜸하게 재시도한다."""
        self.live_busy = False
        self.live_dets = []
        self.live_gap = min(self.live_gap * 2, 5.0)
        self.live_next_ts = time.perf_counter() + self.live_gap

    def on_qr_result(self, data, points):
        """QR 스레드가 결과를 보내왔다. (GUI 스레드에서 실행된다)

        한 번 읽힌 QR 은 QR_HOLD_FRAMES 동안 유지한다. 손이 흔들려 한두 프레임
        놓쳐도 초록 테두리가 깜빡이지 않게 하기 위해서다.
        """
        if data:
            self.last_qr_data = data
            self.last_qr_points = points
            self.qr_seen_frames = self.QR_HOLD_FRAMES
            self._set_qr_state("member" if data.startswith("BV1:") else "other")
            return

        if self.qr_seen_frames > 0:
            self.qr_seen_frames -= 1
        else:
            self.last_qr_data = ""
            self.last_qr_points = None
            self._set_qr_state("none")

    def update_frame(self):
        # ── 이벤트 루프가 밀렸는지 먼저 잰다 ──────────────────────────────
        # 반드시 함수 맨 앞에서 재야 한다. 아래 early return 뒤에서 재면
        # "타이머가 밀린 정도"가 아니라 "새 프레임이 오는 간격"을 재게 되어,
        # 15fps 짜리 카메라를 쓰면 멀쩡한데도 밀린 걸로 오판한다.
        now = time.perf_counter()
        lag = 0.0
        if self.last_tick_ts:
            lag = now - self.last_tick_ts
            if lag > self.CAMERA_INTERVAL * 2.5:
                # 타이머가 제때 못 돌았다 = 그리기가 루프를 잡아먹는 중
                self.display_gap = min(self.display_gap * 1.3, self.DISPLAY_MAX_GAP)
            else:
                self.display_gap = max(self.display_gap * 0.85, self.DISPLAY_MIN_GAP)
        self.last_tick_ts = now

        if self.debug_perf:
            self._dbg_ticks += 1
            self._dbg_lag = max(self._dbg_lag, lag * 1000)
            if now - self._dbg_t0 >= 2.0:
                d = now - self._dbg_t0
                avg = self._dbg_infer_ms / self._dbg_infer if self._dbg_infer else 0.0
                print(f"[성능] 카메라 {self._dbg_frames/d:5.1f} fps | 화면 {self._dbg_drawn/d:5.1f} fps "
                      f"| 추론 {self._dbg_infer/d:4.1f} fps ({avg:4.0f} ms) "
                      f"| 타이머 최대지연 {self._dbg_lag:5.0f} ms | 갱신간격 {self.display_gap*1000:4.0f} ms")
                self._dbg_t0 = now
                self._dbg_ticks = self._dbg_frames = self._dbg_drawn = 0
                self._dbg_lag = 0.0
                self._dbg_infer = 0
                self._dbg_infer_ms = 0.0

        # 캡처 스레드가 갱신해 둔 최신 프레임을 논블로킹으로 가져온다.
        if hasattr(self.camera, "read_latest"):
            ok, frame, frame_id = self.camera.read_latest()
        else:
            ok, frame = self.camera.read()
            frame_id = self.last_frame_id + 1

        if not ok:
            return

        if frame_id == self.last_frame_id:
            return                      # 아직 새 프레임이 없다. 그릴 필요도 없다.
        self.last_frame_id = frame_id
        self._dbg_frames += 1

        # Keep the untouched frame for AI/product recognition and data collection.
        self.frame = frame.copy()

        current = self.tabs.currentIndex()

        # QR 디코딩은 한 장에 27ms 쯤 든다 (실측). GUI 스레드에서 돌리면 그만큼 멈추므로
        # 전담 스레드에 던지기만 하고 결과는 on_qr_result 로 받는다.
        # QR 표시는 POS 탭에만 있으므로 그 탭에서 QR_EVERY 프레임마다 한 번만 보낸다.
        self.qr_frame_tick += 1
        if (current == self.TAB_POS and self.qr_frame_tick >= self.QR_EVERY
                and not self.qr_worker.busy):
            self.qr_frame_tick = 0
            self.qr_worker.submit(frame.copy())

        # 그리기에는 늘 가장 최근 결과를 쓴다.
        qr_data, qr_points = self.last_qr_data, self.last_qr_points

        if now - self.last_display_ts < self.display_gap:
            return
        self.last_display_ts = now
        self._dbg_drawn += 1

        # 보이지 않는 탭까지 그릴 이유가 없다.
        if current == self.TAB_POS:
            if (hasattr(self, "live_detect_check")
                    and self.live_detect_check.isChecked()):
                self.update_live_detections(frame, now)
            else:
                self.live_dets = []
            overlay = self._draw_qr_overlay(frame, qr_data, qr_points)
            shown = self.pos_dets(self.live_dets)
            if shown:
                # QR 오버레이가 원본을 그대로 넘겼을 수 있으니 여기서 한 번 복사
                if overlay is frame:
                    overlay = frame.copy()
                draw_detections(overlay, shown,
                                labels=self.class_name_map(), inplace=True)
            self.show_frame(self.pos_camera, overlay, guide=False)
        elif current == self.TAB_COLLECT:
            # AI 자동 라벨을 쓸 때는 가운데 가이드 박스 대신
            # "지금 찍으면 저장될 박스" 를 초록으로 보여준다.
            auto = (hasattr(self, "auto_label_check")
                    and self.auto_label_check.isChecked())
            if auto:
                # 아는 빵은 모델이 자동으로 라벨을 달아 주므로 계속 추론을 돌린다.
                # (전에는 배경 분리가 실패할 때만 불렀다. 그때는 한 장에 물체가
                #  하나뿐이라 모델이 없어도 됐지만, 지금은 화면에 있는 것을 전부
                #  라벨링해야 해서 아는 빵의 클래스를 매 프레임 알아야 한다)
                self.update_live_detections(frame, now)
                view = self.collect_preview(frame)
            else:
                view = frame
            self.show_frame(self.collect_camera, view, guide=not auto)


    @staticmethod
    def _decide_preview_max():
        """미리보기 그림의 최대 변 길이(px). 0 이면 제한 없이 라벨을 꽉 채운다.

        기본은 제한 없음이다 (bakery_gui 와 같은 큰 화면).

        다만 SSH X11 로 보고 있다면 그림 크기가 곧 전송량이다.
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
                print(f"[화면] BREADVISION_PREVIEW_MAX 가 숫자가 아닙니다: {env}")

        disp = os.environ.get("DISPLAY", "")
        host, _, _num = disp.rpartition(":")
        if disp and host not in ("", "unix"):
            # DISPLAY 에 호스트가 붙어 있으면 그림이 TCP 로 나간다 (SSH X11 등).
            print("[화면] 원격 X 화면(SSH X11)입니다. 미리보기가 클수록 느려집니다.\n"
                  "        느리면: BREADVISION_PREVIEW_MAX=640 (약 12fps) 또는 480 (30fps)")
        return 0

    def _preview_area(self, label):
        """라벨이 레이아웃에서 원래 차지하는 크기 (가로, 세로).

        원격 화면에서는 아래에서 라벨을 그림 크기로 고정해 버리므로,
        그 뒤에 label.width() 를 읽으면 줄어든 값이 나와 화면이 점점 작아진다.
        그래서 처음(레이아웃이 준) 크기를 기억해 두고 그걸 기준으로 삼는다.
        창 크기가 바뀌면 resizeEvent 에서 지운다.
        """
        key = id(label)
        area = self._preview_area_cache.get(key)
        if area is None:
            area = (label.width(), label.height())
            if area[0] > 0 and area[1] > 0:
                self._preview_area_cache[key] = area
        return area

    def resizeEvent(self, event):
        # 창이 바뀌면 라벨이 차지할 크기도 다시 재야 한다.
        self._preview_area_cache.clear()
        for name in ("pos_camera", "collect_camera"):
            lab = getattr(self, name, None)
            if lab is not None:
                self._reset_preview_size(lab)
        super().resizeEvent(event)

    def show_frame(self, label, frame, guide=False):
        # Qt 의 SmoothTransformation 축소보다 cv2.resize 가 두 배 이상 빠르다.
        h0, w0 = frame.shape[:2]
        aw, ah = self._preview_area(label)

        if self.preview_max:
            # 원격 화면(SSH X11 등)일 때.
            #
            # 라벨을 그림 크기에 딱 맞춰 고정하는 게 핵심이다.
            # 라벨이 그림보다 크면 Qt 는 매 프레임 라벨 둘레(와 그 아래 배경)까지
            # 다시 그리고, 원격 화면에서는 그 픽셀이 전부 회선을 탄다.
            #   실측(892x503 라벨 안에 640x480 그림): 213 MB/s — 창 전체가 매번 전송됐다.
            #   160x120 짜리 작은 그림을 넣어도 216 MB/s 로 똑같았다.
            #   라벨을 그림 크기에 고정하면  640px → 32 MB/s, 320px → 9.7 MB/s.
            #
            # 젯슨 랜선이 100Mbps(12.5 MB/s)이므로 이 차이가 곧 초당 몇 장이냐를 정한다.
            aw = min(aw, self.preview_max)
            ah = min(ah, self.preview_max)

        if aw > 0 and ah > 0:
            # 라벨을 꽉 채운다. 작으면 키우고 크면 줄인다 (가로세로 비율은 유지).
            r = min(aw / float(w0), ah / float(h0))
            if abs(r - 1.0) > 0.01:
                nw, nh = max(1, int(w0 * r)), max(1, int(h0 * r))
                # 키울 때는 LINEAR, 줄일 때는 AREA 가 각각 빠르고 깨끗하다.
                interp = cv2.INTER_LINEAR if r > 1.0 else cv2.INTER_AREA
                frame = cv2.resize(frame, (nw, nh), interpolation=interp)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())

        if guide:
            painter = QPainter(pix)
            painter.setPen(QPen(QColor(230, 45, 45), 4))
            pw, ph = pix.width(), pix.height()
            bw, bh = int(pw * 0.45), int(ph * 0.45)
            x1, y1 = (pw - bw) // 2, (ph - bh) // 2
            painter.drawRect(x1, y1, bw, bh)

            painter.setFont(QFont("Arial", 13, QFont.Bold))
            painter.drawText(
                x1,
                max(25, y1 - 10),
                "PLACE ONE BREAD INSIDE THIS BOX"
            )
            painter.end()

        # 원격 화면일 때만 라벨을 그림 크기에 딱 맞춘다.
        # 남는 자리가 1px 이라도 있으면 Qt 가 매 프레임 그 둘레와 아래 배경까지
        # 다시 칠하고, 원격 화면에서는 그 픽셀이 전부 회선을 탄다.
        #
        # 로컬 화면(PC·젯슨 모니터)에서는 고정하면 안 된다. 라벨이 그림만큼
        # 쪼그라들어 레이아웃이 남는 자리를 다른 위젯에 줘 버리고,
        # 그 뒤 _preview_area 가 줄어든 값을 기준으로 삼아 화면이 다시는 커지지 않는다.
        # (bakery_gui 처럼 라벨을 꽉 채운 채로 둔다)
        if self.preview_max:
            if label.width() != pix.width() or label.height() != pix.height():
                label.setFixedSize(pix.width(), pix.height())
        label.setPixmap(pix)


    def _reset_preview_size(self, label):
        """카메라를 끄면 라벨 고정을 풀어 안내 문구가 제자리에 보이게 한다."""
        if not self.preview_max:
            return
        label.setMinimumSize(0, 0)
        label.setMaximumSize(16777215, 16777215)

    def export_collected_zip(self):
        """수집한 사진/라벨을 학습에 넣을 수 있는 형태로 포장한다.

        import_master_dataset.py 는 "jpg 옆에 같은 이름의 txt" 인 평면 구조만 읽는다.
        collected/ 는 images/ 와 labels/ 로 나뉘어 있으므로 여기서 한 폴더로 합쳐 압축한다.
        classes.txt 는 products.json 순서 그대로 쓴다 (신규 상품이 그대로 반영된다).
        """
        imgs = sorted(COLLECT_IMG.glob("*.jpg"))
        if not imgs:
            QMessageBox.warning(self, "수집 데이터 없음",
                                "먼저 데이터 수집을 진행해주세요.")
            return

        pairs = [(f, COLLECT_LBL / f"{f.stem}.txt") for f in imgs]
        pairs = [(i, l) for i, l in pairs if l.exists()]
        if not pairs:
            QMessageBox.warning(self, "라벨 없음",
                                "이미지에 대응하는 라벨(.txt)이 없습니다.")
            return

        import_dir = BASE_DIR / "dataset" / "import"
        import_dir.mkdir(parents=True, exist_ok=True)
        out = import_dir / f"collected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

        names = [p["class_name"] for _, p in
                 sorted(self.products.items(), key=lambda x: int(x[0]))]

        import zipfile
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for img, lab in pairs:
                z.write(img, img.name)          # 같은 폴더에 나란히
                z.write(lab, lab.name)
            z.writestr("classes.txt", "\n".join(names) + "\n")

        per_class = {}
        for _, lab in pairs:
            for line in lab.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    cid = int(line.split()[0])
                    per_class[cid] = per_class.get(cid, 0) + 1
        detail = "\n".join(
            f"  {cid} {self.products.get(str(cid), {}).get('display_name', '?')}: {n}장"
            for cid, n in sorted(per_class.items())
        )

        # 젯슨에 바로 올려 준다. 업체가 scp 를 칠 일이 없어야 한다.
        #   replace="collected_" — 옛 collected ZIP 은 지운다.
        #   이 ZIP 은 수집 폴더 '전체'를 담으므로 옛것이 남으면 같은 사진이 두 번 들어간다.
        sent = ""
        if self.can_run_on_jetson():
            self.statusBar().showMessage(f"젯슨으로 전송 중… ({out.stat().st_size//1024//1024} MB)")
            QApplication.processEvents()
            try:
                up = self.ops.upload_zip(str(out), replace_prefix="collected_")
                sent = f"\n\n젯슨으로 전송 완료: {up['saved']}"
                if up.get("removed"):
                    sent += f"\n(옛 파일 정리: {', '.join(up['removed'])})"
            except Exception as e:
                sent = (f"\n\n[전송 실패] {e}\n"
                        "젯슨 서버가 켜져 있는지 확인하세요. 파일은 PC 에 남아 있습니다.")
            finally:
                self.statusBar().clearMessage()

        QMessageBox.information(
            self, "학습용 ZIP 생성 완료",
            f"{out.name}\n총 {len(pairs)}장\n\n{detail}{sent}\n\n"
            "다음 단계: 「모델 배포 및 모델 학습」 탭 →\n"
            "  ① Train / Val 자동 분할  ② 전이학습 시작"
        )

    def clear_collected_data(self):
        """수집한 사진·라벨을 모두 지운다.

        이미지와 라벨을 따로 지우면 짝이 안 맞는 파일이 남아 나중에 조용히 문제가 된다.
        여기서 항상 같이 지운다.
        내보낸 ZIP 은 별개 파일이라 남아 있으므로, 그것도 지울지 따로 물어본다.
        (남겨두면 지운 사진이 다음 학습에 그대로 들어간다)
        """
        imgs = sorted(COLLECT_IMG.glob("*.jpg"))
        labs = sorted(COLLECT_LBL.glob("*.txt"))
        if not imgs and not labs:
            QMessageBox.information(self, "비어 있음", "지울 수집 데이터가 없습니다.")
            return

        # 무엇을 지우는지 품목별로 보여준다
        per_class = {}
        for lab in labs:
            try:
                for line in lab.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        cid = int(line.split()[0])
                        per_class[cid] = per_class.get(cid, 0) + 1
            except OSError:
                continue
        detail = "\n".join(
            f"  {cid} {self.products.get(str(cid), {}).get('display_name', '?')}: {n}장"
            for cid, n in sorted(per_class.items())
        ) or "  (라벨 없음)"

        answer = QMessageBox.question(
            self, "수집 데이터 비우기",
            f"아래 데이터를 모두 지웁니다. 되돌릴 수 없습니다.\n\n"
            f"사진 {len(imgs)}장 · 라벨 {len(labs)}개\n\n{detail}\n\n"
            f"위치: {COLLECT_IMG.parent}\n\n지울까요?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return

        removed = 0
        failed = []
        for f in imgs + labs:
            try:
                f.unlink()
                removed += 1
            except OSError as e:
                failed.append(f"{f.name}: {e}")

        # 이미 내보낸 ZIP 도 지울지 물어본다. 남겨두면 지운 사진이 학습에 계속 들어간다.
        zips = sorted((BASE_DIR / "dataset" / "import").glob("collected_*.zip"))
        zip_msg = ""
        if zips:
            names = "\n".join(f"  {z.name}  ({z.stat().st_size // 1024 // 1024} MB)" for z in zips)
            ans2 = QMessageBox.question(
                self, "내보낸 ZIP 도 지울까요?",
                f"사진은 지웠지만 이미 내보낸 ZIP 이 남아 있습니다.\n"
                f"이걸 두면 지운 사진이 다음 학습에 그대로 들어갑니다.\n\n{names}\n\n"
                "같이 지울까요?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ans2 == QMessageBox.Yes:
                gone = []
                for z in zips:
                    try:
                        z.unlink()
                        gone.append(z.name)
                    except OSError as e:
                        failed.append(f"{z.name}: {e}")
                if gone:
                    zip_msg = "\n지운 ZIP: " + ", ".join(gone)
            else:
                zip_msg = "\nZIP 은 남겨두었습니다."

        self.collect_count = 0
        self.collect_skipped = 0
        if hasattr(self, "collect_status"):
            self.collect_status.setText("수집 데이터를 비웠습니다.")
        self.statusBar().showMessage(f"수집 데이터 {removed}개 삭제", 5000)

        msg = f"{removed}개 파일을 지웠습니다.{zip_msg}"
        if failed:
            msg += "\n\n지우지 못한 것:\n" + "\n".join(failed[:5])
        QMessageBox.information(self, "삭제 완료", msg)

    def side_sweep_note(self, span2):
        """좌우 왕복(가운데→왼쪽→오른쪽→가운데)이 촬영 시간 안에 끝나는지 알려 준다."""
        if span2 <= 0 or not self.smooth_check.isChecked():
            return ""
        need = 4.0 * span2 / 360.0 * 2038 * 0.002          # 좌우 모터가 낼 수 있는 최고 속도 기준
        have = self.target_count * float(self.interval_spin.value())
        if need <= have:
            return ""
        return (f"\n⚠ 좌우 ±{span2:.0f}도 왕복에 최소 {need:.0f}초가 필요한데 촬영은 {have:.0f}초입니다.\n"
                f"   장수를 {int(need / max(0.1, float(self.interval_spin.value()))) + 1}장 이상으로 늘리거나 각도를 줄이세요.")

    def start_collection(self):
        if self.frame is None:
            QMessageBox.warning(self, "카메라 필요", "먼저 「수집용 카메라 켜기」를 눌러주세요.")
            return

        if self.collect_product_combo.count() == 0:
            QMessageBox.warning(self, "품목 없음", "② 품목 · 가격 관리에서 상품을 먼저 등록해주세요.")
            return

        self.collect_count = 0
        self.collect_skipped = 0
        self.reset_steady_box()
        self.target_count = self.target_spin.value()
        self.collecting = True
        self.collect_progress.setValue(0)
        if self.turn_check.isChecked():
            # 턴테이블이 박자를 잡는다. 한 칸 돌고 → 한 장 찍고 → 다음 칸.
            # 타이머로 따로 찍으면 도는 중에 찍혀 흔들린 사진이 섞인다.
            span = float(self.span_spin.value())
            span2 = float(self.span2_spin.value())
            self.turn_worker = TurntableWorker(
                self.target_count, span=span, span2=span2,
                interval=float(self.interval_spin.value()),
                smooth=self.smooth_check.isChecked())
            self.turn_worker.arrived.connect(self.on_turntable_arrived)
            self.turn_worker.failed.connect(self.on_turntable_failed)
            self.turn_worker.done.connect(self.on_turntable_done)
            step = self.turn_worker.step_deg()
            self.collect_status.setText(
                f"현재 상태: 자동 수집 중 (턴테이블)\n"
                f"±{span:.0f}도 왕복, 한 칸 {step:.1f}도 x {self.target_count}장"
                + (f"\n좌우 모터: 가운데 → 왼쪽 → 오른쪽 → 가운데 (±{span2:.0f}도)" if span2 > 0 else "")
                + ("\n끊김 없이 계속 움직이는 중" if self.smooth_check.isChecked() else "")
                + self.side_sweep_note(span2)
            )
            self.turn_worker.start()
        else:
            self.collect_status.setText(
                "현재 상태: 자동 수집 중\n빵을 빨간 박스 안에서 조금씩 회전/이동해주세요."
            )
            self.collect_timer.start(int(self.interval_spin.value() * 1000))

    # ── 턴테이블 ──────────────────────────────────────────────────
    def on_turntable_arrived(self, angle):
        """한 칸 돌고 멈췄다. 지금 한 장 찍는다. (GUI 스레드에서 실행된다)"""
        if not self.collecting:
            return
        self.capture_dataset()
        if self.turn_worker:
            self.turn_worker.shot_done()          # 다음 칸으로 가도 된다

    def on_turntable_failed(self, message):
        self.collecting = False
        self.collect_status.setText(f"현재 상태: 턴테이블 오류\n{message}")
        QMessageBox.warning(
            self, "턴테이블",
            f"{message}\n\n턴테이블 없이 찍으려면 「촬영 중 턴테이블 자동 회전」 을 끄세요.")

    def on_turntable_done(self):
        self.turn_worker = None
        if self.collecting:
            self.collecting = False
            self.collect_status.setText(
                f"현재 상태: 회전 왕복 완료\n이번 세션에서 {self.collect_count}장 저장됨")
            self.refresh_all()

    def stop_collection(self):
        self.collecting = False
        self.collect_timer.stop()
        if self.turn_worker:
            self.turn_worker.stop()               # 제자리로 돌아온 뒤 끝난다
        self.collect_status.setText(
            f"현재 상태: 수집 중지\n이번 세션에서 {self.collect_count}장 저장됨"
        )

    # 자동 라벨 판정 기준
    # 신뢰도 구간이 끊기면 안 된다. 중간이 비면 그 빵이 화면에서 사라져
    # 빨강으로 고를 수가 없다. 15% 위로는 노랑이든 초록이든 항상 보인다.
    AUTOLABEL_CONF = 0.15       # 이보다 낮은 검출은 화면에도 안 그린다
    LOW_CONF = 0.55             # 15~55% 노랑, 55% 이상 초록 (사이에 빈 구간 없음)
    POS_CONF = 0.55             # 주문에 담을 때 요구하는 신뢰도

    def pos_dets(self, detections):
        """POS 화면·주문에 쓸 검출만 남긴다.

        추론 서버는 낮은 것까지 다 보낸다 (수집 탭에서 "신뢰도 낮음" 을
        노랑으로 보여줘야 하므로). 서버가 미리 잘라 버리면 그럴 수가 없다.
        대신 돈이 오가는 POS 쪽은 여기서 확실한 것만 남긴다.
        """
        return [d for d in (detections or [])
                if float(d.get("confidence", 0)) >= self.POS_CONF]
    AUTOLABEL_MIN_AREA = 0.01   # 화면의 1% 미만이면 오검출로 본다
    AUTOLABEL_MAX_AREA = 0.85   # 85% 초과면 배경을 잡은 것으로 본다
    SINGLE_MODEL_CONF = 0.35    # 한 개 방식: 배경 분리가 안 돼 모델 박스를 쓸 때 최소 신뢰도 (예전 기준)

    # 박스 흔들림 다듬기 (한 개 방식)
    STEADY_FOLLOW = 0.35        # 새 박스를 얼마나 따라갈지 (작을수록 덜 흔들린다)
    STEADY_JUMP_IOU = 0.30      # 이만큼도 안 겹치면 "다른 물체" 로 본다
    STEADY_JUMP_FRAMES = 3      # 다른 물체라도 이만큼 계속 나오면 그때 옮긴다
    STEADY_KEEP_FRAMES = 5      # 잠깐 놓쳐도 이만큼은 이전 박스를 유지한다

    def reset_steady_box(self):
        self._steady_box = None
        self._steady_miss = 0
        self._steady_pending = 0

    def steady_box(self, box):
        """프레임마다 박스가 튀지 않게 다듬는다.

        · 가까운 자리면 조금씩 따라간다 (떨림 제거)
        · 엉뚱한 곳이면 몇 프레임 계속 나올 때만 옮긴다 (튐 방지)
        · 잠깐 놓치면 직전 박스를 유지한다 (깜빡임 방지)
        """
        if not hasattr(self, "_steady_box"):
            self.reset_steady_box()
        prev = self._steady_box
        if box is None:
            self._steady_miss += 1
            if prev is not None and self._steady_miss <= self.STEADY_KEEP_FRAMES:
                return tuple(int(round(v)) for v in prev)
            self._steady_box = None
            return None
        self._steady_miss = 0
        box = tuple(float(v) for v in box)
        if prev is None:
            self._steady_box, self._steady_pending = box, 0
        elif self._iou(prev, box) >= self.STEADY_JUMP_IOU:
            a = self.STEADY_FOLLOW
            self._steady_box = tuple(prev[i] * (1 - a) + box[i] * a for i in range(4))
            self._steady_pending = 0
        else:
            self._steady_pending += 1
            if self._steady_pending >= self.STEADY_JUMP_FRAMES:
                self._steady_box, self._steady_pending = box, 0
        return tuple(int(round(v)) for v in self._steady_box)

    def pick_label_box(self, detections, frame_shape, min_conf=None):
        """검출 결과 중 라벨로 쓸 만한 박스 하나를 고른다.

        클래스는 모델 판정이 아니라 사용자가 고른 품목을 쓸 것이므로 여기서는 위치만 본다.
        (트레이에 무엇을 올렸는지는 사람이 알고, 모델은 위치만 잡아주면 된다)
        쓸 수 없으면 (None, 이유) 를 돌려준다.
        수집 저장과 수집 탭 미리보기가 완전히 같은 기준을 쓰도록 여기 한 곳에만 둔다.
        """
        if not detections:
            return None, "no bread found"

        h, w = frame_shape[:2]
        cx0, cy0 = w / 2.0, h / 2.0
        ok = [d for d in detections
              if float(d.get("confidence", 0)) >= (self.AUTOLABEL_CONF if min_conf is None else min_conf)]
        if ok:      # 가운데에 놓고 찍으므로, 화면 중심에 가장 가까운 것을 고른다
            best = min(ok, key=lambda d: ((d["xyxy"][0] + d["xyxy"][2]) / 2 - cx0) ** 2
                                          + ((d["xyxy"][1] + d["xyxy"][3]) / 2 - cy0) ** 2)
        else:
            best = max(detections, key=lambda d: d["confidence"])
        if best["confidence"] < (self.AUTOLABEL_CONF if min_conf is None else min_conf):
            return None, f"low conf {best['confidence']:.2f}"

        x1, y1, x2, y2 = best["xyxy"]
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            return None, "bad box"

        area = (bw * bh) / float(w * h)
        if area < self.AUTOLABEL_MIN_AREA:
            return None, f"too small {area * 100:.1f}%"
        if area > self.AUTOLABEL_MAX_AREA:
            return None, f"too big {area * 100:.0f}%"

        return best, f"conf {best['confidence']:.2f}"

    def label_box_pixels(self, frame, detections=None):
        """라벨로 쓸 박스를 픽셀 좌표 (x1,y1,x2,y2) 로 돌려준다.

        신메뉴는 모델이 모르는 빵이므로 **배경 분리**를 먼저 쓴다.
        (흰 유산지/밝은 트레이 위 물체를 색·밝기로 잡는다. 클래스는 보지 않는다)
        배경이 지저분해 분리가 안 되면, 아는 빵일 수도 있으니 모델 검출로 넘어간다.
        """
        roi = self.roi_spin.value() / 100.0 if hasattr(self, "roi_spin") else 0.0
        box, why, cand = object_box(frame,
                                    min_area=self.AUTOLABEL_MIN_AREA,
                                    max_area=self.AUTOLABEL_MAX_AREA,
                                    roi=roi)
        self.last_segment_candidate = cand
        if box is not None:
            self.collect_need_model = False
            return self.steady_box(box), "auto"

        self.collect_need_model = True

        if detections is None and self.detector.ready():
            try:
                _, detections, _ = self.detector.infer(frame)
            except Exception:
                detections = []
        best, why2 = self.pick_label_box(detections or [], frame.shape,
                                         min_conf=self.SINGLE_MODEL_CONF)
        if best is not None:
            return self.steady_box(tuple(best["xyxy"])), "model"
        return self.steady_box(None), why

    def auto_label_box(self, frame):
        """촬영 순간의 라벨 좌표 (xc, yc, w, h) 정규화 값."""
        box, why = self.label_box_pixels(frame)
        if box is None:
            return None, why

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        return (((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h,
                (x2 - x1) / w, (y2 - y1) / h), why

    def capture_dataset(self):
        if not self.collecting or self.frame is None:
            return

        cid = int(self.collect_product_combo.currentData())
        p = self.products[str(cid)]
        class_name = p["class_name"]

        COLLECT_IMG.mkdir(parents=True, exist_ok=True)
        COLLECT_LBL.mkdir(parents=True, exist_ok=True)

        frame = self.frame.copy()
        h, w = frame.shape[:2]

        auto = hasattr(self, "auto_label_check") and self.auto_label_check.isChecked()
        note = ""
        lines = []
        if auto and self.collect_mode() == "single":
            # 한 개 방식 (예전 방식): 빵 하나만 찾아 고른 품목으로 저장한다.
            box, why = self.auto_label_box(frame)
            if box is None:
                # 라벨을 믿을 수 없으면 저장하지 않는다. 잘못된 라벨은 모델을 망친다.
                self.collect_skipped += 1
                self.collect_status.setText(
                    f"현재 상태: 자동 수집 중 (한 개 방식)\n"
                    f"{p['display_name']} — {self.collect_count}/{self.target_count}장\n"
                    f"방금 건너뜀: {why}  (누적 {self.collect_skipped}장)"
                )
                return
            xc, yc, nw, nh = box
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            note = f"  [한 개 · {'배경 분리' if why == 'auto' else '모델'}]"
        elif auto:
            # 화면에 있는 것을 전부 라벨로 쓴다.
            # 하나라도 품목이 안 정해졌으면 저장하지 않는다 — 라벨 없는 빵이
            # 사진에 남으면 YOLO 가 그것을 배경으로 학습해 버린다.
            boxes = self.collect_label_boxes(frame)
            todo = [b for b in boxes if b["class_id"] is None]
            if not boxes or todo:
                # 라벨이 없는 물체가 사진에 남으면 YOLO 가 그것을 배경으로
                # 학습해 버린다. 이름을 다 붙이기 전까지는 저장하지 않는다.
                why = "잡힌 물체 없음" if not boxes else f"이름 없는 것 {len(todo)}개"
                self.collect_skipped += 1
                self.collect_status.setText(
                    f"현재 상태: 자동 수집 중 (AI 자동 라벨)\n"
                    f"{self.collect_count}/{self.target_count}장\n"
                    f"방금 건너뜀: {why}  (누적 {self.collect_skipped}장)\n"
                    "「다음 ▶」 로 빨간 테두리를 옮겨 지정하거나 무시하세요."
                )
                return
            for b in boxes:
                x1, y1, x2, y2 = b["xyxy"]
                lines.append(
                    f"{b['class_id']} {((x1 + x2) / 2) / w:.6f} {((y1 + y2) / 2) / h:.6f} "
                    f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                )
            fixed_n = sum(1 for b in boxes if b["fixed"])
            note = f"  [라벨 {len(lines)}개 · 내가 바꾼 것 {fixed_n}]"
        else:
            # 고정 박스: 빵을 화면 중앙 가이드에 맞춰 놓아야 한다
            bw, bh = int(w * 0.45), int(h * 0.45)
            x1, y1 = (w - bw) // 2, (h - bh) // 2
            xc = ((x1 + x1 + bw) / 2) / w
            yc = ((y1 + y1 + bh) / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        existing = self.count_collected_for_class(class_name)
        idx = existing + 1
        stem = f"{class_name}_{idx:05d}"

        cv2.imwrite(str(COLLECT_IMG / f"{stem}.jpg"), frame)
        (COLLECT_LBL / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

        self.collect_count += 1
        pct = int(self.collect_count / self.target_count * 100)
        self.collect_progress.setValue(min(100, pct))

        self.collect_status.setText(
            f"현재 상태: 자동 수집 중{note}\n"
            f"{p['display_name']} — {self.collect_count}/{self.target_count}장 "
            f"(누적 {existing + 1}장)"
        )

        if self.collect_count >= self.target_count:
            self.collect_timer.stop()
            if self.turn_worker:
                self.turn_worker.stop()      # 목표를 채웠으면 모터도 멈춘다
            self.collecting = False
            self.refresh_all()

            QMessageBox.information(
                self,
                "데이터 수집 완료",
                f"{p['display_name']} 데이터 {self.collect_count}장 수집 완료\n\n"
                "JPG 이미지와 YOLO Bounding Box TXT 라벨이 함께 저장되었습니다."
            )

    # -------------------- TRAIN --------------------
    def build_train_tab(self):
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        title = QLabel("모델 배포 및 모델 학습")
        title.setObjectName("PageTitle")
        desc = QLabel(
            "정식 업데이트는 Colab ONNX 배포를 그대로 사용하고, 긴급 신상품은 전이학습할 수 있습니다."
        )
        desc.setObjectName("PageDesc")
        root.addWidget(title)
        root.addWidget(desc)

        # 여기 버튼들이 어디서 실행되는지 한 줄로 알려준다.
        self.ops_status = QLabel()
        self.ops_status.setWordWrap(True)
        root.addWidget(self.ops_status)
        self.refresh_ops_status()

        deploy = QGroupBox("① 외부 모델 배포 — Colab → Jetson")
        dlay = QGridLayout(deploy)
        self.onnx_path_edit = QLineEdit()
        self.onnx_path_edit.setPlaceholderText("Colab에서 만든 best.onnx 선택")
        btn_onnx = QPushButton("ONNX 선택")
        btn_onnx.clicked.connect(self.select_onnx_model)
        btn_convert = QPushButton("ONNX → TensorRT ENGINE")
        btn_convert.setObjectName("Primary")
        btn_convert.clicked.connect(self.convert_selected_onnx)
        self.engine_path_label = QLabel("활성 ENGINE: 미설정")
        self.engine_path_label.setWordWrap(True)
        btn_engine = QPushButton("기존 .engine 선택")
        btn_engine.clicked.connect(self.select_engine_model)
        btn_reload = QPushButton("ENGINE 다시 로드")
        btn_reload.clicked.connect(self.reload_active_engine)
        dlay.addWidget(QLabel("ONNX"), 0, 0); dlay.addWidget(self.onnx_path_edit, 0, 1); dlay.addWidget(btn_onnx, 0, 2)
        dlay.addWidget(btn_convert, 1, 1)
        dlay.addWidget(QLabel("현재 POS 모델"), 2, 0); dlay.addWidget(self.engine_path_label, 2, 1)
        dlay.addWidget(btn_engine, 3, 1); dlay.addWidget(btn_reload, 3, 2)
        root.addWidget(deploy)

        emergency = QGroupBox("② Jetson 긴급 전이학습")
        elay = QGridLayout(emergency)
        self.training_model_label = QLabel("기본 .pt: 미설정")
        self.training_model_label.setWordWrap(True)
        btn_pt = QPushButton("기존 best.pt 선택")
        btn_pt.clicked.connect(self.select_training_model)
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 300); self.epochs_spin.setValue(30)
        self.imgsz_spin = QSpinBox(); self.imgsz_spin.setRange(320, 1280); self.imgsz_spin.setSingleStep(32); self.imgsz_spin.setValue(640)
        btn_prepare = QPushButton("Master 데이터셋 준비/갱신")
        btn_prepare.setToolTip("찍은 사진을 Train/Val 로 나눕니다.\n"
                               "아래 합성이 켜져 있으면 이어서 합성까지 합니다.")
        btn_prepare.clicked.connect(self.prepare_dataset_split)

        # 신상품은 턴테이블로 혼자 있는 모습만 찍힌다. 그것만 학습하면
        # 매대에서 겹쳐 놓인 빵을 못 잡으므로, 오려내 여러 개짜리로 합성한다.
        self.synth_check = QCheckBox("합성 데이터 만들기 (신상품 보강)")
        self.synth_check.setChecked(True)
        self.synth_check.setToolTip(
            "찍은 사진에서 빵만 오려내 여러 개 놓인 장면으로 합성합니다.\n"
            "라벨은 자동으로 붙습니다 — 사람이 그리지 않습니다.\n"
            "한 개씩만 찍어도 겹친 빵을 인식하게 하려는 것입니다.")
        self.synth_spin = QSpinBox()
        self.synth_spin.setRange(50, 3000)
        self.synth_spin.setValue(400)
        self.synth_spin.setSingleStep(50)
        self.synth_spin.setSuffix(" 장")
        self.synth_spin.setToolTip(
            "최소 장수입니다. 신상품이 합성 안에 250번은 나오도록\n"
            "젯슨이 필요하면 알아서 더 만듭니다.\n\n"
            "실측 — 신상품 등장 횟수와 인식률(놓치지 않는 비율):\n"
            "  115회 82%  ·  177회 92%  ·  231회 98%\n"
            "231회면 실제 사진으로 배운 것과 같습니다.")
        btn_train = QPushButton("전이학습 시작 (학습 → 변환 → 적용 자동)")
        btn_train.setObjectName("Success")
        btn_train.setToolTip("학습이 끝나면 ENGINE 변환과 POS 적용까지 이어서 자동으로 합니다.")
        btn_train.clicked.connect(self.start_training)
        btn_build = QPushButton("변환·적용만 다시 하기")
        btn_build.setToolTip("이미 학습된 best.pt 를 ENGINE 으로 만들어 적용합니다. "
                             "학습 중 창을 닫았거나 적용만 다시 하고 싶을 때 씁니다.")
        btn_build.clicked.connect(self.apply_last_best_model)
        elay.addWidget(self.synth_check,0,0,1,2); elay.addWidget(self.synth_spin,0,2)
        elay.addWidget(QLabel("전이학습 시작 모델"),1,0); elay.addWidget(self.training_model_label,1,1); elay.addWidget(btn_pt,1,2)
        elay.addWidget(QLabel("Epoch"),2,0); elay.addWidget(self.epochs_spin,2,1)
        elay.addWidget(QLabel("Image size"),3,0); elay.addWidget(self.imgsz_spin,3,1)
        elay.addWidget(btn_prepare,4,0); elay.addWidget(btn_train,4,1); elay.addWidget(btn_build,4,2)
        root.addWidget(emergency)

        self.phase_label = QLabel("")
        self.phase_label.setWordWrap(True)
        self.phase_label.setVisible(False)
        root.addWidget(self.phase_label)

        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumHeight(220)
        self.train_log.setPlainText(
            "[운영 원칙] Colab = 정식 학습 / Jetson = 긴급 전이학습 / TensorRT ENGINE = 실제 POS 추론\n"
            "dataset/import 에 원본 ZIP을 넣고 import_master_dataset.py를 실행하면 master 데이터셋을 만들 수 있습니다."
        )
        root.addWidget(self.train_log)

        note = QLabel(
            "학습은 젯슨의 NVIDIA PyTorch Docker 컨테이너에서 돌기 때문에 젯슨의 파이썬 환경을 건드리지 않습니다. "
            "TensorRT ENGINE 은 그 기기·그 버전 전용 바이너리라 PC 에서는 만들 수 없어, 변환도 젯슨이 맡습니다. "
            "PC 에서는 파일을 보내고 버튼만 누르면 됩니다."
        )
        note.setWordWrap(True)
        note.setStyleSheet("background:#0C1722; border:1px solid #24405B; padding:12px; border-radius:8px; color:#AFC4D9;")
        root.addWidget(note)
        root.addStretch(1)
        self.refresh_model_labels()
        return w

    def set_phase(self, text=None):
        """학습 → 변환 → 적용 중 지금 어디인지 보여준다. (한 시간짜리 작업이라 필요하다)"""
        if not hasattr(self, "phase_label"):
            return
        if not text:
            self.phase_label.setVisible(False)
            return
        self.phase_label.setText(text)
        self.phase_label.setStyleSheet(
            "background:#10233A; border:1px solid #2D5C8F; padding:10px; "
            "border-radius:7px; color:#9FC8F5; font-weight:700;")
        self.phase_label.setVisible(True)

    def refresh_ops_status(self):
        """이 탭의 버튼이 어디서 실행되는지 보여준다."""
        if not hasattr(self, "ops_status"):
            return
        if self.can_run_on_jetson():
            url = self.ops.url
            self.ops_status.setText(
                f"● 젯슨 연결됨 ({url}) — 아래 버튼을 누르면 젯슨에서 실행됩니다. "
                "젯슨에 따로 접속할 필요가 없습니다."
            )
            self.ops_status.setStyleSheet(
                "background:#0B2418; border:1px solid #1E5B3A; padding:9px; "
                "border-radius:7px; color:#7FE0A8;")
        elif self.is_jetson():
            self.ops_status.setText("● 젯슨에서 직접 실행 중 — 아래 기능을 모두 쓸 수 있습니다.")
            self.ops_status.setStyleSheet(
                "background:#0B2418; border:1px solid #1E5B3A; padding:9px; "
                "border-radius:7px; color:#7FE0A8;")
        else:
            self.ops_status.setText(
                "● 젯슨에 연결되지 않았습니다 — 학습·변환은 젯슨이 해야 하는 일이라 "
                "여기서는 실행할 수 없습니다.\n"
                "   젯슨에서 ./jetson/start.sh 를 실행하고, "
                "BREADVISION_INFER_URL 을 젯슨 주소로 지정한 뒤 다시 켜주세요."
            )
            self.ops_status.setStyleSheet(
                "background:#2A1410; border:1px solid #6B2E22; padding:9px; "
                "border-radius:7px; color:#F0A99B;")

    def select_onnx_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Colab ONNX 선택", str(BASE_DIR / "models"), "ONNX Model (*.onnx)")
        if path:
            self.onnx_path_edit.setText(path)

    def convert_selected_onnx(self):
        if self.can_run_on_jetson():
            self.convert_onnx_on_jetson()
            return
        if not self.require_jetson("ONNX → TensorRT ENGINE 변환"):
            return

        path = self.onnx_path_edit.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "ONNX 필요", "변환할 .onnx 파일을 먼저 선택해주세요.")
            return
        try:
            out = str(Path(path).with_suffix(".engine"))
            import subprocess
            cmd = ["/usr/src/tensorrt/bin/trtexec", f"--onnx={path}", f"--saveEngine={out}", "--fp16"]
            self.train_log.appendPlainText("[TRT] " + " ".join(cmd))
            subprocess.run(cmd, check=True)
            self.detector.load(out)
            self.settings["active_engine"] = out
            self.save_settings()
            self.refresh_model_labels()
            QMessageBox.information(self, "배포 완료", f"ENGINE 생성 및 적용 완료\n{out}")
        except Exception as e:
            QMessageBox.critical(self, "TensorRT 변환 실패", str(e))

    def convert_onnx_on_jetson(self):
        """Colab 에서 받은 .onnx 를 젯슨에 올려 ENGINE 으로 만들고 바로 적용한다.

        TensorRT ENGINE 은 그 기기·그 버전 전용 바이너리라 PC 에서 만들 수 없다.
        그래서 파일만 보내고 변환은 젯슨이 한다.
        """
        if self.busy_with_job():
            return
        path = self.onnx_path_edit.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "ONNX 필요", "변환할 .onnx 파일을 먼저 선택해주세요.")
            return

        size_mb = Path(path).stat().st_size / 1024 / 1024
        self.train_log.appendPlainText(f"[젯슨] ONNX 전송 중… {Path(path).name} ({size_mb:.0f} MB)")
        QApplication.processEvents()
        try:
            up = self.ops.upload_zip(path)          # .onnx 도 같은 창구로 올라간다
        except Exception as e:
            QMessageBox.critical(self, "전송 실패", str(e))
            return
        remote = up["saved"]
        self.train_log.appendPlainText(f"[젯슨] 전송 완료 → {remote}")

        # 젯슨(리눅스)에 보낼 경로는 항상 슬래시로. 윈도우에서 Path 를 쓰면 역슬래시가 된다.
        engine_path = Path(remote).with_suffix(".engine").as_posix()
        self.start_remote_task("onnx2engine", "ONNX → ENGINE 변환",
                               lambda r: self.deploy_engine_on_jetson({"engine": engine_path}),
                               onnx=remote)

    def refresh_train_summary(self):
        self.refresh_model_labels()

    def choose_model(self):
        self.select_training_model()

    def prepare_dataset_split(self):
        if self.can_run_on_jetson():
            # PC 의 상품 목록을 같이 보내서 젯슨의 data.yaml 이 PC 기준으로 만들어지게 한다
            if self.busy_with_job():
                return
            self.start_remote_task("split", "데이터셋 분할", self.after_split,
                                   products=self.products)
            return

        import subprocess
        script = BASE_DIR / "import_master_dataset.py"
        try:
            r = subprocess.run([sys.executable, str(script), "--project", str(BASE_DIR)], capture_output=True, text=True, check=True)
            if hasattr(self, "train_log"):
                self.train_log.appendPlainText(r.stdout)
            QMessageBox.information(self, "Master 데이터셋 준비 완료", r.stdout[-1200:] if r.stdout else "완료")
        except subprocess.CalledProcessError as e:
            msg = (e.stdout or "") + "\n" + (e.stderr or "")
            if hasattr(self, "train_log"):
                self.train_log.appendPlainText(msg)
            QMessageBox.critical(self, "데이터셋 준비 실패", msg[-2000:])

    def after_split(self, _result):
        """분할이 끝났다. 합성을 켜 두었으면 이어서 합성까지 한다.

        신상품은 턴테이블로 혼자 있는 모습만 찍힌다. 그 사진만 학습하면
        매대에서 겹쳐 놓인 빵을 못 잡는다. 그래서 찍은 사진에서 빵을 오려내
        여러 개 놓인 장면으로 합성해 train 에 채워 넣는다. (val 은 실제 그대로)
        """
        if not (hasattr(self, "synth_check") and self.synth_check.isChecked()):
            QMessageBox.information(
                self, "데이터셋 준비 완료",
                "젯슨에서 Train/Val 분할이 끝났습니다.\n아래 로그에서 장수와 클래스를 확인하세요.")
            return

        count = self.synth_spin.value()
        self.train_log.appendPlainText(
            f"[자동] 분할 완료 → 합성 {count}장 만들기로 이어갑니다.")

        def done(result):
            self.set_phase(None)
            QMessageBox.information(
                self, "데이터셋 준비 완료",
                "Train/Val 분할과 합성이 모두 끝났습니다.\n\n"
                "합성 사진은 실제 사진에서 빵만 오려내 여러 개 놓은 것이고,\n"
                "라벨은 자동으로 붙었습니다. 아래 로그에서 품목별 개수를 확인하세요.\n\n"
                "이제 「전이학습 시작」 을 누르면 됩니다.")

        self.set_phase(f"합성 중…   {count}장")
        self.start_remote_task("synth", "합성 데이터 생성", done, count=count)

    def generate_data_yaml(self):
        ordered = sorted(self.products.items(), key=lambda x: int(x[0]))

        lines = [
            f"path: {str((BASE_DIR / 'dataset' / 'master').resolve()).replace(chr(92), '/')}",
            "train: images/train",
            "val: images/val",
            "",
            "names:"
        ]

        for cid, p in ordered:
            lines.append(f"  {cid}: {p['class_name']}")

        DATA_YAML.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def generate_data_yaml_with_message(self):
        self.generate_data_yaml()
        self.train_log.appendPlainText(f"[YAML CREATED] {DATA_YAML}")
        QMessageBox.information(
            self,
            "data.yaml 생성 완료",
            f"현재 등록 품목 기준으로 data.yaml을 생성했습니다.\n{DATA_YAML}"
        )

    def try_load_saved_engine(self):
        """시작할 때 POS 모델을 맞춘다.

        원격(PC + 젯슨) 일 때는 **젯슨이 물고 있는 모델을 그대로 따른다.**
        전에는 PC 의 settings.json 에 적힌 경로를 젯슨에 밀어 넣었는데,
        그 경로는 젯슨의 경로라 PC 에서는 존재 확인도 안 되고,
        무엇보다 젯슨에서 의도적으로 띄운 모델을 조용히 갈아치워 버린다.
        모델을 바꾸는 것은 사용자가 「기존 .engine 선택」 을 눌렀을 때만 일어나야 한다.
        """
        if self.ops is not None or os.environ.get("BREADVISION_INFER_URL", "").strip():
            try:
                self.detector.refresh()
            except Exception:
                pass
            path = getattr(self.detector, "model_path", None)
            if path:
                self.settings["active_engine"] = path
            self.refresh_model_labels()
            return

        engine = self.settings.get("active_engine", "")
        if not engine:
            return
        if not Path(engine).exists():
            return
        try:
            self.detector.load(engine)
            self.refresh_model_labels()
        except Exception as e:
            print(f"[ENGINE AUTO LOAD WARNING] {e}")

    def refresh_model_labels(self):
        engine = self.settings.get("active_engine", "")

        if hasattr(self, "engine_path_label"):
            self.engine_path_label.setText(
                f"활성 ENGINE: {Path(engine).name}" if engine else "활성 ENGINE: 미설정"
            )
        if hasattr(self, "training_model_label"):
            pt = self.settings.get("training_base_model", "")
            self.training_model_label.setText(f"기본 .pt: {Path(pt).name}" if pt else "기본 .pt: 미설정")

        if hasattr(self, "active_model_badge"):
            if engine:
                self.active_model_badge.setText(f"ENGINE  {Path(engine).name}")
                self.active_model_badge.setStyleSheet(
                    "background:#0E2740; color:#7CC0FF; padding:8px 12px; "
                    "border:1px solid #245B91; border-radius:6px; font-weight:800;"
                )
            else:
                self.active_model_badge.setText("ENGINE  NOT SET")

    def select_engine_model(self):
        if self.can_run_on_jetson():
            picked = self.pick_jetson_file("engine", "젯슨의 .engine 선택")
            if not picked:
                return
            try:
                res = self.ops.deploy(picked)
            except Exception as e:
                QMessageBox.critical(self, "적용 실패", str(e))
                return
            self.settings["active_engine"] = res.get("deployed", picked)
            self.save_settings()
            try:
                self.detector.refresh()
            except Exception:
                pass
            self.refresh_model_labels()
            QMessageBox.information(self, "적용 완료", res.get("deployed", picked))
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Jetson POS TensorRT ENGINE 선택",
            str(BASE_DIR / "models"),
            "TensorRT Engine (*.engine)"
        )
        if not path:
            return

        try:
            self.detector.load(path)
        except Exception as e:
            QMessageBox.critical(self, "ENGINE 로드 실패", str(e))
            return

        self.settings["active_engine"] = path
        self.settings["active_model_classes"] = {
            str(cid): p["class_name"] for cid, p in self.products.items()
        }
        self.save_settings()
        self.refresh_model_labels()
        QMessageBox.information(
            self,
            "ENGINE 적용 완료",
            f"POS 추론 모델을 적용했습니다.\n\n{Path(path).name}"
        )

    def reload_active_engine(self):
        path = self.settings.get("active_engine", "")
        if not path:
            QMessageBox.information(self, "ENGINE 미설정", "먼저 .engine 모델을 선택해주세요.")
            return
        try:
            self.detector.load(path)
            self.refresh_model_labels()
            QMessageBox.information(self, "ENGINE 로드 완료", Path(path).name)
        except Exception as e:
            QMessageBox.critical(self, "ENGINE 로드 실패", str(e))

    def select_training_model(self):
        if self.can_run_on_jetson():
            picked = self.pick_jetson_file("pt", "젯슨의 .pt 선택 (전이학습 기준 모델)")
            if not picked:
                return
            self.settings["training_base_model"] = picked
            self.save_settings()
            self.refresh_model_labels()
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "전이학습 기본 YOLO 모델 선택",
            str(BASE_DIR / "models"),
            "PyTorch Model (*.pt)"
        )
        if not path:
            return
        self.settings["training_base_model"] = path
        self.save_settings()
        self.refresh_model_labels()

    @staticmethod
    def is_jetson():
        """젯슨에서 돌고 있는가. (TensorRT 변환·긴급 전이학습은 젯슨에서만 가능)"""
        return sys.platform.startswith("linux") and Path("/etc/nv_tegra_release").exists()

    # ── 젯슨 원격 작업 ───────────────────────────────────────────────
    def can_run_on_jetson(self):
        """PC 에서 눌러도 젯슨이 대신 해 줄 수 있는 상태인가."""
        return self.ops is not None

    def busy_with_job(self):
        if self.remote_job:
            QMessageBox.warning(self, "작업 진행 중",
                                "젯슨에서 작업이 진행 중입니다. 끝난 뒤에 다시 실행하세요.")
            return True
        if self.train_process and self.train_process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "작업 진행 중", "이미 진행 중인 작업이 있습니다.")
            return True
        return False

    def start_remote_task(self, task, title, on_success, **args):
        """젯슨에 작업을 걸고, 로그를 1초마다 받아 학습 로그창에 이어 붙인다."""
        try:
            job = self.ops.start(task, **args)
        except Exception as e:
            QMessageBox.critical(self, f"{title} 실패", str(e))
            return
        self.remote_job = {"id": job, "next": 0, "title": title, "on_success": on_success}
        self.train_log.appendPlainText("")
        self.train_log.appendPlainText(f"[젯슨] {title} 시작")
        self.statusBar().showMessage(f"젯슨에서 {title} 진행 중…")
        self.remote_timer.start()

    def poll_remote_job(self):
        job = self.remote_job
        if not job:
            self.remote_timer.stop()
            return
        try:
            st = self.ops.poll(job["id"], job["next"])
        except Exception as e:
            self.train_log.appendPlainText(f"[젯슨] 연결 끊김: {e}")
            self.set_phase(None)
            self.finish_remote_job(None)
            return

        for line in st.get("lines", []):
            self.train_log.appendPlainText(line)
        job["next"] = st.get("next", job["next"])

        if st.get("running"):
            return

        code = st.get("exit")
        if code == 0:
            self.train_log.appendPlainText(f"[젯슨] {job['title']} 완료")
            self.finish_remote_job(st.get("result") or {})
        else:
            self.train_log.appendPlainText(f"[젯슨] {job['title']} 실패 (exit={code})")
            self.set_phase(None)
            self.finish_remote_job(None)
            QMessageBox.warning(self, f"{job['title']} 실패",
                                "젯슨에서 작업이 정상적으로 끝나지 않았습니다.\n"
                                "아래 로그를 확인해주세요.")

    def finish_remote_job(self, result):
        self.remote_timer.stop()
        job, self.remote_job = self.remote_job, None
        self.statusBar().clearMessage()
        if result is not None and job and job.get("on_success"):
            job["on_success"](result)

    def pick_jetson_file(self, kind, title):
        """젯슨에 있는 파일을 목록에서 고른다. (PC 에서 젯슨 폴더를 뒤질 수 없으므로)"""
        try:
            files = self.ops.files(kind)
        except Exception as e:
            QMessageBox.critical(self, "목록을 받지 못했습니다", str(e))
            return None
        if not files:
            QMessageBox.information(self, "파일 없음",
                                    f"젯슨에 {kind} 파일이 없습니다.")
            return None
        labels = [f"{f['path']}   ({f['size']//1024//1024} MB)" for f in files]
        choice, ok = QInputDialog.getItem(self, title, "젯슨에 있는 파일:", labels, 0, False)
        if not ok:
            return None
        return files[labels.index(choice)]["path"]

    def require_jetson(self, feature):
        """젯슨 전용 기능을 PC 에서 눌렀을 때 원인을 알려준다."""
        if self.is_jetson():
            return True
        QMessageBox.information(
            self,
            f"{feature} — 젯슨에서만 가능",
            f"{feature} 은(는) 젯슨에서만 동작합니다.\n\n"
            "지금 이 GUI 는 PC 에서 돌고 있고, 추론만 젯슨에 맡기는 구성입니다.\n"
            "이 기능은 젯슨 GPU(TensorRT · Docker) 를 직접 써야 해서 PC 에서는 실행할 수 없습니다.\n\n"
            "[운영 원칙]\n"
            "  · 정식 학습     → Colab (BreadVision_Colab_Train.ipynb)\n"
            "  · 긴급 전이학습 → 젯슨에서 이 GUI 를 직접 실행\n"
            "  · ENGINE 변환   → 젯슨에서 ./onnx_to_engine.sh models/best.onnx"
        )
        return False

    def start_training(self):
        if self.can_run_on_jetson():
            self.start_training_on_jetson()
            return
        if not self.require_jetson("긴급 전이학습"):
            return

        model_path = self.settings.get("training_base_model", "").strip()

        if not model_path:
            QMessageBox.warning(
                self,
                "기존 모델 필요",
                "STEP 1에서 기존 .pt 모델을 먼저 선택해주세요."
            )
            return

        if not Path(model_path).exists():
            QMessageBox.warning(self, "모델 파일 없음", "선택한 .pt 파일을 찾을 수 없습니다.")
            return

        if len(list(TRAIN_IMG.glob("*.jpg"))) == 0:
            QMessageBox.warning(
                self,
                "학습 데이터 없음",
                "STEP 2의 「Train / Val 80:20 자동 분할」을 먼저 실행해주세요."
            )
            return

        if not DATA_YAML.exists():
            self.generate_data_yaml()

        if self.train_process and self.train_process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "학습 진행 중", "현재 YOLO 학습이 이미 진행 중입니다.")
            return

        run_name = "bread_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        self.train_process = QProcess(self)
        self.train_process.setProcessChannelMode(QProcess.MergedChannels)
        self.train_process.readyReadStandardOutput.connect(self.read_training_output)
        self.train_process.finished.connect(self.training_finished)

        args = [
            str(BASE_DIR / "training" / "run_training_container.sh"),
            model_path,
            str(DATA_YAML),
            str(self.epochs_spin.value()),
            str(self.imgsz_spin.value()),
            run_name,
        ]

        self.set_phase(f"1 / 3  전이학습 중…   Epoch {self.epochs_spin.value()} · "
                       f"약 {max(1, self.epochs_spin.value() * 100 // 60)}분 예상")
        self.train_log.appendPlainText("")
        self.train_log.appendPlainText("[TRAIN START]")
        self.train_log.appendPlainText(f"기존 모델: {model_path}")
        self.train_log.appendPlainText(
            f"Epoch={self.epochs_spin.value()}, imgsz={self.imgsz_spin.value()}"
        )

        self.train_process.start("bash", args)

    def start_training_on_jetson(self):
        """PC 에서 눌러 젯슨에서 학습시킨다. 로그는 아래 창에 그대로 흐른다."""
        if self.busy_with_job():
            return
        model = self.settings.get("training_base_model", "").strip()
        if not model:
            QMessageBox.warning(self, "기존 모델 필요",
                                "먼저 「기존 best.pt 선택」에서 젯슨에 있는 .pt 를 골라주세요.")
            return

        epochs = self.epochs_spin.value()
        imgsz = self.imgsz_spin.value()
        answer = QMessageBox.question(
            self, "젯슨에서 학습 시작",
            f"젯슨에서 전이학습을 시작합니다.\n\n"
            f"기준 모델: {model}\nEpoch: {epochs}   이미지 크기: {imgsz}\n\n"
            f"Epoch 30 기준 약 50분 걸립니다.\n"
            "그동안 이 창을 닫아도 젯슨에서는 계속 진행됩니다. 시작할까요?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer != QMessageBox.Yes:
            return

        def trained(result):
            best = result.get("best_pt", "")
            self.last_best_model = best
            if best:
                self.settings["last_best_model"] = best
                self.save_settings()
            if not best:
                self.set_phase(None)
                QMessageBox.warning(
                    self, "best.pt 를 찾지 못했습니다",
                    "학습은 끝났지만 결과 경로를 로그에서 찾지 못했습니다.\n"
                    "「변환·적용만 다시 하기」 로 직접 골라 진행해주세요.")
                return
            # 학습이 끝나면 곧바로 변환 → 적용까지 이어서 한다.
            self.set_phase(f"2 / 3  ENGINE 변환 중…   ({Path(best).parent.parent.name})")
            self.train_log.appendPlainText(f"[자동] 학습 완료 → ENGINE 변환으로 이어갑니다: {best}")
            self.start_remote_task("engine", "ENGINE 변환", self.deploy_engine_on_jetson,
                                   pt=best, imgsz=imgsz)

        self.set_phase(f"1 / 3  전이학습 중…   Epoch {epochs} · 약 {max(1, epochs * 100 // 60)}분 예상")
        self.start_remote_task("train", "전이학습", trained,
                               model=model, epochs=epochs, imgsz=imgsz)

    def read_training_output(self):
        if not self.train_process:
            return

        text = bytes(
            self.train_process.readAllStandardOutput()
        ).decode("utf-8", errors="replace")

        for line in text.splitlines():
            self.train_log.appendPlainText(line)
            if line.startswith("BEST_MODEL="):
                self.last_best_model = line.split("=", 1)[1].strip()

    def training_finished(self, exit_code, exit_status):
        if exit_code == 0:
            self.train_log.appendPlainText("[TRAIN COMPLETE]")
            best = self.find_last_best_model()
            if best is not None:
                # 학습이 끝나면 곧바로 변환 → 적용까지 이어서 한다. (버튼 한 번으로 끝나게)
                self.set_phase(f"2 / 3  ENGINE 변환 중…   ({best.parent.parent.name})")
                self.train_log.appendPlainText(f"[자동] 학습 완료 → ENGINE 변환으로 이어갑니다: {best}")
                self.start_local_engine_build(best, self.imgsz_spin.value())
                return

            self.set_phase(None)
            QMessageBox.information(
                self,
                "YOLO 학습 완료",
                "학습은 끝났지만 best.pt 를 찾지 못했습니다.\n"
                "「변환·적용만 다시 하기」 로 직접 진행해주세요."
            )
        else:
            self.train_log.appendPlainText(f"[TRAIN ERROR] exit_code={exit_code}")
            self.set_phase(None)
            QMessageBox.warning(
                self,
                "학습 실패",
                "YOLO 학습이 정상적으로 끝나지 않았습니다.\n"
                "아래 로그와 ultralytics 설치 여부를 확인해주세요."
            )

    def find_last_best_model(self):
        """이번 실행에서 학습한 best.pt. 없으면 runs/ 에서 가장 최근 것을 찾는다.

        (앱을 다시 켠 뒤에도 직전 학습 결과를 변환할 수 있게 한다)
        """
        if self.last_best_model and Path(self.last_best_model).exists():
            return Path(self.last_best_model)

        candidates = sorted(RUNS_DIR.glob("*/weights/best.pt"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    def apply_last_best_model(self):
        """best.pt → ONNX → TensorRT ENGINE 변환 후 POS 활성 모델로 적용."""
        if self.can_run_on_jetson():
            self.build_engine_on_jetson()
            return
        if not self.require_jetson("ENGINE 변환"):
            return

        best = self.find_last_best_model()
        if best is None:
            QMessageBox.warning(
                self, "best.pt 없음",
                "학습된 best.pt 를 찾을 수 없습니다. 먼저 모델 학습을 완료해주세요."
            )
            return

        if self.train_process and self.train_process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "작업 진행 중",
                                "학습 또는 변환이 이미 진행 중입니다.")
            return

        imgsz = self.imgsz_spin.value()
        answer = QMessageBox.question(
            self, "ENGINE 변환",
            f"아래 모델을 TensorRT ENGINE 으로 변환하고 POS 에 적용합니다.\n\n"
            f"{best}\n이미지 크기: {imgsz}\n\n"
            "몇 분 걸립니다. 진행할까요?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if answer != QMessageBox.Yes:
            return

        self.start_local_engine_build(best, imgsz)

    def start_local_engine_build(self, best, imgsz):
        """젯슨에서 직접 실행할 때의 ENGINE 변환. (학습 직후 자동 연결에서도 쓴다)"""
        self.pending_engine = str(Path(best).with_suffix(".engine"))
        self.train_log.appendPlainText("")
        self.train_log.appendPlainText("[ENGINE BUILD START]")
        self.train_log.appendPlainText(f"입력: {best}")

        self.train_process = QProcess(self)
        self.train_process.setProcessChannelMode(QProcess.MergedChannels)
        self.train_process.readyReadStandardOutput.connect(self.read_training_output)
        self.train_process.finished.connect(self.engine_build_finished)
        self.train_process.start("bash", [
            str(BASE_DIR / "training" / "export_and_build_engine.sh"),
            str(best), str(imgsz),
        ])

    def build_engine_on_jetson(self):
        """젯슨에서 best.pt → ONNX → ENGINE 을 만들고 곧바로 POS 에 적용한다."""
        if self.busy_with_job():
            return
        best = (getattr(self, "last_best_model", "")
                or self.settings.get("last_best_model", ""))
        if not best:
            best = self.pick_jetson_file("pt", "변환할 .pt 선택")
            if not best:
                return

        imgsz = self.imgsz_spin.value()
        answer = QMessageBox.question(
            self, "ENGINE 변환 및 적용",
            f"젯슨에서 아래 모델을 TensorRT ENGINE 으로 만들고 POS 에 적용합니다.\n\n"
            f"{best}\n이미지 크기: {imgsz}\n\n몇 분 걸립니다. 진행할까요?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer != QMessageBox.Yes:
            return

        self.start_remote_task("engine", "ENGINE 변환", self.deploy_engine_on_jetson,
                               pt=best, imgsz=imgsz)

    def deploy_engine_on_jetson(self, result):
        """변환된 엔진을 젯슨 추론 서버에 즉시 물린다."""
        engine = result.get("engine", "")
        if not engine:
            self.set_phase(None)
            QMessageBox.warning(self, "ENGINE 없음",
                                "변환은 끝났지만 .engine 경로를 로그에서 찾지 못했습니다.")
            return
        self.set_phase("3 / 3  POS 에 적용 중…")
        try:
            res = self.ops.deploy(engine)
        except Exception as e:
            self.set_phase(None)
            QMessageBox.critical(self, "적용 실패", str(e))
            return
        self.settings["active_engine"] = res.get("deployed", engine)
        self.save_settings()
        try:
            self.detector.refresh()      # 서버가 바꾼 모델을 다시 확인
        except Exception:
            pass
        self.refresh_model_labels()
        self.train_log.appendPlainText(f"[젯슨] POS 적용 완료 → {res.get('deployed')}")
        self.set_phase(None)
        QMessageBox.information(
            self, "완료",
            f"새 모델이 POS 에 적용됐습니다. 지금부터 바로 이 모델로 인식합니다.\n\n"
            f"{res.get('deployed')}\n\n"
            "프로그램을 다시 켜도 이 모델이 유지됩니다.")

    def engine_build_finished(self, exit_code, exit_status):
        if exit_code != 0:
            self.set_phase(None)
            self.train_log.appendPlainText(f"[ENGINE BUILD ERROR] exit_code={exit_code}")
            QMessageBox.warning(self, "변환 실패",
                                "ENGINE 변환이 정상적으로 끝나지 않았습니다.\n로그를 확인해주세요.")
            return

        engine = getattr(self, "pending_engine", "")
        if not engine or not Path(engine).exists():
            self.train_log.appendPlainText("[ENGINE BUILD] 파일을 찾지 못했습니다.")
            QMessageBox.warning(self, "ENGINE 없음",
                                "변환은 끝났지만 .engine 파일을 찾지 못했습니다.")
            return

        self.train_log.appendPlainText(f"[ENGINE BUILD COMPLETE] {engine}")
        self.set_phase("3 / 3  POS 에 적용 중…")
        try:
            self.detector.load(engine)
        except Exception as e:
            QMessageBox.warning(self, "ENGINE 로드 실패", str(e))
            return

        self.settings["active_engine"] = engine
        self.save_settings()
        self.refresh_model_labels()
        self.set_phase(None)
        QMessageBox.information(
            self, "완료",
            f"새 모델이 POS 에 적용됐습니다. 지금부터 바로 이 모델로 인식합니다.\n\n"
            f"{Path(engine).name}\n\n"
            "프로그램을 다시 켜도 이 모델이 유지됩니다."
        )

    def show_frame_on_label(self, frame, label):
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        label.setPixmap(
            pix.scaled(
                label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def closeEvent(self, event):
        self.stop_camera()
        self.detector.shutdown()
        self.qr_worker.shutdown()
        if self.train_process and self.train_process.state() != QProcess.NotRunning:
            self.train_process.kill()
        event.accept()
