"""데이터 수집 탭 — 새 빵 사진을 찍으며 YOLO 라벨을 함께 저장한다.

두 방식
    한 개: 흰 바탕에 빵 1개. 배경 분리로 박스를 잡고, 품목은 콤보박스에서 고른 것으로 저장.
    여러 개: 화면의 빵 전부. 모델이 부른 이름을 바로잡거나 무시하고, 촬영 대상을 고정한다
            (규칙은 core/collect_labeler.py).
"""
from datetime import datetime

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                             QFrame, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QProgressBar,
                             QSpinBox, QVBoxLayout, QWidget)

from ai.segment import object_box
from core import collect_labeler as rules
from core import dataset_io
from core import products as product_db
from core.boxes import SteadyBox, center_guide_box, pick_label_box, yolo_line
from core.collect_labeler import CollectLabeler
from core.log import get_logger
from gui import collect_preview, style
from gui.camera_view import CameraView
from gui.style import button, page_header
from gui.turntable_worker import TurntableWorker

log = get_logger("collect")

SINGLE_MODEL_CONF = 0.35    # 한 개 방식: 배경 분리가 안 돼 모델 박스를 쓸 때 최소 신뢰도

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
PICK_HINT = (
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


class CollectTab(QWidget):
    def __init__(self, ctx, camera, live, preview_max=0, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.camera = camera
        self.live = live
        self.labeler = CollectLabeler()
        self.steady = SteadyBox()
        self.last_candidate = None      # 한 개 방식: 잡았다가 버린 후보 (주황으로 보여준다)

        self.collecting = False
        self.count = 0
        self.skipped = 0
        self.target = 0
        self.turn_worker = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.capture)

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        page_header(root, "신규 상품 데이터 수집",
                    "빵 1개를 빨간 중앙 박스 안에 놓으면 이미지와 YOLO Bounding Box 라벨을 자동 저장합니다.")

        content = QHBoxLayout()
        left_card = QFrame()
        left_card.setObjectName("Card")
        left = QVBoxLayout(left_card)
        self.view = CameraView("먼저 「수집용 카메라 켜기」를 눌러주세요.", 900, 650, preview_max,
                               "background:#2B211C; color:#FFF8ED; border-radius:14px; font-size:16px;")
        left.addWidget(self.view)

        right_card = QFrame()
        right_card.setObjectName("Card")
        right = QVBoxLayout(right_card)
        right.addWidget(self._build_settings())
        right.addWidget(self._build_pick_group())

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        right.addWidget(self.progress)
        self.status = QLabel("현재 상태: 대기 중")
        self.status.setStyleSheet(style.STATUS_BOX)
        self.status.setWordWrap(True)
        right.addWidget(self.status)

        right.addWidget(button("수집 카메라 시작", lambda: self.camera.start_with_message(self),
                               "Primary", "카메라를 켜서 신규 빵 배치 상태를 확인합니다."))
        right.addWidget(button("자동 수집 시작", self.start_collection, "Success",
                               "설정한 간격으로 자동 촬영하며 JPG 이미지와 YOLO TXT 라벨을 함께 저장합니다."))
        right.addWidget(button("수집 중지", self.stop_collection, "Danger",
                               "진행 중인 자동 촬영을 즉시 중지합니다."))
        right.addWidget(button("학습용 ZIP으로 내보내기", self.export_zip, tooltip=(
            "수집한 사진과 라벨을 dataset/import/ 에 ZIP 으로 저장합니다.\n"
            "그 뒤 「모델 배포 및 모델 학습」 탭에서 Train/Val 분할 → 학습을 진행하세요.")))
        right.addWidget(button("수집 데이터 비우기", self.clear_data, "Danger", (
            "지금까지 찍은 사진과 라벨을 모두 지웁니다.\n"
            "지우기 전에 몇 장인지, 어떤 품목인지 보여주고 한 번 더 물어봅니다.")))

        self.guide = QLabel()
        self.guide.setWordWrap(True)
        self.guide.setStyleSheet(style.NOTE_BOX)
        right.addWidget(self.guide)
        right.addStretch()

        content.addWidget(left_card, 3)
        content.addWidget(right_card, 1)
        root.addLayout(content)

        ctx.products_changed.connect(self.refresh_products)
        camera.stopped.connect(self._on_camera_stopped)
        self.refresh_products()
        self.apply_mode()

    # ── 화면 구성 ────────────────────────────────────────────────────────
    def _build_settings(self):
        group = QGroupBox("수집 설정")
        form = QFormLayout(group)

        self.product_combo = QComboBox()
        self.target_spin = _spin(10, 5000, 200, " 장")
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.1, 10.0)
        self.interval_spin.setSingleStep(0.1)
        self.interval_spin.setValue(0.5)
        self.interval_spin.setSuffix(" 초")

        self.auto_label_check = QCheckBox("AI 자동 라벨 (빵을 움직이며 촬영)")
        self.auto_label_check.setChecked(True)
        self.auto_label_check.setToolTip(
            "현재 POS 모델로 빵 위치를 찾아 그 박스를 라벨로 저장합니다.\n"
            "끄면 화면 정중앙 고정 박스를 라벨로 씁니다(빵을 가이드 박스에 맞춰야 함).")

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("한 개 — 빵 1개 자동 인식 (예전 방식)", "single")
        self.mode_combo.addItem("여러 개 — 박스 고르기·고정", "multi")
        self.mode_combo.setToolTip(
            "한 개: 흰 바탕에 빵 1개만 올리면 박스를 자동으로 잡아 고른 품목으로 저장합니다.\n"
            "여러 개: 화면의 빵을 모두 라벨링합니다. 모르는 빵은 골라서 품목을 지정해야 저장됩니다.")
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(
            self.ctx.settings.get("collect_mode", "single"))))
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        self.turn_check = QCheckBox("촬영 중 턴테이블 자동 회전")
        self.turn_check.setChecked(True)
        self.turn_check.setToolTip("목표 장수에 맞춰 조금씩 돌리며 찍습니다.\n"
                                   "카메라 뒤 케이블 때문에 한 방향으로 계속 돌지 않고 왕복만 합니다.")
        self.span_spin = _spin(10, 180, 180, " 도", "왼쪽으로 이만큼 갔다가 오른쪽으로 이만큼 돌아옵니다.")
        # 2번(좌우) 모터. 촬영 내내 가운데 → 왼쪽 → 오른쪽 → 가운데 로 한 번 훑는다.
        self.span2_spin = _spin(0, 720, 400, " 도",
                                "촬영하는 동안 좌우 모터가 가운데 → 왼쪽 → 오른쪽 → 가운데 로 한 번 움직입니다.\n"
                                "이 장비에서는 400도가 끝에서 끝까지입니다.\n"
                                "회전판과 동시에 움직여서 같은 빵을 더 여러 각도로 담습니다.\n"
                                "0 으로 두면 좌우 모터는 움직이지 않습니다.")
        self.roi_spin = _spin(0, 100, 60, " %",
                              "화면 가운데 이 범위 안에 있는 물체만 잡습니다 (한 개 방식).\n"
                              "가장자리의 케이블·손·트레이 모서리를 무시해서 박스가 덜 튑니다.\n"
                              "0 으로 두면 화면 전체를 봅니다.")
        self.smooth_check = QCheckBox("끊김 없이 계속 움직이기")
        self.smooth_check.setChecked(True)
        self.smooth_check.setToolTip("촬영 시간 내내 두 모터가 아주 느리게 계속 움직입니다 (권장).\n"
                                     "끄면 한 장 찍을 때마다 조금 움직였다 멈추는 예전 방식으로 돕니다.")

        form.addRow("수집할 상품", self.product_combo)
        form.addRow("수집 방식", self.mode_combo)
        form.addRow("목표 촬영 수", self.target_spin)
        form.addRow("자동 촬영 간격", self.interval_spin)
        form.addRow("라벨 방식", self.auto_label_check)
        form.addRow("턴테이블", self.turn_check)
        form.addRow("왕복 각도 (±)", self.span_spin)
        form.addRow("가운데 영역", self.roi_spin)
        form.addRow("좌우 모터 (±)", self.span2_spin)
        form.addRow("움직임", self.smooth_check)
        return group

    def _build_pick_group(self):
        group = QGroupBox("박스 고르기 — 이름 바로잡기")
        pick = QVBoxLayout(group)
        hint = QLabel(PICK_HINT)
        hint.setObjectName("PageDesc")
        hint.setWordWrap(True)
        pick.addWidget(hint)
        pick.addWidget(button("다음 ▶", self.cycle_focus, tooltip=(
            "고를 박스를 바꿉니다. 고른 것은 빨간 테두리가 됩니다.\n초록(모델이 부른 것)도 고를 수 있습니다.")))
        self.lock_btn = button("이것으로 고정 (촬영 대상)", self.toggle_lock, "Success", (
            "고른 박스를 촬영 대상으로 고정합니다.\n"
            "고정하면 인식이 잠깐 끊겨도 빨간 테두리가 그 자리에 머물고,\n"
            "이름 없는 주황 박스는 화면에서 사라집니다."))
        pick.addWidget(self.lock_btn)
        pick.addWidget(button("이것을 위 품목으로 지정", self.assign_focus, "Primary", (
            "고른 박스의 라벨을 「수집할 상품」 으로 덮어씁니다.\n모델이 뭐라고 불렀든 이 품목으로 저장됩니다.")))
        pick.addWidget(button("이건 빵이 아님 (무시)", self.ignore_focus, tooltip=(
            "책상·손·그림자, 또는 모델 오검출을 후보에서 뺍니다.\n그대로 두면 저장이 막히거나 틀린 라벨이 학습됩니다.")))
        pick.addWidget(button("지정·무시 초기화", self.clear_assign,
                              tooltip="바꾼 품목과 무시를 모두 되돌립니다 (모델이 부른 대로)."))
        self.pick_group = group
        return group

    # ── 품목·방식 ────────────────────────────────────────────────────────
    def refresh_products(self):
        current = self.product_combo.currentData()
        self.product_combo.clear()
        for cid, p in product_db.ordered(self.ctx.products):
            n = dataset_io.count_collected(self.ctx.paths.collect_img, p["class_name"])
            self.product_combo.addItem(f"[Class {cid}] {p['display_name']} — 현재 {n}장", int(cid))
        if current is not None:
            self.select_product(current)

    def select_product(self, class_id):
        idx = self.product_combo.findData(int(class_id))
        if idx >= 0:
            self.product_combo.setCurrentIndex(idx)

    def selected_class_id(self):
        data = self.product_combo.currentData()
        return None if data is None else int(data)

    def mode(self):
        return self.mode_combo.currentData()

    def apply_mode(self):
        single = self.mode() == "single"
        self.pick_group.setVisible(not single)
        self.guide.setText(GUIDE_SINGLE if single else GUIDE_MULTI)

    def on_mode_changed(self, _index=None):
        self.ctx.settings["collect_mode"] = self.mode()
        self.ctx.save_settings()
        # 여러 개 방식에서 고른·고정한 상태가 남으면 한 개 방식과 섞인다. 비운다.
        self.clear_assign()
        self.apply_mode()

    # ── 여러 개 방식 조작 ────────────────────────────────────────────────
    def _show(self, message):
        if message:
            self.status.setText(message)
        else:
            self.update_hint()

    def cycle_focus(self):
        self._show(self.labeler.cycle())

    def assign_focus(self):
        cid = self.selected_class_id()
        if cid is not None:
            self._show(self.labeler.assign(cid))

    def ignore_focus(self):
        self._show(self.labeler.ignore())

    def toggle_lock(self):
        cid = self.selected_class_id()
        if cid is None and not self.labeler.lock:
            return
        locked, message = self.labeler.toggle_lock(self.camera.frame, cid if cid is not None else 0)
        self.lock_btn.setText("고정 해제" if locked else "이것으로 고정 (촬영 대상)")
        self._show(message)

    def clear_assign(self):
        self.labeler.clear()
        self.lock_btn.setText("이것으로 고정 (촬영 대상)")
        self.update_hint()

    def update_hint(self):
        self.status.setText(self.labeler.hint(self.ctx.products))

    # ── 미리보기 ────────────────────────────────────────────────────────
    def render(self, frame, now):
        auto = self.auto_label_check.isChecked()
        if not auto:
            self.view.show_frame(frame, guide=True)
            return
        # 아는 빵은 모델이 자동으로 라벨을 달아 주므로 계속 추론을 돌린다.
        self.live.update(frame, self.camera.last_frame_id, now)
        if self.mode() == "single":
            box, why = self.label_box_pixels(frame, self.live.dets)
            cid = self.selected_class_id()
            name = self.ctx.class_names().get(cid, "?") if cid is not None else "?"
            view = collect_preview.draw_single(frame, self.roi_spin.value() / 100.0, box, why,
                                               self.last_candidate, name)
        else:
            boxes = self.labeler.refresh(frame, self.live.dets)
            view = collect_preview.draw_multi(frame, boxes, self.labeler.focus, self.ctx.class_names(),
                                              bool(self.labeler.lock), rules.LOW_CONF)
        self.view.show_frame(view, guide=False)

    def label_box_pixels(self, frame, detections=None):
        """한 개 방식의 라벨 박스 (x1,y1,x2,y2) 와 이유.

        신메뉴는 모델이 모르는 빵이므로 배경 분리를 먼저 쓴다 (색·밝기로 잡는다. 클래스는 안 본다).
        배경이 지저분해 분리가 안 되면, 아는 빵일 수도 있으니 모델 검출로 넘어간다.
        """
        box, why, cand = object_box(frame, min_area=rules.AUTOLABEL_MIN_AREA,
                                    max_area=rules.AUTOLABEL_MAX_AREA,
                                    roi=self.roi_spin.value() / 100.0)
        self.last_candidate = cand
        if box is not None:
            return self.steady.update(box), "auto"

        detector = self.ctx.detector
        if detections is None and detector.ready():
            try:
                _, detections, _ = detector.infer(frame)
            except Exception as e:          # noqa: BLE001 - 촬영은 계속하고, 이 장면만 건너뛴다
                log.warning("수집 중 추론 실패 (이 장면은 건너뜀): %s", e)
                detections = []
        best, _why2 = pick_label_box(detections or [], frame.shape, SINGLE_MODEL_CONF,
                                     rules.AUTOLABEL_MIN_AREA, rules.AUTOLABEL_MAX_AREA)
        if best is not None:
            return self.steady.update(tuple(best["xyxy"])), "model"
        return self.steady.update(None), why

    # ── 수집 ────────────────────────────────────────────────────────────
    def side_sweep_note(self, span2):
        """좌우 왕복(가운데→왼쪽→오른쪽→가운데)이 촬영 시간 안에 끝나는지 알려 준다."""
        if span2 <= 0 or not self.smooth_check.isChecked():
            return ""
        need = 4.0 * span2 / 360.0 * 2038 * 0.002          # 좌우 모터가 낼 수 있는 최고 속도 기준
        interval = float(self.interval_spin.value())
        have = self.target * interval
        if need <= have:
            return ""
        return (f"\n⚠ 좌우 ±{span2:.0f}도 왕복에 최소 {need:.0f}초가 필요한데 촬영은 {have:.0f}초입니다.\n"
                f"   장수를 {int(need / max(0.1, interval)) + 1}장 이상으로 늘리거나 각도를 줄이세요.")

    def start_collection(self):
        if self.camera.frame is None:
            QMessageBox.warning(self, "카메라 필요", "먼저 「수집용 카메라 켜기」를 눌러주세요.")
            return
        if self.product_combo.count() == 0:
            QMessageBox.warning(self, "품목 없음", "② 품목 · 가격 관리에서 상품을 먼저 등록해주세요.")
            return

        self.count = 0
        self.skipped = 0
        self.steady.reset()
        self.target = self.target_spin.value()
        self.collecting = True
        self.progress.setValue(0)
        if self.turn_check.isChecked():
            # 턴테이블이 박자를 잡는다. 한 칸 돌고 → 한 장 찍고 → 다음 칸.
            # 타이머로 따로 찍으면 도는 중에 찍혀 흔들린 사진이 섞인다.
            span, span2 = float(self.span_spin.value()), float(self.span2_spin.value())
            smooth = self.smooth_check.isChecked()
            self.turn_worker = TurntableWorker(self.target, span=span, span2=span2,
                                               interval=float(self.interval_spin.value()), smooth=smooth)
            self.turn_worker.arrived.connect(self.on_turntable_arrived)
            self.turn_worker.failed.connect(self.on_turntable_failed)
            self.turn_worker.done.connect(self.on_turntable_done)
            self.status.setText(
                f"현재 상태: 자동 수집 중 (턴테이블)\n"
                f"±{span:.0f}도 왕복, 한 칸 {self.turn_worker.step_deg():.1f}도 x {self.target}장"
                + (f"\n좌우 모터: 가운데 → 왼쪽 → 오른쪽 → 가운데 (±{span2:.0f}도)" if span2 > 0 else "")
                + ("\n끊김 없이 계속 움직이는 중" if smooth else "")
                + self.side_sweep_note(span2))
            self.turn_worker.start()
        else:
            self.status.setText("현재 상태: 자동 수집 중\n빵을 빨간 박스 안에서 조금씩 회전/이동해주세요.")
            self.timer.start(int(self.interval_spin.value() * 1000))
        log.info("수집 시작: class %s, 목표 %s장, 방식 %s", self.selected_class_id(), self.target, self.mode())

    def on_turntable_arrived(self, _angle):
        """한 칸 돌고 멈췄다. 지금 한 장 찍는다. (GUI 스레드)"""
        if not self.collecting:
            return
        self.capture()
        if self.turn_worker:
            self.turn_worker.shot_done()          # 다음 칸으로 가도 된다

    def on_turntable_failed(self, message):
        self.collecting = False
        self.status.setText(f"현재 상태: 턴테이블 오류\n{message}")
        QMessageBox.warning(self, "턴테이블",
                            f"{message}\n\n턴테이블 없이 찍으려면 「촬영 중 턴테이블 자동 회전」 을 끄세요.")

    def on_turntable_done(self):
        self.turn_worker = None
        if self.collecting:
            self.collecting = False
            self.status.setText(f"현재 상태: 회전 왕복 완료\n이번 세션에서 {self.count}장 저장됨")
            self.ctx.products_changed.emit()      # 수집 장수 표시 갱신

    def stop_collection(self):
        self.collecting = False
        self.timer.stop()
        if self.turn_worker:
            self.turn_worker.stop()               # 제자리로 돌아온 뒤 끝난다
        self.status.setText(f"현재 상태: 수집 중지\n이번 세션에서 {self.count}장 저장됨")

    def _on_camera_stopped(self):
        self.timer.stop()
        self.collecting = False
        self.view.reset_size()

    def _label_lines(self, frame, cid, product):
        """이 장면의 라벨 줄들과 상태창 꼬리말. 저장하면 안 되는 장면이면 (None, 이유)."""
        h, w = frame.shape[:2]
        if not self.auto_label_check.isChecked():
            # 고정 박스: 빵을 화면 중앙 가이드에 맞춰 놓아야 한다
            return [yolo_line(cid, center_guide_box(w, h), w, h)], ""

        if self.mode() == "single":
            box, why = self.label_box_pixels(frame)
            if box is None:
                # 라벨을 믿을 수 없으면 저장하지 않는다. 잘못된 라벨은 모델을 망친다.
                return None, (f"현재 상태: 자동 수집 중 (한 개 방식)\n"
                              f"{product['display_name']} — {self.count}/{self.target}장\n"
                              f"방금 건너뜀: {why}  (누적 {self.skipped + 1}장)")
            return [yolo_line(cid, box, w, h)], f"  [한 개 · {'배경 분리' if why == 'auto' else '모델'}]"

        # 여러 개: 하나라도 품목이 안 정해졌으면 저장하지 않는다 —
        # 라벨 없는 빵이 사진에 남으면 YOLO 가 그것을 배경으로 학습해 버린다.
        boxes = self.labeler.label_boxes(frame, self.live.dets)
        todo = [b for b in boxes if b["class_id"] is None]
        if not boxes or todo:
            why = "잡힌 물체 없음" if not boxes else f"이름 없는 것 {len(todo)}개"
            return None, (f"현재 상태: 자동 수집 중 (AI 자동 라벨)\n"
                          f"{self.count}/{self.target}장\n"
                          f"방금 건너뜀: {why}  (누적 {self.skipped + 1}장)\n"
                          "「다음 ▶」 로 빨간 테두리를 옮겨 지정하거나 무시하세요.")
        lines = [yolo_line(b["class_id"], b["xyxy"], w, h) for b in boxes]
        fixed_n = sum(1 for b in boxes if b["fixed"])
        return lines, f"  [라벨 {len(lines)}개 · 내가 바꾼 것 {fixed_n}]"

    def capture(self):
        if not self.collecting or self.camera.frame is None:
            return
        cid = self.selected_class_id()
        product = self.ctx.products.get(str(cid)) if cid is not None else None
        if product is None:
            self.stop_collection()
            return

        frame = self.camera.frame.copy()
        lines, note = self._label_lines(frame, cid, product)
        if lines is None:
            self.skipped += 1
            self.status.setText(note)
            return

        try:
            _stem, existing = dataset_io.save_sample(self.ctx.paths.collect_img, self.ctx.paths.collect_lbl,
                                                     product["class_name"], frame, lines)
        except OSError as e:
            log.error("사진 저장 실패: %s", e)
            self.stop_collection()
            QMessageBox.critical(self, "저장 실패", f"사진을 저장하지 못해 수집을 멈췄습니다.\n\n{e}\n\n"
                                                    "디스크 공간과 폴더 권한을 확인하세요.")
            return

        self.count += 1
        self.progress.setValue(min(100, int(self.count / self.target * 100)))
        self.status.setText(f"현재 상태: 자동 수집 중{note}\n"
                            f"{product['display_name']} — {self.count}/{self.target}장 (누적 {existing + 1}장)")

        if self.count >= self.target:
            self.timer.stop()
            if self.turn_worker:
                self.turn_worker.stop()           # 목표를 채웠으면 모터도 멈춘다
            self.collecting = False
            self.ctx.products_changed.emit()
            log.info("수집 완료: %s %s장", product["class_name"], self.count)
            QMessageBox.information(self, "데이터 수집 완료",
                                    f"{product['display_name']} 데이터 {self.count}장 수집 완료\n\n"
                                    "JPG 이미지와 YOLO Bounding Box TXT 라벨이 함께 저장되었습니다.")

    # ── 내보내기·비우기 ──────────────────────────────────────────────────
    def export_zip(self):
        """수집한 사진/라벨을 학습에 넣을 수 있는 ZIP 으로 만들고, 젯슨에 연결돼 있으면 바로 올린다."""
        paths = self.ctx.paths
        if not list(paths.collect_img.glob("*.jpg")):
            QMessageBox.warning(self, "수집 데이터 없음", "먼저 데이터 수집을 진행해주세요.")
            return
        pairs = dataset_io.collected_pairs(paths.collect_img, paths.collect_lbl)
        if not pairs:
            QMessageBox.warning(self, "라벨 없음", "이미지에 대응하는 라벨(.txt)이 없습니다.")
            return

        out = paths.imports / f"collected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        names = [p["class_name"] for _, p in product_db.ordered(self.ctx.products)]
        dataset_io.write_training_zip(out, pairs, names)
        detail = dataset_io.class_count_detail(dataset_io.labels_per_class([l for _, l in pairs]),
                                               self.ctx.products)

        # 젯슨에 바로 올려 준다. 업체가 scp 를 칠 일이 없어야 한다.
        #   replace="collected_" — 이 ZIP 은 수집 폴더 '전체'를 담으므로 옛것이 남으면 같은 사진이 두 번 들어간다.
        sent = ""
        if self.ctx.can_run_on_jetson():
            self.ctx.show_status(f"젯슨으로 전송 중… ({out.stat().st_size // 1024 // 1024} MB)")
            QApplication.processEvents()
            try:
                up = self.ctx.ops.upload_zip(str(out), replace_prefix="collected_")
                sent = f"\n\n젯슨으로 전송 완료: {up['saved']}"
                if up.get("removed"):
                    sent += f"\n(옛 파일 정리: {', '.join(up['removed'])})"
            except (OSError, RuntimeError, ValueError) as e:
                log.warning("ZIP 전송 실패: %s", e)
                sent = f"\n\n[전송 실패] {e}\n젯슨 서버가 켜져 있는지 확인하세요. 파일은 PC 에 남아 있습니다."
            finally:
                self.ctx.clear_status()

        QMessageBox.information(self, "학습용 ZIP 생성 완료",
                                f"{out.name}\n총 {len(pairs)}장\n\n{detail}{sent}\n\n"
                                "다음 단계: 「모델 배포 및 모델 학습」 탭 →\n"
                                "  ① Train / Val 자동 분할  ② 전이학습 시작")

    def clear_data(self):
        """수집한 사진·라벨을 모두 지운다. 이미 내보낸 ZIP 도 지울지 따로 물어본다."""
        paths = self.ctx.paths
        imgs = sorted(paths.collect_img.glob("*.jpg"))
        labs = sorted(paths.collect_lbl.glob("*.txt"))
        if not imgs and not labs:
            QMessageBox.information(self, "비어 있음", "지울 수집 데이터가 없습니다.")
            return

        detail = dataset_io.class_count_detail(dataset_io.labels_per_class(labs), self.ctx.products) \
            or "  (라벨 없음)"
        answer = QMessageBox.question(
            self, "수집 데이터 비우기",
            f"아래 데이터를 모두 지웁니다. 되돌릴 수 없습니다.\n\n"
            f"사진 {len(imgs)}장 · 라벨 {len(labs)}개\n\n{detail}\n\n"
            f"위치: {paths.collect_img.parent}\n\n지울까요?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return

        removed, failed = dataset_io.delete_files(imgs + labs)

        zips = sorted(paths.imports.glob("collected_*.zip"))
        zip_msg = ""
        if zips:
            names = "\n".join(f"  {z.name}  ({z.stat().st_size // 1024 // 1024} MB)" for z in zips)
            ans2 = QMessageBox.question(
                self, "내보낸 ZIP 도 지울까요?",
                f"사진은 지웠지만 이미 내보낸 ZIP 이 남아 있습니다.\n"
                f"이걸 두면 지운 사진이 다음 학습에 그대로 들어갑니다.\n\n{names}\n\n같이 지울까요?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ans2 == QMessageBox.Yes:
                gone, zip_failed = dataset_io.delete_files(zips)
                failed += zip_failed
                if gone:
                    zip_msg = f"\n지운 ZIP: {gone}개"
            else:
                zip_msg = "\nZIP 은 남겨두었습니다."

        self.count = 0
        self.skipped = 0
        self.status.setText("수집 데이터를 비웠습니다.")
        self.ctx.show_status(f"수집 데이터 {removed}개 삭제", 5000)
        self.ctx.products_changed.emit()
        log.info("수집 데이터 삭제: %s개 (실패 %s)", removed, len(failed))

        msg = f"{removed}개 파일을 지웠습니다.{zip_msg}"
        if failed:
            msg += "\n\n지우지 못한 것:\n" + "\n".join(failed[:5])
        QMessageBox.information(self, "삭제 완료", msg)

    def shutdown(self):
        self.timer.stop()
        if self.turn_worker:
            self.turn_worker.stop()
            self.turn_worker.wait(3000)


def _spin(lo, hi, value, suffix, tooltip=None):
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    s.setSuffix(suffix)
    if tooltip:
        s.setToolTip(tooltip)
    return s
