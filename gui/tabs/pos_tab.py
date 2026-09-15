"""POS 탭 — 카메라 화면, 상품 인식, 주문표, 결제."""
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFrame, QGridLayout,
                             QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                             QMessageBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ai.render import draw_detections
from core import products as product_db
from core.cart import Cart, count_by_class, pos_detections
from core.log import get_logger
from gui import style
from gui.camera_view import CameraView
from gui.style import button
from gui.tabs.member_qr import MEMBER_API_ERRORS, MemberQrPanel

log = get_logger("pos")


class PosTab(QWidget):
    def __init__(self, ctx, camera, live, qr_worker, preview_max=0, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.camera = camera
        self.live = live
        self.cart = Cart()

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)
        root.addWidget(self._build_header())

        main = QHBoxLayout()
        main.setSpacing(10)
        main.addWidget(self._build_left(preview_max), 3)
        main.addWidget(self._build_right(qr_worker), 2)
        root.addLayout(main)

        ctx.products_changed.connect(self.refresh_products)
        ctx.model_changed.connect(self.refresh_model_badge)
        camera.started.connect(self._on_camera_started)
        camera.stopped.connect(self._on_camera_stopped)

        self.refresh_camera_combo()
        self.refresh_products()
        self.refresh_model_badge()

    # ── 화면 구성 ────────────────────────────────────────────────────────
    def _build_header(self):
        header = QFrame()
        header.setObjectName("POSHeader")
        lay = QHBoxLayout(header)
        title_box = QVBoxLayout()
        title = QLabel("BreadVision POS | JETSON")
        title.setStyleSheet('font-family:"Jua"; color:#3E2723; font-size:26px;')
        subtitle = QLabel("AI Recognition  ·  Order  ·  Price Calculation  ·  Payment")
        subtitle.setStyleSheet("color:#8D6E63; font-size:12px;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        self.camera_state = QLabel("CAMERA  OFF")
        self.camera_state.setStyleSheet(style.BADGE_OFF)
        self.model_badge = QLabel("MODEL  NOT SET")
        self.model_badge.setStyleSheet(style.BADGE_MODEL_UNSET)

        lay.addLayout(title_box)
        lay.addStretch()
        lay.addWidget(self.camera_state)
        lay.addWidget(self.model_badge)
        return header

    def _build_left(self, preview_max):
        card = QFrame()
        card.setObjectName("Card")
        left = QVBoxLayout(card)

        head = QHBoxLayout()
        cam_title = QLabel("LIVE VIEW / PRODUCT RECOGNITION")
        cam_title.setStyleSheet('font-family:"Jua"; font-size:19px; color:#5D4037;')
        guide = QLabel("트레이를 카메라 영역에 맞춘 뒤 인식을 실행하세요.")
        guide.setObjectName("PageDesc")
        head.addWidget(cam_title)
        head.addStretch()
        head.addWidget(guide)
        left.addLayout(head)

        self.view = CameraView("NO CAMERA SIGNAL\n카메라 시작 버튼을 눌러주세요.", 760, 500,
                               preview_max, style.PREVIEW)
        left.addWidget(self.view, 1)

        self.live_check = QCheckBox("실시간 인식 표시")
        self.live_check.setChecked(True)
        self.live_check.setToolTip(
            "대기 중에도 화면에서 빵을 찾아 초록 박스로 표시합니다.\n"
            "주문에 담는 것은 「상품 인식 실행」 버튼을 눌렀을 때만 이뤄집니다.")

        # 카메라 고르기 — PC 에서 젯슨 영상 대신 내장 카메라가 열리는 일이 있어서
        # 사람이 직접 바꿀 수 있게 한다. 고른 것은 settings.json 에 기억한다.
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(200)
        self.camera_combo.setToolTip(
            "어느 카메라를 쓸지 고릅니다. 고르면 바로 바뀌고 다음 실행에도 기억합니다.\n"
            "· 젯슨 카메라: 젯슨에 꽂힌 USB 카메라 영상(8090)\n"
            "· PC 카메라: 이 컴퓨터의 카메라 (보통 0번이 내장)")
        self.camera_combo.activated.connect(self.on_camera_combo)

        row = QHBoxLayout()
        row.addWidget(self.camera_combo)
        row.addWidget(button("카메라 찾기", self.scan_cameras,
                             tooltip="이 컴퓨터에 연결된 카메라 번호를 찾아 목록에 표시합니다."))
        row.addWidget(button("카메라 시작", self.start_camera, "Dark",
                             "USB 카메라를 연결하고 실시간 화면을 시작합니다."))
        row.addWidget(button("상품 인식 실행", self.recognize, "Warning",
                             "현재 카메라 화면에서 빵을 인식하고 주문 목록에 반영합니다."), 2)
        row.addWidget(button("카메라 종료", self.camera.stop, tooltip="카메라 연결을 종료합니다."))
        row.addWidget(self.live_check)
        left.addLayout(row)

        quick = QGroupBox("MANUAL PRODUCT SELECT")
        self.product_grid = QGridLayout(quick)
        left.addWidget(quick)
        return card

    def _build_right(self, qr_worker):
        card = QFrame()
        card.setObjectName("Card")
        right = QVBoxLayout(card)

        head = QHBoxLayout()
        receipt_title = QLabel("ORDER")
        receipt_title.setStyleSheet('font-family:"Jua"; font-size:21px; color:#5D4037;')
        self.item_count_label = QLabel("총 0개")
        self.item_count_label.setObjectName("ReceiptInfo")
        head.addWidget(receipt_title)
        head.addStretch()
        head.addWidget(self.item_count_label)
        right.addLayout(head)

        self.member_panel = MemberQrPanel(self.ctx, self.camera, self.live, qr_worker)
        right.addWidget(self.member_panel)

        self.cart_table = QTableWidget(0, 5)
        self.cart_table.setHorizontalHeaderLabels(["상품", "단가", "수량", "금액", ""])
        self.cart_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cart_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cart_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cart_table.setAlternatingRowColors(True)
        self.cart_table.verticalHeader().setVisible(False)
        self.cart_table.setMinimumWidth(520)
        self.cart_table.setMinimumHeight(380)
        right.addWidget(self.cart_table, 1)

        qty = QGridLayout()
        qty.addWidget(button("수량 감소", lambda: self.change_selected_qty(-1),
                             tooltip="선택한 상품 수량을 1개 줄입니다."), 0, 0)
        qty.addWidget(button("수량 증가", lambda: self.change_selected_qty(1), "Primary",
                             "선택한 상품 수량을 1개 늘립니다."), 0, 1)
        qty.addWidget(button("선택 항목 삭제", self.delete_selected, "Danger",
                             "선택한 상품을 주문에서 제거합니다."), 1, 0)
        qty.addWidget(button("주문 초기화", self.clear_cart, "Dark",
                             "현재 주문 내역을 전부 초기화합니다."), 1, 1)
        right.addLayout(qty)

        total_box = QFrame()
        total_box.setObjectName("TotalBox")
        total_lay = QVBoxLayout(total_box)
        cap_row = QHBoxLayout()
        cap = QLabel("TOTAL")
        cap.setObjectName("TotalCaption")
        self.total_detail = QLabel("상품 0종 / 수량 0개")
        self.total_detail.setObjectName("ReceiptInfo")
        cap_row.addWidget(cap)
        cap_row.addStretch()
        cap_row.addWidget(self.total_detail)
        self.total_value = QLabel("0원")
        self.total_value.setObjectName("TotalValue")
        self.total_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        total_lay.addLayout(cap_row)
        total_lay.addWidget(self.total_value)
        right.addWidget(total_box)

        right.addWidget(button("결제 처리", self.complete_payment, "Success",
                               "현재 주문 금액을 결제 완료 처리하고 새 주문 상태로 초기화합니다."))
        help_label = QLabel("Workflow  |  Camera → Recognition → Order Check → Total → Payment")
        help_label.setStyleSheet("color:#8D6E63; padding:6px 2px; font-size:12px;")
        help_label.setWordWrap(True)
        right.addWidget(help_label)
        return card

    # ── 카메라 ──────────────────────────────────────────────────────────
    def start_camera(self):
        self.camera.start_with_message(self)

    def _on_camera_started(self, label):
        self.camera_state.setText(f"CAMERA  ON · {label[:24]}")
        self.camera_state.setStyleSheet(style.BADGE_ON)

    def _on_camera_stopped(self):
        self.member_panel.on_camera_stopped()
        self.view.reset_size()
        self.camera_state.setText("CAMERA  OFF")
        self.camera_state.setStyleSheet(style.BADGE_OFF)

    def refresh_camera_combo(self, found=None):
        """found: {번호: "1280x720"} — 「카메라 찾기」 결과. None 이면 0~2번을 그냥 보여준다."""
        combo = self.camera_combo
        combo.blockSignals(True)
        combo.clear()
        if self.camera.stream_url():
            combo.addItem("📷 젯슨 카메라 (8090)", "jetson")
        indices = found if found is not None else {i: "" for i in range(3)}
        cur = self.camera.current_key()
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
                text=str(self.camera.camera.index))
            if not ok or not text.strip():
                self.refresh_camera_combo()
                return
            key = text.strip()
        self.camera.switch(key)
        self.refresh_camera_combo()
        self.start_camera()                      # 고르면 바로 켜서 보여준다

    def scan_cameras(self):
        found, was_local = self.camera.scan_local()
        self.refresh_camera_combo(found)
        if was_local:
            self.start_camera()
        if found:
            lines = "\n".join(f"  · {i}번  {s}" for i, s in sorted(found.items()))
            msg = (f"이 컴퓨터에서 찾은 카메라:\n{lines}\n\n"
                   "보통 0번이 노트북 내장 카메라, 1번이 USB 카메라입니다.\n"
                   "카메라 목록에서 골라 주세요.")
        else:
            msg = ("이 컴퓨터에서 쓸 수 있는 카메라를 찾지 못했습니다.\n"
                   "(젯슨 카메라는 목록의 「젯슨 카메라」를 고르세요)")
        if self.camera.stream_url():
            msg += f"\n\n젯슨 카메라 주소: {self.camera.stream_url()}"
        QMessageBox.information(self, "카메라 찾기", msg)

    # ── 프레임 ──────────────────────────────────────────────────────────
    def on_new_frame(self, frame, now, visible):
        self.member_panel.on_new_frame(frame, now, visible)

    def render(self, frame, now):
        # 스캔 모드에서는 빵을 찾지도 그리지도 않는다.
        if not self.member_panel.scanning and self.live_check.isChecked():
            self.live.update(frame, self.camera.last_frame_id, now)
        else:
            self.live.clear()
        view = self.member_panel.overlay(frame)
        shown = pos_detections(self.live.dets)
        if shown:
            if view is frame:                    # 원본에 그리지 않도록 한 번만 복사
                view = frame.copy()
            draw_detections(view, shown, labels=self.ctx.class_names(), inplace=True)
        self.view.show_frame(view, guide=False)

    # ── 상품 인식 ────────────────────────────────────────────────────────
    def recognize(self):
        """「상품 인식 실행」 — 지금 화면의 빵을 인식해 주문에 더한다."""
        if self.member_panel.scanning:
            self.member_panel.stop_scan()        # 빵을 인식하겠다고 눌렀으니 QR 스캔은 그만둔다
        detector = self.ctx.detector
        if not detector.ready():
            engine = self.ctx.active_engine()
            if not engine:
                QMessageBox.information(self, "POS 추론 모델 미설정",
                                        "모델 학습 탭에서 Jetson용 TensorRT .engine 파일을 먼저 선택해주세요.")
                return
            try:
                detector.load(engine)
            except Exception as e:               # noqa: BLE001 - TensorRT·네트워크 오류 종류가 다양하다
                log.exception("ENGINE 로드 실패: %s", engine)
                QMessageBox.critical(self, "ENGINE 로드 실패", str(e))
                return

        frame = self.camera.frame
        if frame is None:
            QMessageBox.information(self, "카메라 프레임 없음",
                                    "카메라를 시작한 뒤 트레이에 빵을 올리고 다시 인식해주세요.")
            return

        try:
            # 한 번만 보므로 좌우반전 TTA 로 한 번 더 본다 (19 → 39ms).
            # 실측: 원본에서 rotus 0.22 로 잘못 잡히던 휘낭시에가 fruit 0.76 으로 통과.
            _plotted, detections, _counts = detector.infer(frame.copy(), draw=False, tta=True)
        except Exception as e:                   # noqa: BLE001
            log.exception("YOLO 추론 실패")
            QMessageBox.critical(self, "YOLO 추론 실패", str(e))
            return

        detections = pos_detections(detections)  # 확실한 것만 주문에 담는다
        plotted, _ = draw_detections(frame.copy(), detections, labels=self.ctx.class_names())
        self.view.show_scaled(plotted)

        added, unknown = self.cart.add_counts(count_by_class(detections), self.ctx.products)
        self.refresh_cart_table()
        if added:
            names = ", ".join(f"{product_db.display_name(self.ctx.products, k)} +{v}"
                              for k, v in sorted(added.items()))
            self.ctx.show_status(f"주문 추가: {names}", 4000)
        else:
            self.ctx.show_status("인식된 상품이 없습니다.", 4000)
        if unknown:
            QMessageBox.warning(self, "등록되지 않은 클래스",
                                "YOLO에서 검출했지만 products.json에 없는 Class ID: "
                                + ", ".join(map(str, unknown)))

    # ── 주문표 ──────────────────────────────────────────────────────────
    def refresh_products(self):
        while self.product_grid.count():
            widget = self.product_grid.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        for i, (cid, p) in enumerate(product_db.ordered(self.ctx.products)):
            btn = button(f"{p['display_name']}   |   {int(p['price']):,}원",
                         lambda checked=False, class_id=int(cid): self.add_to_cart(class_id),
                         "ProductButton", f"{p['display_name']} 1개를 주문 목록에 추가합니다.")
            self.product_grid.addWidget(btn, i // 2, i % 2)
        self.refresh_cart_table()

    def add_to_cart(self, class_id):
        self.cart.add(class_id)
        self.refresh_cart_table()

    def refresh_cart_table(self):
        lines = self.cart.lines(self.ctx.products)
        self.cart_table.setRowCount(len(lines))
        for r, (cid, product, qty, subtotal) in enumerate(lines):
            name_item = QTableWidgetItem(product["display_name"])
            name_item.setData(Qt.UserRole, cid)     # 이름이 같은 상품이 있어도 헷갈리지 않게
            self.cart_table.setItem(r, 0, name_item)
            self.cart_table.setItem(r, 1, QTableWidgetItem(f"{int(product['price']):,}원"))
            self.cart_table.setItem(r, 2, QTableWidgetItem(str(qty)))
            self.cart_table.setItem(r, 3, QTableWidgetItem(f"{subtotal:,}원"))
            self.cart_table.setCellWidget(r, 4, button(
                "삭제", lambda checked=False, class_id=cid: self.remove_cart_item(class_id), "Danger",
                f"{product['display_name']}을 주문에서 제거합니다."))

        self.total_value.setText(f"{self.cart.total(self.ctx.products):,}원")
        self.item_count_label.setText(f"총 {self.cart.total_qty()}개")
        self.total_detail.setText(f"상품 {self.cart.kinds()}종 / 수량 {self.cart.total_qty()}개")

    def remove_cart_item(self, class_id):
        self.cart.remove(class_id)
        self.refresh_cart_table()

    def _selected_class_id(self, message):
        row = self.cart_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "상품 선택", message)
            return None
        return self.cart_table.item(row, 0).data(Qt.UserRole)

    def delete_selected(self):
        cid = self._selected_class_id("삭제할 상품을 주문 목록에서 먼저 선택해주세요.")
        if cid is not None:
            self.remove_cart_item(cid)

    def change_selected_qty(self, delta):
        cid = self._selected_class_id("주문 목록에서 상품을 먼저 선택해주세요.")
        if cid is not None:
            self.cart.change(cid, delta)
            self.refresh_cart_table()

    def clear_cart(self):
        if self.cart:
            self.cart.clear()
            self.refresh_cart_table()

    def refresh_model_badge(self):
        engine = self.ctx.active_engine()
        if engine:
            self.model_badge.setText(f"ENGINE  {Path(engine).name}")
            self.model_badge.setStyleSheet(style.BADGE_MODEL_SET)
        else:
            self.model_badge.setText("ENGINE  NOT SET")

    # ── 결제 ────────────────────────────────────────────────────────────
    def complete_payment(self):
        if not self.cart:
            QMessageBox.information(self, "주문 없음", "결제할 상품이 없습니다.")
            return

        total = self.cart.total(self.ctx.products)
        member = self.member_panel.member
        extra = ""
        if member:
            try:
                result = self.ctx.membership_api.complete_payment(member["id"], total)
            except MEMBER_API_ERRORS as e:
                # 예전에는 여기서 오류가 나면 프로그램이 꺼졌다. 주문은 그대로 두고 다시 누를 수 있게 한다.
                log.warning("포인트 적립 실패 (회원 %s, %s원): %s", member.get("id"), total, e)
                QMessageBox.critical(self, "포인트 적립 실패",
                                     f"회원 포인트를 적립하지 못했습니다.\n\n{e}\n\n"
                                     "주문은 그대로 남아 있습니다. 인터넷 연결을 확인한 뒤 다시 「결제 처리」를 누르거나, "
                                     "「회원 해제」 후 결제하세요.")
                return
            if result:
                extra = (f"\n\n회원: {member['name']}"
                         f"\n적립: +{result['earned_point']:,}P"
                         f"\n보유 포인트: {result['current_point']:,}P")

        QMessageBox.information(self, "결제 완료", f"결제가 완료되었습니다.\n총 금액: {total:,}원{extra}")
        log.info("결제 완료: %s원%s", total, f" (회원 {member.get('id')})" if member else "")
        self.cart.clear()
        self.member_panel.clear_member()
        self.refresh_cart_table()
