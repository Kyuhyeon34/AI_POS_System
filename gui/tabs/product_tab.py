"""상품 관리 탭 — 판매할 빵 등록, 가격 변경."""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QMessageBox, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from core import products as product_db
from core.dataset_io import count_collected
from core.log import get_logger
from gui.style import button, page_header

log = get_logger("products")


class ProductTab(QWidget):
    product_registered = pyqtSignal(int)      # 새로 배정한 class_id

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        page_header(root, "상품 및 가격 관리",
                    "판매할 빵을 등록하고 가격을 관리합니다. 신규 등록 시 YOLO Class ID가 자동 배정됩니다.")

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Class ID", "YOLO Class Name", "화면 표시 상품명", "판매 가격", "수집 이미지"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        root.addWidget(self.table)

        group = QGroupBox("신규 상품 추가")
        form = QFormLayout(group)
        self.display_edit = QLineEdit()
        self.display_edit.setPlaceholderText("예: 크림빵")
        self.class_edit = QLineEdit()
        self.class_edit.setPlaceholderText("예: cream_bread")
        self.price_spin = QSpinBox()
        self.price_spin.setRange(0, 1000000)
        self.price_spin.setSingleStep(100)
        self.price_spin.setSuffix(" 원")
        self.next_id_label = QLabel("-")
        self.next_id_label.setStyleSheet("font-weight:800; color:#8B4513;")
        form.addRow("상품명", self.display_edit)
        form.addRow("YOLO Class Name", self.class_edit)
        form.addRow("판매 가격", self.price_spin)
        form.addRow("자동 배정 Class ID", self.next_id_label)
        root.addWidget(group)

        buttons = QHBoxLayout()
        buttons.addWidget(button(
            "＋ 신규 상품 등록", self.register_product, "Primary",
            "상품명·Class Name·가격을 저장하고 Class ID를 자동 배정합니다. "
            "등록 즉시 POS와 데이터 수집 탭에도 반영됩니다."))
        buttons.addWidget(button(
            "₩ 선택 상품 가격 변경", self.update_selected_price,
            tooltip="위 표에서 선택한 상품의 가격을 현재 입력된 판매 가격으로 변경합니다."))
        root.addLayout(buttons)

        how = QLabel("버튼 설명  |  신규 상품 등록: 상품 마스터에 새 품목 추가  |  "
                     "선택 상품 가격 변경: 표에서 고른 품목의 판매가 수정")
        how.setObjectName("PageDesc")
        root.addWidget(how)

        ctx.products_changed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        rows = product_db.ordered(self.ctx.products)
        self.table.setRowCount(len(rows))
        for r, (cid, p) in enumerate(rows):
            values = [cid, p["class_name"], p["display_name"], f"{int(p['price']):,}원",
                      f"{count_collected(self.ctx.paths.collect_img, p['class_name'])}장"]
            for c, value in enumerate(values):
                self.table.setItem(r, c, QTableWidgetItem(value))
        self.next_id_label.setText(str(product_db.next_class_id(self.ctx.products)))

    def register_product(self):
        display = self.display_edit.text()
        price = self.price_spin.value()
        try:
            cid = product_db.add_product(self.ctx.products, display, self.class_edit.text(), price)
        except ValueError as e:
            title = "중복" if "이미" in str(e) else "입력 오류"
            QMessageBox.warning(self, title, str(e))
            return
        self.ctx.save_products()
        self.ctx.paths.collect_img.mkdir(parents=True, exist_ok=True)
        self.ctx.paths.collect_lbl.mkdir(parents=True, exist_ok=True)
        self.ctx.write_data_yaml()
        log.info("상품 등록: %s (class %s, %s원)", display.strip(), cid, price)

        self.display_edit.clear()
        self.class_edit.clear()
        self.price_spin.setValue(0)

        self.ctx.products_changed.emit()
        self.product_registered.emit(cid)
        QMessageBox.information(
            self, "상품 등록 완료",
            f"{display.strip()}이 등록되었습니다.\nClass ID: {cid}\n판매 가격: {price:,}원\n\n"
            "POS / 데이터 수집 / 모델 학습 목록에 함께 반영되었습니다.")

    def update_selected_price(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "상품 선택", "가격을 바꿀 상품을 표에서 먼저 선택해주세요.")
            return
        cid = self.table.item(row, 0).text()
        old = int(self.ctx.products[cid]["price"])
        new = self.price_spin.value()
        self.ctx.products[cid]["price"] = new
        self.ctx.save_products()
        self.ctx.products_changed.emit()
        QMessageBox.information(self, "가격 변경 완료",
                                f"{self.ctx.products[cid]['display_name']}\n{old:,}원 → {new:,}원")
