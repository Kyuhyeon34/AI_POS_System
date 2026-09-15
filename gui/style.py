"""화면 스타일 — 회원 앱(BreadVision Member)과 같은 따뜻한 빵집 톤.

    배경 #FBF1E3 크림 · 카드 #F3E1C7 · 포인트 #B5651D 캐러멜 · 글자 #3E2723
    제목·버튼·큰 숫자는 Jua, 표·입력·작은 글자는 맑은 고딕
"""
from pathlib import Path

from PyQt5.QtGui import QFontDatabase

from core.log import get_logger

log = get_logger("style")

FONT_DIR = Path(__file__).resolve().parent / "fonts"

# 상태 배지 (카메라 켜짐/꺼짐, 모델)
BADGE_OFF = ("background:#F3E1C7; color:#8D6E63; padding:8px 12px; "
             "border:1px solid #E3CBA7; border-radius:10px; font-weight:800;")
BADGE_ON = ("background:#E6F2DF; color:#2E7D32; padding:8px 12px; "
            "border:1px solid #B9D8AC; border-radius:10px; font-weight:800;")
BADGE_MODEL_UNSET = ("background:#F3E1C7; color:#B5651D; padding:8px 12px; "
                     "border:1px solid #D9A66C; border-radius:10px; font-weight:800;")
BADGE_MODEL_SET = ("background:#F6E3C8; color:#8B4513; padding:8px 12px; "
                   "border:1px solid #D9A66C; border-radius:10px; font-weight:800;")

PREVIEW = "background:#2B211C; color:#FFF8ED; border-radius:14px; font-size:17px; font-weight:700;"
NOTE_BOX = "background:#F6E3C8; border:1px solid #E3CBA7; padding:12px; border-radius:12px; color:#5D4037;"
STATUS_BOX = "background:#FFFBF4; border:1px solid #E3CBA7; padding:10px; border-radius:10px; color:#3E2723;"
OK_BOX = "background:#E6F2DF; border:1px solid #B9D8AC; padding:9px; border-radius:10px; color:#2E7D32;"
ERROR_BOX = "background:#FBE3DF; border:1px solid #E8B1A8; padding:9px; border-radius:10px; color:#B3261E;"
PHASE_BOX = ("background:#F6E3C8; border:1px solid #D9A66C; padding:10px; "
             "border-radius:10px; color:#8B4513; font-weight:700;")


def load_fonts():
    """Jua 를 앱 전용 폰트로 등록한다. 위젯에 스타일을 붙이기 전에 불러야 적용된다.
    실패하면 기본 한글 폰트로 보인다."""
    path = FONT_DIR / "Jua-Regular.ttf"
    if path.exists() and QFontDatabase.addApplicationFont(str(path)) < 0:
        log.warning("폰트 등록 실패: %s", path)


STYLESHEET = """
    QMainWindow, QDialog, QMessageBox, QInputDialog, QFileDialog {
        background: #FBF1E3;
    }

    QWidget {
        color: #3E2723;
        font-size: 14px;
        font-family: "Malgun Gothic", "Noto Sans CJK KR", "NanumGothic", sans-serif;
    }

    QLabel, QCheckBox, QRadioButton {
        background: transparent;
    }

    QToolTip {
        background: #3E2723;
        color: #FFF8ED;
        border: none;
        padding: 6px 8px;
    }

    QTabWidget::pane {
        border: none;
        border-top: 2px solid #E3CBA7;
        background: #FBF1E3;
        top: -2px;
    }

    QTabBar::tab {
        background: #F3E1C7;
        color: #8D6E63;
        border: 1px solid #E3CBA7;
        border-bottom: none;
        border-top-left-radius: 12px;
        border-top-right-radius: 12px;
        padding: 12px 24px;
        margin-right: 4px;
        min-width: 150px;
        font-family: "Jua";
        font-size: 16px;
    }

    QTabBar::tab:hover {
        background: #EED6B4;
        color: #5D4037;
    }

    QTabBar::tab:selected {
        background: #FBF1E3;
        color: #8B4513;
        border: 2px solid #E3CBA7;
        border-bottom: 2px solid #FBF1E3;
        border-top: 3px solid #B5651D;
    }

    QLabel#PageTitle {
        font-family: "Jua";
        font-size: 26px;
        color: #3E2723;
    }

    QLabel#PageDesc {
        color: #8D6E63;
        font-size: 13px;
    }

    QLabel#TotalCaption {
        font-family: "Jua";
        font-size: 17px;
        color: #8D6E63;
    }

    QLabel#TotalValue {
        font-family: "Jua";
        color: #B5651D;
        font-size: 48px;
    }

    QLabel#ReceiptInfo {
        color: #8D6E63;
        font-size: 13px;
        font-weight: 700;
    }

    QGroupBox {
        background: #F3E1C7;
        border: 1px solid #E3CBA7;
        border-radius: 16px;
        margin-top: 18px;
        padding: 14px;
        font-family: "Jua";
        font-size: 15px;
        color: #5D4037;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        left: 14px;
        padding: 0 6px;
        color: #8B4513;
    }

    QPushButton {
        background: #FFFBF4;
        color: #5D4037;
        border: 1px solid #D9B98E;
        border-radius: 12px;
        padding: 10px 14px;
        font-family: "Jua";
        font-size: 15px;
    }

    QPushButton:hover {
        background: #F6E3C8;
        border-color: #B5651D;
    }

    QPushButton:pressed {
        background: #EED6B4;
    }

    QPushButton:disabled {
        background: #F3E6D4;
        color: #BCAAA4;
        border-color: #E8D5BA;
    }

    QPushButton#Primary {
        background: #B5651D;
        color: #FFF8ED;
        border: 1px solid #A0561A;
    }

    QPushButton#Primary:hover {
        background: #C47329;
    }

    QPushButton#Primary:pressed {
        background: #8B4513;
    }

    QPushButton#Success {
        background: #8B4513;
        color: #FFF8ED;
        border: 1px solid #7A3C10;
        border-radius: 14px;
        font-size: 20px;
        padding: 16px;
    }

    QPushButton#Success:hover {
        background: #A0561A;
    }

    QPushButton#Success:pressed {
        background: #6D360F;
    }

    QPushButton#Danger {
        background: #FFF4F1;
        color: #B3261E;
        border: 1px solid #E8B1A8;
    }

    QPushButton#Danger:hover {
        background: #FBE3DF;
        border-color: #C9574B;
    }

    QPushButton#Warning {
        background: #D9A66C;
        color: #3E2723;
        border: 1px solid #C48F52;
    }

    QPushButton#Warning:hover {
        background: #E3B47E;
    }

    QPushButton#Dark {
        background: #F3E1C7;
        color: #5D4037;
        border: 1px solid #D9B98E;
    }

    QPushButton#Dark:hover {
        background: #EED6B4;
    }

    QPushButton#ProductButton {
        background: #FFFBF4;
        color: #3E2723;
        border: 1px solid #E3CBA7;
        border-radius: 14px;
        min-height: 64px;
        font-size: 16px;
        text-align: left;
        padding-left: 14px;
    }

    QPushButton#ProductButton:hover {
        background: #F6E3C8;
        border: 2px solid #B5651D;
        color: #8B4513;
    }

    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background: #FFFBF4;
        color: #3E2723;
        border: 1px solid #D9B98E;
        border-radius: 10px;
        padding: 8px;
        min-height: 24px;
        selection-background-color: #EBCB9F;
        selection-color: #3E2723;
    }

    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
        border: 2px solid #B5651D;
    }

    QComboBox QAbstractItemView {
        background: #FFFBF4;
        color: #3E2723;
        selection-background-color: #EBCB9F;
        selection-color: #3E2723;
        border: 1px solid #D9B98E;
    }

    QCheckBox {
        spacing: 8px;
    }

    QTableWidget {
        background: #FFFBF4;
        alternate-background-color: #F8EBD8;
        gridline-color: #EAD7BC;
        selection-background-color: #EBCB9F;
        selection-color: #3E2723;
        border: 1px solid #E3CBA7;
        border-radius: 10px;
        font-size: 14px;
    }

    QHeaderView {
        background: #F3E1C7;
        border: none;
    }

    QHeaderView::section {
        background: #F3E1C7;
        color: #5D4037;
        padding: 9px;
        border: none;
        border-right: 1px solid #E3CBA7;
        border-bottom: 1px solid #E3CBA7;
        font-weight: 700;
    }

    QTableCornerButton::section {
        background: #F3E1C7;
        border: none;
    }

    QPlainTextEdit {
        background: #FFFBF4;
        color: #4E342E;
        border: 1px solid #E3CBA7;
        border-radius: 10px;
        font-family: Consolas, "D2Coding", monospace;
    }

    QProgressBar {
        background: #F3E1C7;
        border: 1px solid #E3CBA7;
        border-radius: 8px;
        text-align: center;
        min-height: 22px;
        color: #3E2723;
    }

    QProgressBar::chunk {
        background: #B5651D;
        border-radius: 7px;
    }

    QFrame#Card {
        background: #F3E1C7;
        border: 1px solid #E3CBA7;
        border-radius: 20px;
    }

    QFrame#POSHeader {
        background: #FFF8ED;
        border: 1px solid #E3CBA7;
        border-radius: 20px;
    }

    QFrame#TotalBox {
        background: #FFF8ED;
        border: 2px solid #D9A66C;
        border-radius: 16px;
    }

    QScrollBar:vertical {
        background: transparent;
        width: 11px;
        margin: 0;
    }

    QScrollBar::handle:vertical {
        background: #D9B98E;
        min-height: 25px;
        border-radius: 5px;
    }

    QScrollBar::handle:vertical:hover {
        background: #B5651D;
    }

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0;
    }
"""


def page_header(layout, title, desc):
    """탭 맨 위 제목·설명 두 줄."""
    from PyQt5.QtWidgets import QLabel
    t = QLabel(title)
    t.setObjectName("PageTitle")
    d = QLabel(desc)
    d.setObjectName("PageDesc")
    layout.addWidget(t)
    layout.addWidget(d)
    return t, d


def button(text, slot=None, name=None, tooltip=None):
    """스타일 이름·툴팁·클릭 연결을 한 줄로."""
    from PyQt5.QtWidgets import QPushButton
    b = QPushButton(text)
    if name:
        b.setObjectName(name)
    if tooltip:
        b.setToolTip(tooltip)
    if slot is not None:
        b.clicked.connect(slot)
    return b
