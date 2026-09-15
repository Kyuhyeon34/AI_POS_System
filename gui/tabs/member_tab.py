"""회원 관리 탭 — 회원 데이터가 어디에 있고 어떻게 연결되는지 안내한다."""
from PyQt5.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

from gui.style import page_header


class MemberTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)
        page_header(root, "회원 관리", "중앙 DB: Supabase   |   회원 앱: Android APK")

        flow = QGroupBox("시연 흐름")
        QVBoxLayout(flow).addWidget(QLabel(
            "Android APK 회원가입/로그인 → 개인 QR 표시 → POS 회원 QR 스캔 "
            "→ 결제 → 결제금액의 3% 포인트 적립"))
        root.addWidget(flow)

        db = QGroupBox("중앙 DB 상태")
        QVBoxLayout(db).addWidget(QLabel(
            "회원정보와 포인트는 Jetson 로컬 DB가 아니라 Supabase 중앙 DB에 저장됩니다."))
        root.addWidget(db)
        root.addStretch()
