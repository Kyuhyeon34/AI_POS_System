"""POS 탭 오른쪽의 「MEMBERSHIP / QR」 — 회원 QR 스캔과 회원 표시.

「회원 QR 스캔」 을 누르면 실시간 빵 인식을 멈추고 QR 만 찾는다.
회원 QR 이 잡히면 버튼을 다시 누를 필요 없이 바로 조회해 회원으로 등록한다.
적립은 「결제 처리」 때 결제 금액 기준으로 들어간다 (QR 을 먼저 대고 빵을 나중에 담아도 금액이 맞게).

인증 뒤에도 휴대폰이 화면에 있는 동안은 스캔 모드를 유지한다.
바로 빵 인식으로 돌아가면 앱 화면(주황 버튼)을 rotus 로 잡기 때문이다.
QR 이 QR_EXIT_AFTER_SEC 동안 안 보이면 그때 끝낸다.
"""
import time

import cv2
import requests
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QMessageBox, QVBoxLayout

from core.log import get_logger
from gui.style import button
from membership.config import QR_PREFIX

log = get_logger("member")

# 회원 조회가 실패하는 경우: 인터넷 끊김(RequestException), Supabase 오류 응답(RuntimeError),
# 응답이 JSON 이 아님(ValueError)
MEMBER_API_ERRORS = (requests.RequestException, RuntimeError, ValueError)

_GREEN = "color:#2E7D32; font-size:12px; font-weight:900;"
_AMBER = "color:#9A6700; font-size:12px; font-weight:900;"
LABEL_STYLE = {
    "scan": ("● QR 스캔 중 — 휴대폰 회원 QR을 카메라에 대 주세요 (빵 인식 멈춤)", _GREEN),
    "member": ("● 회원 QR 인식 — 회원 확인 중…", _GREEN),
    "done": ("● 회원 인증 완료 — 휴대폰을 치우면 2초 뒤 빵 인식으로 돌아갑니다", _GREEN),
    "other": ("● BreadVision 회원 QR이 아닙니다 — 앱의 QR을 보여주세요", _AMBER),
    "notfound": ("● 등록되지 않은 회원 QR입니다 — 앱에서 다시 로그인해 주세요", _AMBER),
    "error": ("● 회원 조회 실패 — 인터넷 연결을 확인하세요",
              "color:#C62828; font-size:12px; font-weight:900;"),
    "none": ("「회원 QR 스캔」을 누르고 휴대폰 QR을 카메라에 대 주세요",
             "color:#8D6E63; font-size:12px; font-weight:700;"),
}


class MemberQrPanel(QGroupBox):
    member_changed = pyqtSignal()

    SCAN_EVERY = 2            # 스캔 모드에서는 2프레임마다 검사 — 대자마자 읽히게
    SCAN_TIMEOUT_SEC = 30.0   # 스캔을 켜 두고 이만큼 QR 이 없으면 저절로 끈다
    RETRY_SEC = 3.0           # 조회에 실패한 같은 QR 은 이 시간 동안 다시 조회하지 않는다
    EXIT_AFTER_SEC = 2.0      # 회원 인증 뒤 QR 이 이만큼 안 보이면 스캔을 끝낸다
    HOLD_SEC = 1.0            # 마지막으로 읽힌 뒤 이 시간이 지나면 QR 표시를 지운다

    def __init__(self, ctx, camera, live, qr_worker, parent=None):
        super().__init__("MEMBERSHIP / QR", parent)
        self.ctx = ctx
        self.camera = camera
        self.live = live
        self.qr_worker = qr_worker

        self.member = None
        self.scanning = False
        self.scan_started = 0.0
        self.member_done = False      # 회원 인증은 끝났고 휴대폰을 치우기를 기다리는 중인가
        self.failed = ("", 0.0)       # (조회에 실패한 QR 값, 시각) — 같은 QR 연타 조회 방지
        self.last_data = ""
        self.last_points = None
        self.seen_ts = 0.0
        self.frame_tick = 0
        self.state = None

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.status_label = QLabel("회원 미인증")
        self.status_label.setStyleSheet('font-family:"Jua"; font-size:17px; color:#5D4037;')
        self.point_label = QLabel("")
        self.point_label.setObjectName("ReceiptInfo")
        top.addWidget(self.status_label)
        top.addStretch()
        top.addWidget(self.point_label)
        lay.addLayout(top)

        self.live_label = QLabel()
        lay.addWidget(self.live_label)
        self._set_state("none")

        btns = QHBoxLayout()
        self.scan_btn = button(
            "회원 QR 스캔", self.toggle_scan, "Primary",
            "누르면 빵 인식을 멈추고 QR 만 찾습니다.\n"
            "휴대폰 멤버십 QR 을 카메라에 대면 자동으로 회원이 등록됩니다. 다시 누르면 취소.")
        btns.addWidget(self.scan_btn, 2)
        btns.addWidget(button("회원 해제", self.clear_member, "Dark"))
        lay.addLayout(btns)

        qr_worker.result.connect(self.on_qr_result)

    # ── 스캔 모드 ────────────────────────────────────────────────────────
    def toggle_scan(self):
        if self.scanning:
            self.stop_scan("QR 스캔을 끝냈습니다." if self.member_done else "QR 스캔을 취소했습니다.")
            return
        if self.camera.frame is None or not self.camera.is_running():
            QMessageBox.information(self, "카메라 필요",
                                    "카메라를 먼저 시작한 뒤 「회원 QR 스캔」을 눌러주세요.")
            return
        self.scanning = True
        self.scan_started = time.perf_counter()
        self.failed = ("", 0.0)
        self.member_done = False
        self.frame_tick = self.SCAN_EVERY          # 다음 프레임부터 바로 검사
        self.live.clear()                          # 화면의 빵 박스를 바로 지운다
        self.scan_btn.setText("QR 스캔 중… (누르면 취소)")
        self._set_state("scan")

    def stop_scan(self, message=""):
        """스캔 모드를 끝내고 빵 인식 화면으로 돌아간다."""
        self.scanning = False
        self.member_done = False
        self.last_data = ""
        self.last_points = None
        self.scan_btn.setText("회원 QR 스캔")
        self._set_state("none")
        if message:
            self.ctx.show_status(message, 5000)

    def on_camera_stopped(self):
        self.stop_scan()
        self.seen_ts = 0.0

    def on_new_frame(self, frame, now, pos_tab_visible):
        """새 프레임마다. QR 디코딩(약 27ms)은 전담 스레드에 던지기만 한다.

        POS 탭을 떠나거나 SCAN_TIMEOUT_SEC 가 지나면 스캔을 끝낸다.
        인증 뒤에는 30초 제한을 걸지 않는다 — 휴대폰을 대고 있는 동안 빵 인식이 켜지면 안 된다.
        """
        if self.scanning:
            if not pos_tab_visible:
                self.stop_scan()
            elif self.member_done and now - self.seen_ts > self.EXIT_AFTER_SEC:
                self.stop_scan("휴대폰을 치웠습니다 — 빵 인식으로 돌아갑니다.")
            elif not self.member_done and now - self.scan_started > self.SCAN_TIMEOUT_SEC:
                self.stop_scan("QR 을 찾지 못해 스캔을 끝냈습니다. 다시 누르면 스캔합니다.")
            else:
                self.frame_tick += 1
                if self.frame_tick >= self.SCAN_EVERY and not self.qr_worker.busy:
                    self.frame_tick = 0
                    self.qr_worker.submit(frame.copy())
        # QR 검사 결과가 안 오는 동안(다른 탭, 워커 지연)에도 시간이 지나면 지운다.
        self._expire(now)

    def on_qr_result(self, data, points):
        """QR 스레드가 결과를 보내왔다. (GUI 스레드)

        한 번 읽힌 QR 은 HOLD_SEC 동안 유지한다 — 손이 흔들려 한두 번 놓쳐도 테두리가 깜빡이지 않게.
        유지 시간은 결과 개수가 아니라 시각으로 잰다. 개수로 세면 fps 에 따라 수십 초씩 늘어난다.
        """
        if not self.scanning:
            return                                  # 스캔을 끝낸 뒤에 늦게 도착한 결과
        if data:
            self.last_data = data
            self.last_points = points
            self.seen_ts = time.perf_counter()
            if not self.member_done:                # 인증 뒤에는 '아직 보인다' 만 기록
                self._lookup_member(data)
            return
        self._expire(time.perf_counter())

    def _expire(self, now):
        if not self.last_data or now - self.seen_ts <= self.HOLD_SEC:
            return
        self.last_data = ""
        self.last_points = None
        if not self.scanning:
            self._set_state("none")
        else:
            self._set_state("done" if self.member_done else "scan")

    def _lookup_member(self, data):
        if not data.startswith(QR_PREFIX):
            self._set_state("other")
            return
        if data == self.failed[0] and time.perf_counter() - self.failed[1] < self.RETRY_SEC:
            return      # 방금 조회에 실패한 QR 을 그대로 들고 있다. Supabase 를 연타하지 않는다.

        self._set_state("member")
        try:
            member = self.ctx.membership_api.find_member_by_qr_payload(data)
        except MEMBER_API_ERRORS as e:
            log.warning("회원 조회 실패: %s", e)
            self.failed = (data, time.perf_counter())
            self._set_state("error")
            self.ctx.show_status(f"Supabase 회원 조회 오류: {e}", 8000)
            return
        if not member:
            self.failed = (data, time.perf_counter())
            self._set_state("notfound")
            return

        self.member = member
        self.refresh_member_status()
        # 바로 끝내지 않는다. 휴대폰을 치울 때까지 빵 인식을 멈춘 채로 둔다.
        # 조회에 시간이 걸렸을 수 있으니 2초는 인증이 끝난 지금부터 센다.
        self.member_done = True
        self.seen_ts = time.perf_counter()
        self.scan_btn.setText("회원 인증됨 — 휴대폰을 치워 주세요")
        self._set_state("done")
        self.ctx.show_status(f"회원 인증 완료 — {member['name']}님 · 보유 {int(member['point']):,}P "
                             "(결제 완료 때 적립됩니다)", 6000)

    # ── 회원 표시 ────────────────────────────────────────────────────────
    def refresh_member_status(self):
        if not self.member:
            self.status_label.setText("회원 미인증")
            self.point_label.setText("")
        else:
            self.status_label.setText(f"회원: {self.member['name']}")
            self.point_label.setText(f"{int(self.member['point']):,}P")
        self.member_changed.emit()

    def clear_member(self):
        self.member = None
        self.refresh_member_status()

    # ── 화면 ────────────────────────────────────────────────────────────
    def overlay(self, frame):
        """QR 테두리와 스캔 안내를 그린다. 그릴 게 없으면 원본을 그대로 돌려준다 (복사 안 함)."""
        view = frame
        if self.last_points is not None:
            view = frame.copy()
            try:
                pts = self.last_points.reshape(-1, 2).astype(int)
                if len(pts) >= 4:
                    for i in range(len(pts)):
                        cv2.line(view, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 4)
                    x = int(min(p[0] for p in pts))
                    y = int(min(p[1] for p in pts))
                    label = "MEMBER QR" if self.last_data.startswith(QR_PREFIX) else "QR DETECTED"
                    cv2.putText(view, label, (max(10, x), max(35, y - 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            except (cv2.error, ValueError, AttributeError) as e:
                log.debug("QR 테두리를 그리지 못했습니다: %s", e)
        if self.scanning:
            if view is frame:
                view = frame.copy()
            for color, thick in (((0, 0, 0), 5), ((0, 200, 0), 2)):
                cv2.putText(view, "QR SCAN - show member QR", (18, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thick, cv2.LINE_AA)
        return view

    def _set_state(self, state):
        """안내 라벨 갱신. setStyleSheet 는 다시 그리기를 유발하므로 상태가 바뀔 때만 건드린다."""
        if state == self.state:
            return
        self.state = state
        text, style = LABEL_STYLE[state]
        self.live_label.setText(text)
        self.live_label.setStyleSheet(style)
