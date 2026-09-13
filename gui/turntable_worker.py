"""수집하는 동안 턴테이블을 조금씩 돌린다.

왜 스레드인가
    motor_serial.Turntable.rotate() 는 다 돌 때까지 돌아오지 않는다.
    GUI 스레드에서 부르면 그동안 화면과 버튼이 통째로 멈춘다.

왜 왕복인가
    카메라 뒤에 케이블이 붙어 있어 한 방향으로 계속 돌릴 수 없다.
    왼쪽으로 span 도 갔다가 오른쪽으로 span 도 돌아오기만 한다.

왜 조금씩인가
    한 번에 많이 돌리면 빵이 화면에서 크게 건너뛰고, 고정해 둔 빨간 박스가
    옆 빵으로 옮겨 붙는다. 목표 장수에 맞춰 한 칸을 잘게 나눈다.
    200장 · 왕복 360도면 한 칸이 1.8도다.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request

from PyQt5.QtCore import QThread, pyqtSignal


class RemoteTable:
    """젯슨에 붙은 턴테이블을 HTTP 로 돌린다.

    모터 보드는 젯슨 USB 에 꽂혀 있고 화면은 PC 에 있다. PC 에서 시리얼을
    열 수 없으므로 inference_server 의 /turn 창구에 "몇 도" 만 보낸다.
    실제 회전이 끝난 뒤에 응답이 오므로, 로컬 Turntable 과 똑같이 블로킹이다.

    180도 되돌리기는 5초쯤 걸린다. 타임아웃을 넉넉히 둔다.
    """

    TIMEOUT = 60.0

    def __init__(self, url):
        self.url = url.rstrip("/")
        self.port = self.url                      # 오류 메시지용
        info = self._get("/health")
        if not info.get("turntable"):
            raise RuntimeError(
                f"{self.url} 에 턴테이블이 없습니다.\n"
                "젯슨에 모터 보드가 꽂혀 있는지, 젯슨 서버가 최신본인지 확인하세요.")

    def _get(self, path):
        with urllib.request.urlopen(f"{self.url}{path}", timeout=10.0) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(self, path, obj):
        data = json.dumps(obj).encode("utf-8")
        req = urllib.request.Request(f"{self.url}{path}", data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode("utf-8")).get("error", "")
            except Exception:
                msg = ""
            raise RuntimeError(msg or f"HTTP {e.code}")

    def set_speed(self, usec):
        return self._post("/turn/speed", {"step_us": int(usec)})

    def rotate(self, deg):
        return self._post("/turn", {"deg": float(deg)})

    def rotate2(self, deg):
        """2번(좌우) 모터. 젯슨이 바로 응답하므로 1번이 도는 동안에도 같이 움직인다."""
        return self._post("/turn2", {"deg": float(deg)})

    def stop2(self):
        return self._post("/turn2/stop", {})

    def close(self):
        try:
            self._post("/turn/release", {})
        except Exception:
            pass


class TurntableWorker(QThread):
    arrived = pyqtSignal(float)      # 한 칸 이동을 마쳤다 (현재 각도)
    failed = pyqtSignal(str)
    done = pyqtSignal()

    MIN_STEP_DEG = 0.3               # 이보다 잘게 나누면 모터가 못 따라간다
    MAX_STEP_DEG = 15.0              # 이보다 크면 고정한 박스를 놓치기 쉽다
    SHOT_WAIT = 3.0                  # 한 장 저장되기를 기다리는 최대 시간

    def __init__(self, shots, span=180.0, settle=0.35, step_us=2500, span2=0.0, parent=None):
        super().__init__(parent)
        self.shots = max(1, int(shots))
        self.span = abs(float(span))
        # 2번(좌우) 모터: 가운데 → 왼쪽 → 오른쪽 → 가운데. 0 이면 안 움직인다.
        self.span2 = abs(float(span2))
        self._side_ok = True
        self.settle = float(settle)
        self.step_us = int(step_us)
        self.table = None
        self._stop = threading.Event()
        self._shot = threading.Event()

    def step_deg(self):
        """한 장당 몇 도 돌 것인가. 왕복 한 바퀴(span x2)에 목표 장수를 나눈다."""
        per = (self.span * 2.0) / self.shots
        return max(self.MIN_STEP_DEG, min(self.MAX_STEP_DEG, per))

    SIDE_MIN_DEG = 0.2               # 이보다 작은 이동은 보내지 않는다
    MOTOR2_STEP_US = 1500            # 펌웨어의 2번 모터 스텝 간격 (되돌아올 시간 계산용)
    MOTOR2_STEPS_PER_REV = 4076.0

    def side_target(self, i):
        """i 번째 장을 찍을 때 좌우 모터가 있어야 할 각도.

        가운데(0) → 왼쪽(-span2) → 오른쪽(+span2) → 가운데(0) 로 한 번 훑는다.
        촬영 내내 조금씩 움직이므로 같은 빵을 여러 각도에서 담게 된다.
        """
        if self.span2 <= 0 or self.shots < 2:
            return 0.0
        t = i / float(self.shots - 1)          # 0 ~ 1
        if t <= 0.25:
            return -self.span2 * (t / 0.25)                       # 가운데 → 왼쪽
        if t <= 0.75:
            return -self.span2 + 2.0 * self.span2 * ((t - 0.25) / 0.5)   # 왼쪽 → 오른쪽
        return self.span2 * (1.0 - (t - 0.75) / 0.25)             # 오른쪽 → 가운데

    def move_side(self, target, side):
        """좌우 모터를 target 으로. 실패하면(옛 젯슨 서버) 조용히 끄고 계속 찍는다."""
        if not self._side_ok or abs(target - side) < self.SIDE_MIN_DEG:
            return side
        try:
            self.table.rotate2(target - side)
            return target
        except Exception as e:
            self._side_ok = False
            print(f"[턴테이블] 좌우 모터를 쓸 수 없어 끕니다: {e}")
            return side

    def side_wait(self, deg):
        """좌우 모터가 그만큼 가는 데 걸리는 시간 (뒤에서 돌기 때문에 기다려 준다)."""
        steps = abs(deg) / 360.0 * self.MOTOR2_STEPS_PER_REV
        return min(20.0, steps * self.MOTOR2_STEP_US / 1e6)

    def shot_done(self):
        """GUI 스레드가 한 장 저장했다고 알려 준다."""
        self._shot.set()

    def stop(self):
        self._stop.set()
        self._shot.set()

    def run(self):
        try:
            # 젯슨에 붙어 있으면 젯슨이 돌린다. PC 에 직접 꽂았으면 시리얼로.
            url = os.environ.get("BREADVISION_INFER_URL", "").strip()
            if url:
                self.table = RemoteTable(url)
            else:
                from motor_serial import Turntable
                self.table = Turntable()
            self.table.set_speed(self.step_us)
        except Exception as e:
            self.failed.emit(str(e))
            self.done.emit()
            return

        step = self.step_deg()
        angle = 0.0
        side = 0.0
        direction = 1
        try:
            for i in range(self.shots):
                if self._stop.is_set():
                    break
                # 좌우 모터를 먼저 보낸다. 보드가 바로 응답하므로 아래 회전과 동시에 움직인다.
                side = self.move_side(self.side_target(i), side)
                # 끝까지 갔으면 방향을 바꾼다. 케이블이 감기지 않게.
                if abs(angle + step * direction) > self.span:
                    direction = -direction
                self.table.rotate(step * direction)
                angle += step * direction

                time.sleep(self.settle)         # 흔들림이 멎기를 기다린다
                if self._stop.is_set():
                    break

                self._shot.clear()
                self.arrived.emit(angle)
                self._shot.wait(self.SHOT_WAIT)

            # 제자리로 돌려 놓는다. 다음 수집이 같은 자리에서 시작하도록.
            if abs(side) > 0.01:
                self.move_side(0.0, side)
                time.sleep(self.side_wait(side))     # 좌우 모터가 가운데로 갈 시간
            if abs(angle) > 0.01:
                self.table.rotate(-angle)
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            try:
                self.table.close()
            except Exception:
                pass
            self.done.emit()
