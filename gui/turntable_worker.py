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

from core.log import get_logger

log = get_logger("turntable")


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
            except (OSError, ValueError, AttributeError):
                msg = ""
            raise RuntimeError(msg or f"HTTP {e.code}")

    def set_speed(self, usec):
        return self._post("/turn/speed", {"step_us": int(usec)})

    def rotate(self, deg):
        return self._post("/turn", {"deg": float(deg)})

    def rotate_async(self, deg):
        """회전판을 돌리기 시작하고 바로 돌아온다 (연속 이동용)."""
        return self._post("/turn/start", {"deg": float(deg)})

    def remaining(self):
        return self._post("/turn/remaining", {})

    def positions(self):
        return self._post("/turn/position", {})

    def stop1(self):
        return self._post("/turn/stop", {})

    def set_speed2(self, usec):
        return self._post("/turn2/speed", {"step_us": int(usec)})

    def rotate2(self, deg):
        """2번(좌우) 모터. 젯슨이 바로 응답하므로 1번이 도는 동안에도 같이 움직인다."""
        return self._post("/turn2", {"deg": float(deg)})

    def stop2(self):
        return self._post("/turn2/stop", {})

    def close(self):
        try:
            self._post("/turn/release", {})
        except (OSError, ValueError, RuntimeError) as e:
            log.warning("젯슨 턴테이블 코일을 끄지 못했습니다 (모터가 뜨거워질 수 있음): %s", e)


class TurntableWorker(QThread):
    arrived = pyqtSignal(float)      # 한 칸 이동을 마쳤다 (현재 각도)
    failed = pyqtSignal(str)
    done = pyqtSignal()

    MIN_STEP_DEG = 0.3               # 이보다 잘게 나누면 모터가 못 따라간다
    MAX_STEP_DEG = 15.0              # 이보다 크면 고정한 박스를 놓치기 쉽다
    SHOT_WAIT = 3.0                  # 한 장 저장되기를 기다리는 최대 시간

    def __init__(self, shots, span=180.0, settle=0.35, step_us=2500, span2=0.0,
                 interval=0.5, smooth=True, parent=None):
        super().__init__(parent)
        self.shots = max(1, int(shots))
        self.span = abs(float(span))
        # 2번(좌우) 모터: 가운데 → 왼쪽 → 오른쪽 → 가운데. 0 이면 안 움직인다.
        self.span2 = abs(float(span2))
        self.interval = max(0.1, float(interval))
        self.smooth = bool(smooth)
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
            log.warning("좌우 모터를 쓸 수 없어 끕니다: %s", e)
            return side

    def side_wait(self, deg):
        """좌우 모터가 그만큼 가는 데 걸리는 시간 (뒤에서 돌기 때문에 기다려 준다)."""
        steps = abs(deg) / 360.0 * self.MOTOR2_STEPS_PER_REV
        return min(20.0, steps * self.MOTOR2_STEP_US / 1e6)

    MIN_STEP_US = 900                # 회전판이 따라올 수 있는 한계
    MIN_STEP_US2 = 2000              # 좌우 모터 한계 (풀스텝 약 15 RPM). 더 빠르면 미끄러진다
    MAX_STEP_US = 300000             # 펌웨어가 허용하는 가장 느린 간격

    def legs(self, span, total_sec):
        """가운데 → 왼쪽 → 오른쪽 → 가운데. (각도, 걸릴 시간) 세 구간."""
        if span <= 0:
            return []
        return [(-span, total_sec * 0.25), (2 * span, total_sec * 0.5), (-span, total_sec * 0.25)]

    def leg_speed(self, deg, seconds, rev=None, floor=None):
        """그 구간을 그 시간에 걸쳐 도는 스텝 간격(us). 느릴수록 부드럽다.

        모터가 따라올 수 없을 만큼 빨라야 하는 경우에는 한계 속도로 돌린다.
        (그러면 촬영이 끝날 때까지 왕복을 다 못 돌고, 남은 만큼은 제자리로 돌아온다)
        """
        steps = max(1.0, abs(deg) / 360.0 * float(rev or self.MOTOR2_STEPS_PER_REV))
        low = self.MIN_STEP_US if floor is None else floor
        return int(min(self.MAX_STEP_US, max(low, seconds / steps * 1e6)))

    def run_smooth(self):
        """끊김 없이: 두 모터를 아주 느리게 계속 돌려 두고 그 위에서 사진만 찍는다.

        한 장마다 돌고 멈추면 화면이 뚝뚝 끊긴다. 촬영 시간 전체(장수 x 간격)에
        걸쳐 한 번 왕복하도록 속도를 맞추면 계속 흐르는 것처럼 보이고,
        사진 한 장(1/30초) 사이의 움직임은 0.1도쯤이라 흔들리지도 않는다.

        세 구간(가운데→왼쪽→오른쪽→가운데)은 속도가 모두 같으므로 속도는 처음에
        한 번만 정한다. 구간을 넘길 때는 시계가 아니라 "모터가 다 갔는지"를 보고
        곧바로 다음 구간을 보낸다. 그래야 방향을 바꾸는 순간 말고는 멈추지 않는다.
        """
        total = max(1.0, self.shots * self.interval)
        legs1, legs2 = self.legs(self.span, total), self.legs(self.span2, total)
        rev1 = rev2 = self.MOTOR2_STEPS_PER_REV
        try:
            here = self.table.positions()
            start1, start2 = here.get("turn", 0.0), here.get("side", 0.0)
            rev1, rev2 = here.get("rev", rev1), here.get("rev2", rev2)
        except Exception as e:              # noqa: BLE001 - 옛 서버는 위치 조회가 없다
            log.warning("시작 위치를 읽지 못해 0도로 가정합니다: %s", e)
            start1 = start2 = 0.0

        # 속도는 한 번만 (세 구간 모두 같은 속도라 다시 보낼 필요가 없다)
        if legs1:
            self.table.set_speed(self.leg_speed(*legs1[0], rev=rev1))
        if legs2 and self._side_ok:
            try:
                self.table.set_speed2(self.leg_speed(*legs2[0], rev=rev2, floor=self.MIN_STEP_US2))
            except Exception as e:
                self._side_ok = False
                log.warning("좌우 모터를 쓸 수 없어 끕니다: %s", e)

        def send(idx1, idx2):
            """각 모터의 다음 구간을 보낸다."""
            if legs1 and idx1 < len(legs1):
                self.table.rotate_async(legs1[idx1][0])
            if legs2 and self._side_ok and idx2 < len(legs2):
                try:
                    self.table.rotate2(legs2[idx2][0])
                except Exception as e:
                    self._side_ok = False
                    log.warning("좌우 모터를 쓸 수 없어 끕니다: %s", e)

        i1, i2 = [0], [0]                  # 지금 몇 번째 구간인지 (뒤 스레드와 같이 본다)
        busy = threading.Event()
        finishing = threading.Event()      # 정리 시작 — 더 이상 다음 구간을 보내지 않는다

        def advance():
            """모터가 다 갔으면 다음 구간을 보낸다. 촬영을 막지 않으려고 따로 돈다."""
            try:
                if finishing.is_set():
                    return
                rem = self.table.remaining()
                if finishing.is_set():
                    return
                if rem.get("turn", 1) == 0 and i1[0] + 1 < len(legs1):
                    i1[0] += 1
                    self.table.rotate_async(legs1[i1[0]][0])
                if rem.get("side", 1) == 0 and self._side_ok and i2[0] + 1 < len(legs2):
                    i2[0] += 1
                    self.table.rotate2(legs2[i2[0]][0])
            except Exception as e:
                log.warning("다음 구간을 보내지 못했습니다: %s", e)
            finally:
                busy.clear()

        send(i1[0], i2[0])
        t0 = time.time()                       # 첫 명령을 보낸 뒤부터 촬영 시계 시작
        for i in range(self.shots):
            if self._stop.is_set():
                break
            target = t0 + i * self.interval
            if time.time() > target + self.interval:      # 많이 밀렸으면 일정을 다시 맞춘다
                t0 = time.time() - i * self.interval      # (몰아서 찍지 않게)
                target = time.time()
            while not self._stop.is_set():
                left = target - time.time()
                if left <= 0:
                    break
                time.sleep(min(0.05, left))
            self._shot.clear()
            self.arrived.emit(float(i))
            self._shot.wait(self.SHOT_WAIT)

            # 다 간 모터가 있으면 곧바로 다음 구간으로.
            # 이 확인·전송은 시리얼 왕복이라 0.5초쯤 걸린다. 촬영 루프에서 직접 하면
            # 그만큼 사진이 밀리므로, 뒤에서 따로 돌린다 (촬영은 제 박자를 지킨다).
            if not busy.is_set() and (i1[0] + 1 < len(legs1)
                                      or (self._side_ok and i2[0] + 1 < len(legs2))):
                busy.set()
                threading.Thread(target=advance, daemon=True).start()
        # 뒤에서 도는 스레드가 구간을 하나 더 보내 버리면 제자리로 못 돌아온다.
        finishing.set()
        for _ in range(60):
            if not busy.is_set():
                break
            time.sleep(0.05)
        self.park(start1, start2)

    def park(self, start1=0.0, start2=0.0):
        """멈추고 처음 자리로 돌려 놓는다.

        보드가 실제로 걸은 걸음 수를 세고 있으므로, 중간에 멈춰도 위치가 정확하다.
        """
        for stop in ("stop1", "stop2"):
            try:
                getattr(self.table, stop)()
            except Exception as e:          # noqa: BLE001 - 하나가 실패해도 나머지는 멈춰야 한다
                log.warning("모터 정지(%s) 실패: %s", stop, e)
        try:
            now = self.table.positions()
        except Exception as e:              # noqa: BLE001
            log.warning("위치를 읽지 못해 제자리로 돌리지 못했습니다: %s", e)
            return
        here1 = now.get("turn", 0.0) - start1
        here2 = now.get("side", 0.0) - start2
        try:
            self.table.set_speed(self.step_us)
            if abs(here2) > 0.2 and self._side_ok:
                self.table.set_speed2(2500)
                self.table.rotate2(-here2)
            if abs(here1) > 0.2:
                self.table.rotate(-here1)              # 다 돌아올 때까지 기다린다
            if abs(here2) > 0.2 and self._side_ok:
                time.sleep(self.side_wait(here2))
        except Exception as e:
            log.warning("제자리로 돌리지 못했습니다: %s", e)

    def _close_table(self):
        try:
            self.table.close()
        except Exception as e:              # noqa: BLE001 - 시리얼/HTTP 어느 쪽이든 닫기만 하면 된다
            log.warning("턴테이블 연결을 닫지 못했습니다: %s", e)

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

        if self.smooth:
            try:
                self.run_smooth()
            except Exception as e:
                self.failed.emit(str(e))
            finally:
                self._close_table()
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
            self._close_table()
            self.done.emit()
