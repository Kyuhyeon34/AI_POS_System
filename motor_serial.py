"""턴테이블(STM32) 을 USB 시리얼로 조종한다.

젯슨 GPIO 대신 STM32 를 쓰는 이유
    · 젯슨 40핀 헤더에서 전압이 나오지 않았다 (멀티미터로 0V 확인)
    · 리눅스는 멀티태스킹이라 파이썬으로 1.5ms 간격을 맞추면
      추론·스트리밍 부하가 걸릴 때 스텝이 밀린다. STM32 는 그럴 일이 없다

쓰는 법
    python3 motor_serial.py --find          연결된 보드 찾기
    python3 motor_serial.py --deg 90        90도 회전
    python3 motor_serial.py --shoot 12      촬영 흐름 흉내 (12등분)
    python3 motor_serial.py --release       코일 끄기

모터가 웅웅대기만 하고 안 돌 때
    python3 motor_serial.py --try-orders    핀 순서를 바꿔가며 시험 (도는 순서 찾기)
    python3 motor_serial.py --order 2143    찾은 순서로 고정
    python3 motor_serial.py --torque        힘 센 방식(2상)으로 바꿔 시험
    python3 motor_serial.py --slow 16       한 스텝씩 아주 느리게 (LED 순서를 눈으로 확인)

코드에서
    from motor_serial import Turntable
    with Turntable() as t:
        t.rotate(90)        # 다 돌 때까지 기다린 뒤 돌아온다
"""
import argparse
import glob
import sys
import time

from core.log import get_logger

log = get_logger("motor")

BAUD = 115200
# 보드가 리셋된 뒤 부팅될 시간. 이걸 안 기다리면 첫 명령이 씹힌다.
BOOT_WAIT = 2.0


class Turntable:
    def __init__(self, port=None, timeout=30.0):
        try:
            import serial
        except ImportError:
            raise SystemExit("pyserial 이 없습니다:  pip3 install --user pyserial")
        self._serial = serial
        self.port = port or self.find_port()
        if not self.port:
            raise RuntimeError(
                "턴테이블 보드를 찾지 못했습니다.\n"
                "  · USB 케이블이 꽂혀 있는지\n"
                "  · ls /dev/ttyACM* 로 장치가 보이는지\n"
                "  · 안 보이면 사용자를 dialout 그룹에 넣으세요:\n"
                "      sudo usermod -aG dialout $USER   (다시 로그인 필요)")
        self.ser = serial.Serial(self.port, BAUD, timeout=timeout)
        self.base_timeout = float(timeout)
        time.sleep(BOOT_WAIT)
        self._drain()
        # 보드에서 현재 설정을 읽어 둔다 (오래 걸리는 이동의 대기 시간 계산용)
        self.step_us, self.steps_per_rev = 1500, 4076
        try:
            parts = self.info().split()
            self.step_us = int(parts[1])
            self.steps_per_rev = int(parts[2])
        except Exception as e:          # noqa: BLE001 - 옛 펌웨어는 ? 응답 형식이 다르다
            log.warning("보드 설정을 읽지 못해 기본값(1500us, 4076스텝)으로 계산합니다: %s", e)
        if self.step_us > 6000:
            # 앞 실행에서 느리게 바꿔 둔 값이 남아 있으면 한 바퀴에 몇 분씩 걸린다.
            # 기본으로 되돌리고 알려 준다.
            print(f"[턴테이블] 스텝 간격이 {self.step_us}us 로 남아 있어 1500us 로 되돌립니다.")
            try:
                self.set_speed(1500)
                self.step_us = 1500
            except Exception as e:      # noqa: BLE001
                log.warning("스텝 간격을 1500us 로 되돌리지 못했습니다 — 회전이 매우 느릴 수 있습니다: %s", e)

    # ── 연결 ────────────────────────────────────────────────────────
    @staticmethod
    def candidates():
        # Nucleo 의 ST-Link 가상 COM 포트는 ttyACM, USB-TTL 변환기는 ttyUSB 로 잡힌다
        return sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))

    @classmethod
    def find_port(cls):
        """P 명령에 제대로 답하는 포트를 고른다. (다른 USB 시리얼 장치와 섞이지 않게)"""
        import serial
        for port in cls.candidates():
            try:
                with serial.Serial(port, BAUD, timeout=1.0) as s:
                    time.sleep(BOOT_WAIT)
                    s.reset_input_buffer()
                    s.write(b"P\n")
                    reply = s.readline().decode("utf-8", "replace").strip()
                    if reply.startswith("OK TURNTABLE"):
                        return port
            except (OSError, serial.SerialException):
                continue
        return None

    # ── 명령 ────────────────────────────────────────────────────────
    def _expected_seconds(self, text):
        """이 명령이 얼마나 걸릴지. 10바퀴처럼 오래 도는 동안 기다리려면 필요하다."""
        try:
            if text[0] in "Rr":
                steps = abs(float(text[1:])) / 360.0 * self.steps_per_rev
            elif text[0] in "Ss":
                steps = abs(int(text[1:]))
            else:
                return 0.0
        except ValueError:
            return 0.0
        return steps * self.step_us / 1e6

    def _drain(self, quiet=0.15, limit=5.0):
        """받는 줄이 없을 때까지 비운다.

        긴 회전 중에 Ctrl+C 로 끊으면 보드는 그 이동을 마치고 뒤늦게 OK 를 보낸다.
        그걸 안 치우면 다음 명령의 답으로 잘못 읽혀서 응답이 한 칸씩 밀린다.
        """
        end = time.perf_counter() + limit
        old = self.ser.timeout
        self.ser.timeout = quiet
        try:
            while time.perf_counter() < end:
                if not self.ser.read(4096):
                    break
        finally:
            self.ser.timeout = old

    def _cmd(self, text):
        # 이동 명령이면 걸릴 시간만큼 넉넉히 기다린다 (기본 30초로는 10바퀴를 못 기다린다)
        need = self._expected_seconds(text) * 1.5 + 10.0
        self.ser.timeout = max(self.base_timeout, need)
        self._drain()
        self.ser.write((text + "\n").encode())
        reply = self.ser.readline().decode("utf-8", "replace").strip()
        if not reply:
            raise RuntimeError(f"보드가 응답하지 않습니다: {text}")
        if reply.startswith("ERR"):
            raise RuntimeError(reply)
        return reply

    def ping(self):
        return self._cmd("P")

    def rotate(self, deg):
        """각도만큼 돌린다. 다 돌 때까지 기다렸다가 돌아온다."""
        return self._cmd(f"R{deg}")

    def rotate_async(self, deg):
        """회전판을 몇 도. 다 돌기를 기다리지 않고 바로 돌아온다.
        그동안 좌우 모터 명령을 따로 보낼 수 있다."""
        return self._cmd(f"D{deg}")

    def busy1(self):
        """회전판에 남은 스텝 수 (0 이면 멈춰 있다)."""
        try:
            return int(str(self._cmd("Y1")).split()[-1])
        except Exception:
            return 0

    def step(self, n):
        return self._cmd(f"S{int(n)}")

    def set_speed(self, usec):
        """스텝 간격(마이크로초). 작을수록 빠르지만 800 아래는 탈조가 난다."""
        return self._cmd(f"V{int(usec)}")

    def set_order(self, order):
        """핀 순서. 28BYJ-48 은 IN 순서가 보드마다 달라서 안 맞으면 웅웅대기만 한다."""
        return self._cmd(f"O{order}")

    def set_mode(self, half):
        """half=False 면 2상 여자(힘 셈, 2038스텝), True 면 하프스텝(4076스텝)."""
        return self._cmd("M1" if half else "M0")

    def test_coils(self):
        """코일을 하나씩 켠다. ULN2003 의 LED 4개가 차례로 켜져야 한다."""
        return self._cmd("T")

    def rotate2(self, deg):
        """2번(좌우) 모터를 몇 도. 보드가 바로 응답하고 뒤에서 돌리므로
        1번 모터 회전과 동시에 움직인다."""
        return self._cmd(f"W{deg}")

    def set_speed2(self, usec):
        """2번 모터 스텝 간격(us). 느릴수록 힘이 세다."""
        return self._cmd(f"U{int(usec)}")

    def set_hold2(self, hold):
        """멈춘 뒤에도 2번 모터 코일을 켜 둘지. 켜 두면 자리를 붙잡지만 뜨거워진다."""
        return self._cmd("H1" if hold else "H0")

    def set_mode2(self, half):
        """2번 모터 방식. False = FULL(힘 셈), True = HALF(부드러움)."""
        return self._cmd("N1" if half else "N0")

    def set_order2(self, order):
        """2번 모터 코일 순서 (예: "1324"). 배선 순서가 다를 때 바꾼다."""
        return self._cmd(f"Q{order}")

    def test_coils2(self):
        """2번 모터 코일 점검 (드라이버 LED 4개가 차례로 켜진다)."""
        return self._cmd("G")

    def positions(self):
        """출발점에서 두 모터가 몇 도 가 있는지. {"turn": 도, "side": 도}"""
        try:
            p = str(self._cmd("K")).split()
            s1, s2, rev1, rev2 = int(p[1]), int(p[2]), float(p[3]), float(p[4])
            return {"turn": s1 / rev1 * 360.0, "side": s2 / rev2 * 360.0,
                    "rev": rev1, "rev2": rev2}
        except Exception:
            return {"turn": 0.0, "side": 0.0, "rev": 4076.0, "rev2": 4076.0}

    def reset_positions(self):
        """지금 자리를 0 으로 삼는다 (버튼으로 몇 도 돌렸는지 잴 때)."""
        return self._cmd("K0")

    def remaining(self):
        """두 모터에 남은 스텝. {"turn": .., "side": ..}"""
        return {"turn": self.busy1(), "side": self.busy2()}

    def stop1(self):
        """회전판을 멈추고 코일을 끈다."""
        return self._cmd("X1")

    def stop2(self):
        """2번 모터를 멈추고 코일을 끈다."""
        return self._cmd("X2")

    def busy2(self):
        """2번 모터에 남은 스텝 수 (0 이면 멈춰 있다)."""
        try:
            return int(str(self._cmd("Y")).split()[-1])
        except Exception:
            return 0

    def release(self):
        """코일을 끈다. 멈춘 채로 켜 두면 모터가 뜨거워진다."""
        return self._cmd("Z")

    def info(self):
        return self._cmd("?")

    def close(self):
        try:
            self.release()
        except Exception as e:          # noqa: BLE001 - 포트는 아래에서 반드시 닫는다
            log.warning("코일을 끄지 못했습니다 (모터가 뜨거워질 수 있음): %s", e)
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def main():
    ap = argparse.ArgumentParser(description="턴테이블(STM32) 시험")
    ap.add_argument("--port", help="예: /dev/ttyACM0 (비우면 자동 탐색)")
    ap.add_argument("--find", action="store_true", help="연결된 보드 찾기")
    ap.add_argument("--deg", type=float, help="이 각도만큼 회전")
    ap.add_argument("--steps", type=int, help="하프스텝만큼 회전")
    ap.add_argument("--speed", type=int, metavar="US", help="스텝 간격 설정(마이크로초)")
    ap.add_argument("--turn", action="store_true", help="한 바퀴 + 시간 측정")
    ap.add_argument("--turns", type=int, metavar="N", help="N 바퀴 연속으로 돌린다")
    ap.add_argument("--wiggle", nargs="?", type=float, const=180.0, metavar="DEG",
                    help="좌우로 크게 왕복 (기본 180도). 도는지 눈으로 보기 좋다")
    ap.add_argument("--times", type=int, default=3, help="--wiggle 왕복 횟수")
    ap.add_argument("--shoot", type=int, metavar="N", help="촬영 흐름 흉내 (N 등분)")
    ap.add_argument("--settle", type=float, default=0.4, help="멈춘 뒤 기다릴 시간(초)")
    ap.add_argument("--release", action="store_true", help="코일 끄기")
    ap.add_argument("--order", help="핀 순서 고정 (예: 2143)")
    ap.add_argument("--try-orders", action="store_true",
                    help="핀 순서를 바꿔가며 90도씩 돌려 본다 (도는 순서 찾기)")
    ap.add_argument("--torque", action="store_true",
                    help="힘 센 2상 여자로 바꾼다 (한 바퀴 2038스텝)")
    ap.add_argument("--coils", action="store_true", help="코일 하나씩 켜보기 (LED 확인)")
    ap.add_argument("--feel", action="store_true",
                    help="코일을 하나씩 켜 두고 축을 만져 확인 (LED 가 안 켜질 때)")
    ap.add_argument("--diag", action="store_true",
                    help="한 번에 진단 — LED·소리·움직임을 순서대로 확인")
    ap.add_argument("--slow", type=int, metavar="N",
                    help="N 스텝을 아주 느리게 (LED 가 순서대로 바뀌는지 눈으로 확인)")
    args = ap.parse_args()

    if args.find:
        print("후보 장치:", Turntable.candidates() or "(없음)")
        port = Turntable.find_port()
        print("턴테이블 보드:", port or "찾지 못함")
        return

    if not any([args.deg is not None, args.steps is not None, args.speed,
                args.turn, args.turns, args.wiggle, args.shoot, args.release, args.order,
                args.try_orders, args.torque, args.coils, args.slow, args.diag,
                args.feel]):
        ap.print_help()
        print("\n처음이면:  python3 motor_serial.py --find")
        return

    with Turntable(args.port) as t:
        print(f"연결: {t.port}  |  {t.ping()}")

        if args.feel:
            print("\n코일을 하나씩 켜 둡니다. 각 단계에서 **축을 손으로 돌려보세요.**")
            print("  뻑뻑하게 잡고 있으면 그 코일은 살아 있는 것입니다.")
            print("  LED 는 보드에 달린 거라, LED 가 안 켜져도 코일은 멀쩡할 수 있습니다.\n")
            result = {}
            for n in (1, 2, 3, 4):
                t._cmd(f"C{n}")
                ans = input(f"  코일 {n} 켬 — 축이 뻑뻑한가요? (y/n) ").strip().lower()
                result[n] = ans.startswith("y")
            t._cmd("C0")
            dead = [n for n, ok in result.items() if not ok]
            print()
            if not dead:
                print("  코일 4개 모두 정상입니다. LED 만 나간 것이니 그냥 쓰셔도 됩니다.")
            else:
                print(f"  코일 {dead} 이(가) 힘을 못 냅니다.")
                print("  · 그 IN 선이 Nucleo 에 제대로 꽂혔는지 (IN1=D2, IN2=D3, IN3=D4, IN4=D5)")
                print("  · 점퍼선을 다른 것으로 바꿔 보세요")

        if args.diag:
            print(f"\n현재 설정: {t.info()}")
            print("   (간격us / 한바퀴스텝 / 누적 / 순서 / 방식)\n")

            print("[1] 코일을 하나씩 켭니다 (약 3초)")
            print("    → ULN2003 의 LED 4개가 차례로 켜지나요?")
            t.test_coils()
            input("    엔터를 누르면 다음 …")

            print("\n[2] 한 코일을 3초간 켜 둡니다")
            print("    → 축을 손으로 돌려보세요. 뻑뻑하게 잡고 있어야 정상입니다")
            t._cmd("O1234"); t.set_speed(40000); t.step(1)
            time.sleep(3)
            t.release()
            input("    엔터를 누르면 다음 …")

            print("\n[3] 아주 느리게 16스텝 (한 스텝 0.4초)")
            print("    → LED 가 규칙적으로 옮겨 다니나요? 축이 조금씩 움직이나요?")
            t._cmd("O2143")
            for i in range(16):
                t.step(1)
                time.sleep(0.4)
            t.release()
            t.set_speed(1500)

            print("\n확인해 주세요:")
            print("  · [1] LED 4개가 다 켜졌나?      → 아니면 그 선이 빠진 것")
            print("  · [2] 축이 뻑뻑하게 잡혔나?      → 아니면 모터 커넥터/전원 문제")
            print("  · [3] LED 는 바뀌는데 축은 안 도나? → 모터 자체 또는 커넥터")
            print("  · 소리는? 웅웅대나, 아무 소리도 안 나나?")

        if args.coils:
            print("\n코일을 하나씩 켭니다 — ULN2003 의 LED 4개가 차례로 켜져야 합니다.")
            t.test_coils()
            print("  안 켜지는 게 있으면 그 선이 안 물린 것입니다 (순서 문제가 아님).")

        if args.torque:
            print(" ", t.set_mode(half=False), "— 2상 여자 (힘 셈)")

        if args.order:
            print(" ", t.set_order(args.order))

        if args.try_orders:
            # 24가지 중 실제로 쓰이는 것들만. 위에서부터 흔한 순서.
            orders = ["1234", "1324", "2143", "1243", "1342", "1423",
                      "2134", "3124", "4321", "3412"]
            print("\n순서를 바꿔가며 90도씩 돌립니다. **도는 순서를 기억하세요.**")
            print("(Ctrl+C 로 중단)\n")
            for o in orders:
                t.set_order(o)
                t.rotate(90)
                print(f"  순서 {o} → 90도 명령 보냈습니다. 돌았나요?")
                time.sleep(2.0)
            print("\n도는 순서를 찾으면 그걸로 고정하세요:")
            print("  python3 motor_serial.py --order 2143")

        if args.slow:
            # 스텝 간격을 최대(40ms)로 두고 한 스텝씩 보낸다.
            # LED 4개가 규칙적으로 옮겨 다니면 전기 쪽은 정상이다.
            print(f"\n{args.slow} 스텝을 아주 느리게 보냅니다 (한 스텝에 약 0.3초).")
            print("  · ULN2003 의 LED 가 규칙적으로 옮겨 다니나요?")
            print("  · 축이 조금씩이라도 움직이나요?\n")
            t.set_speed(40000)
            for i in range(args.slow):
                t.step(1)
                print(f"    {i+1}/{args.slow}")
                time.sleep(0.3)
            t.release()
            print("\n  LED 는 바뀌는데 축이 안 움직이면 → 모터 커넥터나 모터 자체 문제")
            print("  LED 도 안 바뀌면 → 배선이나 펌웨어 문제")

        if args.speed:
            print(" ", t.set_speed(args.speed))
        if args.release:
            t.release(); print("  코일 껐습니다.")
        if args.steps is not None:
            t.step(args.steps); print(f"  {args.steps} 스텝 완료")
        if args.deg is not None:
            t0 = time.perf_counter()
            t.rotate(args.deg)
            print(f"  {args.deg}도 완료 ({time.perf_counter()-t0:.1f}초)")
        if args.wiggle:
            deg = args.wiggle
            print(f"\n좌우로 {deg:.0f}도씩 {args.times}번 왕복합니다.\n")
            for i in range(args.times):
                t0 = time.perf_counter()
                t.rotate(deg)
                print(f"    {i+1}/{args.times}  →  +{deg:.0f}도  ({time.perf_counter()-t0:.1f}초)")
                time.sleep(0.6)
                t0 = time.perf_counter()
                t.rotate(-deg)
                print(f"    {i+1}/{args.times}  ←  -{deg:.0f}도  ({time.perf_counter()-t0:.1f}초)")
                time.sleep(0.6)
            print("\n  제자리로 돌아왔나요? 축에 표시를 붙여두면 확인하기 쉽습니다.")

        if args.turns:
            print(f"\n{args.turns} 바퀴 연속으로 돌립니다. "
                  f"(예상 {args.turns * t.steps_per_rev * t.step_us / 1e6:.0f}초)")
            print("  축에 표시를 붙여 두고 보세요. Ctrl+C 로 중단.\n")
            t0 = time.perf_counter()
            for i in range(args.turns):
                t.rotate(360)
                print(f"    {i+1}/{args.turns} 바퀴  ({time.perf_counter()-t0:.0f}초)")
            d = time.perf_counter() - t0
            print(f"\n  {args.turns}바퀴 {d:.0f}초 → 약 {args.turns*60/d:.1f} RPM")

        if args.turn:
            t0 = time.perf_counter()
            t.rotate(360)
            d = time.perf_counter() - t0
            print(f"  한 바퀴 {d:.1f}초 → 약 {60/d:.1f} RPM")
            print("  실제로 정확히 한 바퀴 돌았는지 눈으로 확인하세요.")
        if args.shoot:
            n = args.shoot
            deg = 360.0 / n
            print(f"  {n}등분 ({deg:.1f}도씩), 멈춘 뒤 {args.settle}초 대기\n")
            t0 = time.perf_counter()
            for i in range(n):
                t.rotate(deg)
                time.sleep(args.settle)      # 흔들림·자동노출이 안정될 시간
                print(f"    {i+1:2d}/{n}  {deg*(i+1):6.1f}도 — 여기서 촬영")
            print(f"\n  한 바퀴 {time.perf_counter()-t0:.1f}초, {n}장")
        print(" ", t.info())


if __name__ == "__main__":
    main()
