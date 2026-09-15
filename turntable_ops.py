"""젯슨에 붙은 턴테이블을 PC 대신 돌려 준다.

모터 보드(STM32)는 젯슨 USB 에 꽂혀 있고, 화면·버튼은 PC 에 있다.
그래서 PC 가 시리얼을 직접 열 수 없다. inference_server 가 창구가 되어
PC 의 요청을 받아 여기서 실제로 돌린다.

시리얼은 한 번에 한 명만 써야 한다.
    · 포트를 여는 것도, 명령을 보내는 것도 전부 락 안에서 한다.
    · 보드 연결은 처음 필요할 때 한 번만 한다 (부팅 대기 2초가 있어 느리다).

PC 쪽 gui/turntable_worker.py 가 이 창구를 쓴다.
"""
import threading
import time

from core.log import get_logger

log = get_logger("turntable")

_lock = threading.Lock()
_table = None
_error = None
_angle = 0.0            # 1번(회전판): 시작 자리에서 얼마나 돌아와 있나 (기록용)
_angle2 = 0.0           # 2번(좌우): 가운데에서 얼마나 치우쳐 있나


def _connect():
    """락을 이미 잡은 상태에서만 부른다."""
    global _table, _error
    if _table is not None:
        return _table
    from motor_serial import Turntable
    _table = Turntable()
    _error = None
    return _table


def available():
    """보드가 꽂혀 있는가. 연결까지 하지는 않는다 (2초 걸린다)."""
    if _table is not None:
        return True
    try:
        from motor_serial import Turntable
        return bool(Turntable.candidates())
    except Exception as e:              # noqa: BLE001 - /health 에서 매번 부르므로 서버를 막으면 안 된다
        log.debug("턴테이블 확인 실패: %s", e)
        return False


def status():
    return {
        "connected": _table is not None,
        "port": getattr(_table, "port", None),
        "angle": round(_angle, 2),
        "angle2": round(_angle2, 2),
        "error": _error,
    }


def set_speed(usec):
    with _lock:
        t = _connect()
        t.set_speed(int(usec))
        t.step_us = int(usec)
        return {"step_us": int(usec)}


def rotate(deg):
    """회전판을 각도만큼 돌리고, 다 돌면 돌아온다.

    도는 동안 시리얼 락을 계속 잡고 있으면 그 사이 좌우 모터 명령이 밀린다
    (하나가 움직이면 다른 하나가 멈춘 것처럼 보인다).
    그래서 보드에 "돌기 시작해라"만 시키고 락을 놓은 뒤, 가끔 남은 양만 확인한다.
    두 모터가 완전히 독립적으로 움직인다.
    """
    global _angle, _error
    try:
        with _lock:
            t = _connect()
            t.rotate_async(float(deg))
        limit = time.time() + abs(float(deg)) / 360.0 * 4076 * (t.step_us / 1e6) * 3 + 30
        while time.time() < limit:
            time.sleep(0.05)
            with _lock:
                if t.busy1() == 0:
                    break
        _angle += float(deg)
        return {"angle": round(_angle, 2)}
    except Exception as e:
        _error = str(e)
        raise


def rotate2(deg):
    """2번(좌우) 모터. 보드가 바로 응답하므로 1번이 도는 동안에도 같이 움직인다."""
    global _angle2, _error
    with _lock:
        try:
            t = _connect()
            t.rotate2(float(deg))
            _angle2 += float(deg)
            return {"angle2": round(_angle2, 2)}
        except Exception as e:
            _error = str(e)
            raise


def set_speed2(usec):
    """2번 모터 속도. 간격이 클수록 느리고 힘이 세다."""
    with _lock:
        t = _connect()
        t.set_speed2(int(usec))
        return {"step_us2": int(usec)}


def set_mode2(half):
    """2번 모터 방식. half=False 면 FULL (힘이 세다)."""
    with _lock:
        t = _connect()
        t.set_mode2(bool(half))
        return {"half2": bool(half)}


def set_order2(order):
    """2번 모터 코일 순서 (예: "1324")."""
    with _lock:
        t = _connect()
        t.set_order2(str(order))
        return {"order2": str(order)}


def set_hold2(hold):
    """멈춘 뒤에도 2번 모터가 자리를 붙잡게 할지."""
    with _lock:
        t = _connect()
        t.set_hold2(bool(hold))
        return {"hold2": bool(hold)}


def test_coils2():
    """코일을 하나씩 켠다. 드라이버 LED 4개가 차례로 켜지면 배선 정상."""
    with _lock:
        t = _connect()
        t.test_coils2()
        return {"tested": True}


def rotate_start(deg):
    """회전판을 돌리기 시작하고 바로 돌아온다 (연속 이동용)."""
    global _angle
    with _lock:
        t = _connect()
        t.rotate_async(float(deg))
        _angle += float(deg)
        return {"angle": round(_angle, 2)}


def positions():
    """보드가 센 실제 위치(도). 명령이 중간에 멈춰도 정확하다."""
    with _lock:
        t = _connect()
        p = t.positions()
        return {"turn": round(p["turn"], 2), "side": round(p["side"], 2),
                "rev": p.get("rev", 4076.0), "rev2": p.get("rev2", 4076.0)}


def reset_positions():
    """지금 자리를 0 도로 삼는다."""
    with _lock:
        t = _connect()
        t.reset_positions()
        return {"turn": 0.0, "side": 0.0}


def remaining():
    """두 모터에 남은 스텝 수. 0 이면 멈춰 있다."""
    with _lock:
        t = _connect()
        return {"turn": t.busy1(), "side": t.busy2()}


def stop1():
    """회전판을 멈춘다."""
    with _lock:
        if _table is None:
            return {"stopped": False}
        _table.stop1()
        return {"stopped": True}


def stop2():
    """2번 모터를 멈춘다."""
    with _lock:
        if _table is None:
            return {"stopped": False}
        _table.stop2()
        return {"stopped": True}


def release():
    """코일을 끈다. 멈춘 채로 켜 두면 모터가 뜨거워진다."""
    with _lock:
        if _table is None:
            return {"released": False}
        _table.release()
        return {"released": True}


def reset_angle():
    global _angle
    with _lock:
        _angle = 0.0
        return {"angle": 0.0}


def close():
    global _table
    with _lock:
        if _table is not None:
            try:
                _table.close()
            except Exception as e:      # noqa: BLE001
                log.warning("턴테이블 연결을 닫지 못했습니다: %s", e)
            _table = None
        return {"closed": True}
