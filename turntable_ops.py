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

_lock = threading.Lock()
_table = None
_error = None
_angle = 0.0            # 시작 자리에서 얼마나 돌아와 있나 (기록용)


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
    except Exception:
        return False


def status():
    return {
        "connected": _table is not None,
        "port": getattr(_table, "port", None),
        "angle": round(_angle, 2),
        "error": _error,
    }


def set_speed(usec):
    with _lock:
        t = _connect()
        t.set_speed(int(usec))
        t.step_us = int(usec)
        return {"step_us": int(usec)}


def rotate(deg):
    """각도만큼 돌리고, 다 돌면 돌아온다."""
    global _angle, _error
    with _lock:
        try:
            t = _connect()
            t.rotate(float(deg))
            _angle += float(deg)
            return {"angle": round(_angle, 2)}
        except Exception as e:
            _error = str(e)
            raise


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
            except Exception:
                pass
            _table = None
        return {"closed": True}
