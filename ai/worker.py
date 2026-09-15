"""추론을 GUI 스레드 밖에서 돌린다.

왜 필요한가
    추론 한 번은 젯슨 로컬 20ms, 원격(PC 모드) 90ms 쯤 걸린다.
    이걸 GUI 타이머 안에서 부르면 그 시간 동안 화면도 버튼도 전부 멈춘다.
    초당 5번만 돌려도 5번의 끊김이 생기고, 원격이면 초당 0.5초를 멈추는 셈이다.

    그래서 추론 전담 스레드를 하나 두고, GUI 는 "이 프레임 좀 봐줘" 하고 바로 돌아온다.
    결과는 나중에 신호(live_ready)로 온다. 카메라 fps 와 추론 fps 가 완전히 분리된다.

밀린 프레임은 버린다
    대기열은 한 칸뿐이다. 추론하는 동안 새 프레임이 3장 들어오면 앞의 2장은 버리고
    가장 최신 것만 남긴다. 밀린 프레임을 전부 처리하면 결과가 점점 과거가 되고
    부하만 늘어난다. (GStreamer 의 drop=true max-buffers=1 과 같은 생각)

CUDA 컨텍스트
    pycuda 의 컨텍스트는 만든 스레드에만 붙는다. 다른 스레드에서 쓰면
    cuMemAlloc failed: invalid device context 가 난다.
    그래서 추론기를 이 스레드 안에서 만든다. 바깥에서는 절대 직접 만들지 않는다.
"""
import threading
import time

from PyQt5.QtCore import QThread, pyqtSignal

from core.log import get_logger

log = get_logger("worker")


class DetectorWorker(QThread):
    """추론기를 감싼 스레드. 기존 detector 와 같은 방식으로 쓸 수 있다."""

    live_ready = pyqtSignal(object, int, float)   # 검출목록, frame_id, 걸린시간(초)
    failed = pyqtSignal(str)                      # 실시간 추론이 실패했을 때

    def __init__(self, factory, parent=None):
        super().__init__(parent)
        self._factory = factory          # 스레드 안에서 부를 것
        self._det = None
        self._start_error = ""

        self._cv = threading.Condition()
        self._live = None                # (frame, frame_id) — 최신 한 장만
        self._job = None                 # 결과를 기다리는 요청
        self._stop = False
        self._booted = threading.Event()

    # ── 바깥에서 쓰는 것들 ───────────────────────────────────────────────
    def boot(self, timeout=30.0):
        """스레드를 띄우고 추론기가 준비될 때까지 기다린다."""
        self.start()
        self._booted.wait(timeout)
        if self._start_error:
            raise RuntimeError(self._start_error)
        return self._det is not None

    def ready(self):
        return self._det is not None and self._det.ready()

    @property
    def model_path(self):
        return getattr(self._det, "model_path", None)

    @property
    def conf(self):
        return getattr(self._det, "conf", 0.45)

    def load(self, path):
        """모델 교체. 이 스레드에서 처리하고 끝날 때까지 기다린다."""
        return self._call("load", path)

    def refresh(self):
        """서버가 물고 있는 모델을 다시 확인한다. (원격 추론일 때만 의미가 있다)"""
        if not hasattr(self._det, "refresh"):
            return False
        return self._call("refresh")

    def infer(self, frame, draw=True, tta=False):
        """한 번만 확실히 추론한다 (「상품 인식 실행」 버튼용).

        결과를 바로 써야 하므로 끝날 때까지 기다린다. 다만 추론 자체는
        이 스레드에서 도니까 CUDA 컨텍스트 문제는 없다.
        """
        return self._call("infer", frame, draw, tta)

    def submit_live(self, frame, frame_id):
        """실시간 미리보기용. 기다리지 않는다. 앞서 대기 중이던 프레임은 버린다."""
        with self._cv:
            self._live = (frame, frame_id)
            self._cv.notify()

    def drop_live(self):
        """대기 중인 실시간 요청을 취소한다 (탭을 옮겼을 때 등)."""
        with self._cv:
            self._live = None

    def shutdown(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self.wait(3000)

    # ── 내부 ─────────────────────────────────────────────────────────────
    def _call(self, kind, *args, timeout=60.0):
        if not self.isRunning():
            raise RuntimeError("추론 스레드가 실행 중이 아닙니다.")
        done = threading.Event()
        box = {}
        with self._cv:
            self._job = (kind, args, box, done)
            self._cv.notify()
        if not done.wait(timeout):
            raise RuntimeError("추론이 시간 안에 끝나지 않았습니다.")
        if "error" in box:
            raise box["error"]
        return box["value"]

    def _release(self):
        """CUDA 컨텍스트는 만든 스레드에서만 정리할 수 있다. 여기가 그 스레드다."""
        close = getattr(self._det, "close", None)
        if close is not None:
            try:
                close()
            except Exception as e:      # noqa: BLE001 - 종료 중이라 기록만 남긴다
                log.warning("추론기 정리 실패: %s", e)

    def run(self):
        try:
            self._det = self._factory()
        except Exception as e:      # noqa: BLE001 - boot() 가 호출자에게 그대로 알린다
            log.exception("추론기를 만들지 못했습니다")
            self._start_error = str(e)
        finally:
            self._booted.set()

        if self._det is None:
            return

        while True:
            with self._cv:
                while not self._stop and self._job is None and self._live is None:
                    self._cv.wait()
                if self._stop:
                    self._release()
                    return
                # 결과를 기다리는 요청이 우선. 실시간은 언제 돌아도 되는 것이다.
                job, self._job = self._job, None
                live, self._live = self._live, None

            if job is not None:
                kind, args, box, done = job
                try:
                    if kind == "load":
                        box["value"] = self._det.load(*args)
                    elif kind == "refresh":
                        box["value"] = self._det.refresh()
                    else:
                        box["value"] = self._det.infer(*args)
                except Exception as e:      # noqa: BLE001 - 그대로 호출자에게 넘긴다
                    box["error"] = e
                finally:
                    done.set()

            if live is not None:
                frame, frame_id = live
                t0 = time.perf_counter()
                try:
                    _, dets, _ = self._det.infer(frame, draw=False)
                except Exception as e:      # noqa: BLE001 - GUI 가 재시도 간격을 늘린다
                    log.debug("실시간 추론 실패: %s", e)
                    self.failed.emit(str(e))
                    continue
                self.live_ready.emit(dets, frame_id, time.perf_counter() - t0)
